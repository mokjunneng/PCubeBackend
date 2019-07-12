import argparse
from datetime import datetime, timedelta, date
import os
import shutil
import threading
import time

import numpy as np
import cv2
from scipy.ndimage.filters import median_filter

from utils.s3helper import get_images_key, download_images, get_latest_image_key
from utils.growth import calculate_growth_from_surface_area
import algorithms.image_processing as image_processing # import hsv_custom_range_threshold, rgb_custom_range_threshold, pseudo_surface_area, color_analysis, undistort_image


TEMP_IMAGE_STORAGE_DIR = os.path.join("..",  "results", "tmp")
PROCESSED_IMAGE_DIR = os.path.join("..",  "results", "output-images")
COLOR_BINS_DIR = os.path.join("..",  "results", "color-bins")
IMAGE_STORAGE_DIR = os.path.join("..",  "results", "stored")
SURFACE_AREA_RESULTS_FILE = os.path.join("..",  "results", "surface_area_results.txt")
GROWTH_FILE = os.path.join("..", "results", "growth.npy")
GROWTH_CURVE_PLOT_FILE = os.path.join("..", "results", "growth_curve.png")
DEFAULT_DURATION = 7

def options():
    parser = argparse.ArgumentParser(description="Imaging processing with opencv")
    parser.add_argument(
        "-d",
        "--startdate",
        help="Get images since [startdate]. Date format: %Y-%m-%d-%H",
        required=True
    )
    parser.add_argument(
        "-e",
        "--enddate",
        help="Get images until [enddate]. Date format: %Y-%m-%d-%H",
        required=True
    )
    parser.add_argument(
        "-s",
        "--stereo",
        help="If True, return images from both cameras, else only return images from camera 2",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "-t",
        "--thread_count",
        help="Set number of worker threads.",
        default=5,
        type=int
    )
    args = parser.parse_args()
    return args

args = options()

class ImageProcessingThread (threading.Thread):
    def __init__(self, threadID, name, images):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.images = images

    def run(self):
        print(f"Starting {self.name}")
        start = time.time()
        for image in self.images:
            image_processing.undistort_image(os.path.join(TEMP_IMAGE_STORAGE_DIR, image))
            run_image_processing_workflow(image)
        print(f"Exiting {self.name}")
        end = time.time()
        running_duration = end - start
        print(f"{self.name} ran for {running_duration}s")

def run_image_processing_workflow(image):
    tmp_img = os.path.join(TEMP_IMAGE_STORAGE_DIR, image)
    img = cv2.imread(tmp_img)

    # Obtain plant mask
    hsv_lower_thresh = [25, 35, 35]
    hsv_upper_thresh = [60, 255, 255]

    rgb_lower_thresh = [0, 100, 0]
    rgb_upper_thresh = [200, 255, 200]

    hsv_mask, hsv_masked_img = image_processing.hsv_custom_range_threshold(img, hsv_lower_thresh, hsv_upper_thresh)
    rgb_mask, rgb_masked_img = image_processing.rgb_custom_range_threshold(img, rgb_lower_thresh, rgb_upper_thresh)

    combined_mask = cv2.bitwise_and(hsv_mask, rgb_mask)
    denoised_mask = median_filter(combined_mask, size=12)

    # Obtain "surface area" information
    surface_area, area_percentage = image_processing.pseudo_surface_area(denoised_mask)

    # Obtain color channels histogram bins
    histogram_bins = image_processing.color_analysis(img, denoised_mask)

    # Write results
    outfile = os.path.join(PROCESSED_IMAGE_DIR, image.split(".")[0] + "_mask.png")
    print(f"Writing image file {outfile}")
    cv2.imwrite(outfile, denoised_mask)
    with open(SURFACE_AREA_RESULTS_FILE, "a") as f:
        f.write(f"{outfile} {surface_area} {area_percentage} \n")
    np.save(os.path.join(COLOR_BINS_DIR, image.split(".")[0] + "_colorbins.npy"), histogram_bins)

    # Move tmp image file to stored dir
    stored_img = os.path.join(IMAGE_STORAGE_DIR, image)
    if not os.path.exists(stored_img):
        shutil.move(tmp_img, stored_img)
    else:
        os.remove(tmp_img)

def run(start_date, end_date, stereo=False, thread_count=5):
    # Remove files
    if os.path.exists(SURFACE_AREA_RESULTS_FILE):
        os.remove(SURFACE_AREA_RESULTS_FILE)

    # Create directories if not exist
    if not os.path.exists(PROCESSED_IMAGE_DIR):
        os.makedirs(PROCESSED_IMAGE_DIR)
    if not os.path.exists(COLOR_BINS_DIR):
        os.makedirs(COLOR_BINS_DIR)
    if not os.path.exists(IMAGE_STORAGE_DIR):
        os.makedirs(IMAGE_STORAGE_DIR)
    if not os.path.exists(TEMP_IMAGE_STORAGE_DIR):
        os.makedirs(TEMP_IMAGE_STORAGE_DIR)

    image_keys = get_images_key(start_date=start_date, end_date=end_date, stereo=stereo)
    download_images(image_keys)

    # Run image processing workflow
    thread_count = thread_count # int(args.thread_count)
    image_files = os.listdir(TEMP_IMAGE_STORAGE_DIR)
    image_segment_length = int(len(image_files) / thread_count)
    threads = []
    for count in range(thread_count):
        if count + 1 == thread_count:
            thread = ImageProcessingThread(count+1, f"Thread{count+1}", image_files[count*image_segment_length:])
        else:
            thread = ImageProcessingThread(count+1, f"Thread{count+1}", image_files[count*image_segment_length:(count+1)*image_segment_length])
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()

    # Calculate growth by gradient of the linear regression line
    calculate_growth_from_surface_area(SURFACE_AREA_RESULTS_FILE, GROWTH_FILE, GROWTH_CURVE_PLOT_FILE)

    print("Image processing done.")

# Main Worflow
def main():
    if not args.startdate and not args.enddate:
        print("Please provide arguments for [-s start_date] and [-e end_date]")
        return

    # Download images from S3
    start_date = datetime.strptime(args.startdate, '%Y-%m-%d-%H').timestamp()
    end_date = datetime.strptime(args.enddate, '%Y-%m-%d-%H').timestamp()
    
    run(start_date, end_date)

if __name__ == "__main__":
    main()
