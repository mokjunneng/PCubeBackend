import sys, traceback
import cv2
import numpy as np
from scipy.ndimage.filters import median_filter
import string

from utils.undistort import undistort_img


def hsv_custom_range_threshold(img, hsv_lower_thresh, hsv_upper_thresh):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Separate channels
    hue, saturation, value = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]

    # Make mask for each channel
    h_mask = cv2.inRange(hue, hsv_lower_thresh[0], hsv_upper_thresh[0])
    s_mask = cv2.inRange(saturation, hsv_lower_thresh[1], hsv_upper_thresh[1])
    v_mask = cv2.inRange(value, hsv_lower_thresh[2], hsv_upper_thresh[2])

    # Apply masks
    masked_img = cv2.bitwise_and(img, img, mask=h_mask)
    masked_img = cv2.bitwise_and(masked_img, masked_img, mask=s_mask)
    masked_img = cv2.bitwise_and(masked_img, masked_img, mask=v_mask)

    # Combine HSV masks
    mask = cv2.bitwise_and(h_mask, s_mask)
    mask = cv2.bitwise_and(mask, v_mask)
    return mask, masked_img

def rgb_custom_range_threshold(img, rgb_lower_thresh, rgb_upper_thresh):
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    # Make mask for each channel
    b_mask = cv2.inRange(b, rgb_lower_thresh[0], rgb_upper_thresh[0])
    g_mask = cv2.inRange(g, rgb_lower_thresh[1], rgb_upper_thresh[1])
    r_mask = cv2.inRange(r, rgb_lower_thresh[2], rgb_upper_thresh[2])

    # Apply masks
    masked_img = cv2.bitwise_and(img, img, mask=b_mask)
    masked_img = cv2.bitwise_and(masked_img, masked_img, mask=g_mask)
    masked_img = cv2.bitwise_and(masked_img, masked_img, mask=r_mask)

    # Combine RGB masks
    mask = cv2.bitwise_and(b_mask, g_mask)
    mask = cv2.bitwise_and(mask, r_mask)
    return mask, masked_img

def pseudo_surface_area(mask):
    """
    Args:
        mask: single channel black and white image (0 - black, 255 - white)
    Returns:
        total_pixels: total white pixels (object of interest)
        pixel_percentage: white pixels over total pixels
    """
    total_pixels = (mask == 255).sum()
    pixel_percentage = (mask == 255).mean()
    return total_pixels, pixel_percentage

def color_analysis(img, mask):
    masked = cv2.bitwise_and(img, img, mask=mask)
    b, g, r = cv2.split(masked)
    lab = cv2.cvtColor(masked, cv2.COLOR_BGR2LAB)
    l, m, y = cv2.split(lab)
    hsv = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    channels = {"b": b, "g": g, "r": r, "l": l, "m": m, "y": y, "h": h, "s": s, "v": v}
    histogram_bins = {}
    for channel, channel_arr in channels.items():
        histogram_bins[channel] = cv2.calcHist([channel_arr], [0], mask, [256], [0, 255])

    return histogram_bins

def undistort_image(img_path):
    img = cv2.imread(img_path)
    img = undistort_img(img)
    cv2.imwrite(img_path, img)
