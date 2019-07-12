#Here is a simple program that displays the camera feed in a cv2.namedWindow and will take an snapshot when you hit SPACE. It will also quit if you hit ESC.

import cv2
import os.path
import os

width = 1920
height = 1080
camera_name ='fishlens2'

cam = cv2.VideoCapture(0)
cam.set(3, width)
cam.set(4, height)

cv2.namedWindow("test")

img_counter = 0
path = './output/' + camera_name

if not os.path.exists(path):
    os.makedirs(path)

while(True):
    # Capture frame-by-frame
    ret, frame = cam.read()
    cv2.imshow("test", cv2.flip(frame, 1))

    k = cv2.waitKey(1)
    
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = os.path.join(path, "{}_{}.png".format(camera_name, img_counter))
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()