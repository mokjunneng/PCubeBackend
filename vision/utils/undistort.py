# You should replace these 3 lines with the output in calibration step
import numpy as np
from PIL import Image
import cv2
import sys

# #Fishlens1
#DEPRECATED
# DIM=(1920, 1080)
# K=np.array([[787.309819741564, 0.0, 956.0785528821714], [0.0, 790.3435902112437, 498.0626065655846], [0.0, 0.0, 1.0]])
# D=np.array([[-0.033511329099994866], [-0.02714740152337251], [0.03055781916222119], [-0.012223281393521065]])

#Fishlens2
# #DEPRECATED
# DIM=(1920, 1080)
# K=np.array([[933.0882155883498, 0.0, 1000.2752543861673], [0.0, 933.326327967666, 513.0966829950255], [0.0, 0.0, 1.0]])
# D=np.array([[-0.05501445742824531], [-0.04691908489784053], [0.10442307834908801], [-0.06724079789902383]])

# #Fishlens1_1
# DIM=(1920, 1080)
# K=np.array([[882.5610945176084, 0.0, 960.4176578273838], [0.0, 883.2088503984525, 504.6787529224246], [0.0, 0.0, 1.0]])
# D=np.array([[-0.013164192123636714], [-0.04977075513175517], [0.13990753432701702], [-0.0935692024721368]])

#Fishlens2_1
DIM=(1920, 1080)
K=np.array([[878.9856693064507, 0.0, 984.229488239518], [0.0, 880.5670859176757, 522.2643917752418], [0.0, 0.0, 1.0]])
D=np.array([[0.004507856557801049], [-0.07740119163750458], [0.13680212258662927], [-0.07831334029804657]])

def undistort_img(img):
    DIM=(1920, 1080)
    K=np.array([[878.9856693064507, 0.0, 984.229488239518], [0.0, 880.5670859176757, 522.2643917752418], [0.0, 0.0, 1.0]])
    D=np.array([[0.004507856557801049], [-0.07740119163750458], [0.13680212258662927], [-0.07831334029804657]])
    h,w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    # print(map1.shape, map2.shape)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img



if __name__ == '__main__':
    img = Image.open("2019-05-24-T16_30_39Z_Camera-Top-1.jpeg")
    img = undistort_img(np.array(img))
    img = Image.fromarray(img)
    img.save("undistorted-testimg1.jpeg")