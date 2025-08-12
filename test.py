from __future__ import print_function

import cv2 as cv
import numpy as np
"""
W = 400

def my_ellipse(img, angle):
    thickness = 2
    line_type = 8

    cv.ellipse(img,
                (W // 2, W // 2),
                (W // 4, W // 16),
                angle,
                0,
                360,
                (255, 0, 0),
                thickness,
                line_type)

def my_filled_circle(img, center):
    thickness = -1
    line_type = 8

    cv.circle(img,
               center,
               W // 32,
               (0, 0, 255),
               thickness,
               line_type)

def my_polygon(img):
    line_type = 8

    # Create some points
    ppt = np.array([[W / 4, 7 * W / 8], [ 3*W / 4, 7 * W / 8],
                    [3 * W / 4, 13 * W / 16], [11 * W / 16, 13 * W / 16],
                    [19 * W / 32, 3 * W / 8], [3 * W / 4, 3 * W / 8],
                    [3 * W / 4, W / 8], [26 * W / 40, W / 8],
                    [26 * W / 40, W / 4], [22 * W / 40, W / 4],
                    [22 * W / 40, W / 8], [18 * W / 40, W / 8],
                    [18 * W / 40, W / 4], [14 * W / 40, W / 4],
                    [14 * W / 40, W / 8], [W / 4, W / 8],
                    [W / 4, 3 * W / 8], [13 * W / 32, 3 * W / 8],
                    [5 * W / 16, 13 * W / 16], [W / 4, 13 * W / 16]], np.int32)
    ppt = ppt.reshape((-1, 1, 2))
    cv.fillPoly(img, [ppt], (255, 255, 255), line_type)
    # Only drawind the lines would be:
    # cv.polylines(img, [ppt], True, (255, 0, 255), line_type)

def my_line(img, start, end):
    thickness = 2
    line_type = 8

    cv.line(img,
             start,
             end,
             (0, 0, 0),
             thickness,
             line_type)

atom_window = "Drawing 1: Atom"
rook_window = "Drawing 2: Rook"

# Create black empty images
size = W, W, 3
atom_image = np.zeros(size, dtype=np.uint8)
rook_image = np.zeros(size, dtype=np.uint8)

print(atom_image)
print(rook_image)

# 1.a. Creating ellipses
my_ellipse(atom_image, 90)
#my_ellipse(atom_image, 0)
#my_ellipse(atom_image, 45)
#my_ellipse(atom_image, -45)

# 1.b. Creating circles
my_filled_circle(atom_image, (W // 2, W // 2))

# 2. Draw a rook
# ------------------
# 2.a. Create a convex polygon
my_polygon(rook_image)

cv.rectangle(rook_image,
              (0, 7 * W // 8),
              (W, W),
              (0, 255, 0),
              -1,
              8)

#  2.c. Create a few lines
#my_line(rook_image, (0, 15 * W // 16), (W, 15 * W // 16))
#my_line(rook_image, (W // 4, 7 * W // 8), (W // 4, W))
#my_line(rook_image, (W // 2, 7 * W // 8), (W // 2, W))
#my_line(rook_image, (3 * W // 4, 7 * W // 8), (3 * W // 4, W))

cv.imshow(atom_window, atom_image)
cv.moveWindow(atom_window, 0, 200)
cv.imshow(rook_window, rook_image)
cv.moveWindow(rook_window, W, 200)

cv.waitKey(0)
cv.destroyAllWindows()
"""

"""
import cv2 as cv
import time
# Find the file path (optional step if you know the file is local)
filename = cv.samples.findFile('Aman.jpg')

# Read the image in color
src = cv.imread(filename, cv.IMREAD_COLOR)

# Show the image
cv.imshow('result', src)
#t = round(time.time())

kernel = np.array([[0, -2, 0],
                       [-2, 10, -2],
                       [0, -2, 0]], np.float32)

dst1 = cv.filter2D(src, -1, kernel)

cv.imshow("Output", dst1)

cv.waitKey()
cv.destroyAllWindows()
"""

#from __future__ import print_function


"""

import cv2 as cv

alpha = 0.5

try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3

print(''' Simple Linear Blender
-----------------------
* Enter alpha [0.0-1.0]: ''')
input_alpha = float(raw_input().strip())
if 0 <= alpha <= 1:
    alpha = input_alpha
# [load]
src1 = cv.imread(cv.samples.findFile('ravi.jpg'))
src2 = cv.imread(cv.samples.findFile('sav12.jpg'))

src2_resized = cv.resize(src2, (src1.shape[1], src1.shape[0]))


shape1 = src1.shape
shape2= src2_resized.shape

print(src1.shape,'\n')
print(shape2)
if (shape1 != shape2):
    exit(-1)
# [load]
if src1 is None:
    print("Error loading src1")
    exit(-1)
elif src2 is None:
    print("Error loading src2")
    exit(-1)
# [blend_images]
beta = (1.0 - alpha)
dst = cv.addWeighted(src1, alpha, src2_resized, beta, 0.0)
# [blend_images]
# [display]
cv.imshow('dst', dst)
cv.waitKey(0)
# [display]
cv.destroyAllWindows()
"""

import numpy as np
import cv2 as cv
#from matplotlib import pyplot as plt

img = cv.imread('Aman.jpg', cv.IMREAD_GRAYSCALE)

# Initiate ORB detector
orb = cv.ORB_create()

# find the keypoints with ORB
kp = orb.detect(img,None)

print(kp,'\n')
# compute the descriptors with ORB
kp, des = orb.compute(img, kp)

print('kp', kp, '\n', 'des',des)
# draw only keypoints location,not size and orientation
img2 = cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
cv.imshow('features',img2)
cv.waitKey()
cv.destroyAllWindows()