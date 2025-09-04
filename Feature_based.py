
# SIFT (Scale Invariant Feature Transform)


import cv2 as cv
import os
import math
import numpy as np

dir = 'Master_fold'
test_gall = 'test'
input_img = 'AmanP.jpg'

img_read = os.path.join(test_gall, input_img)
img = cv.imread(img_read)
img = cv.resize(img, (500, 500))

import numpy as np

gray_img= cv.cvtColor(img,cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()
kp = sift.detect(gray_img,None)
#print(kp)
#img=cv.drawKeypoints(gray_img,kp,img)
img=cv.drawKeypoints(gray_img,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

kp, des = sift.compute(gray_img, kp)

#print(kp,'  ',des,'\n')
cv.imshow('image with key points',img)
#cv.imwrite('sift_keypoints.jpg',img)
cv.waitKey()

# for point in kp[:5]:   # just first 5 for readability
#     print("pt:", point.pt, "size:", point.size, "angle:", point.angle, "response:", point.response)

#print(kp,'\n',len(kp),'\n',des)
#print(des.shape)
print('\nlenght of test image descriptor for ', input_img, ': ',len(des),'\n')

#print('\n',len(des), '\n',des.shape)

kp_Des = []
names=[]

matches= []

for file in os.listdir(dir):


    image_path = os.path.join(dir, file)

    image = cv.imread(image_path)

    names.append(file)

    image = cv.resize(image, (500, 500))
    gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    kp2 = sift.detect(gray_img,None)
    kp2, des2 = sift.compute(gray_img, kp2)
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
    print('\n length of des2 for :',file,' ',len(des2))
    match = bf.match(des, des2)
    matches.append(match)
    #distance = math.sqrt(np.sum((des - des2)**2))
    #kp_Des.append(distance)


# for dis in kp_Des:
#     print(dis," ")

for match, name in zip(matches,names):
    print('\n match des for',name,' : ',match,' mtch len: ',len(match))

#min_index = np.argmin(kp_Des)

#print("\nnearest match: ",names[min_index])



#ORB (oriented fast and brief descriptor)

"""
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

import os

dir = 'Master_fold'
test_gall = 'test'
input_img = 'AmanP.jpg'

img_read = os.path.join(test_gall, input_img)
img = cv.imread(img_read)
img = cv.resize(img, (500, 500))
gray_img= cv.cvtColor(img,cv.COLOR_BGR2GRAY)

# Initiate ORB detector
orb = cv.ORB_create()

# find the keypoints with ORB
kp = orb.detect(gray_img,None)

# compute the descriptors with ORB
kp, des = orb.compute(gray_img, kp)

# draw only keypoints location,not size and orientation
#img2 = cv.drawKeypoints(gray_img, kp, None, color=(0,255,0), flags=0)
#plt.imshow(img2), plt.show()

#print(kp," \n",des)

import math

kp_Des = []
names=[]

#des = cv.resize(des,(480))


for file in os.listdir(dir):


    image_path = os.path.join(dir, file)

    image = cv.imread(image_path)

    names.append(file)

    image = cv.resize(image, (500, 500))
    gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    orb = cv.ORB_create()

    # find the keypoints with ORB
    kp2 = orb.detect(gray_img, None)

    # compute the descriptors with ORB
    kp2, des2 = orb.compute(gray_img, kp2)



    distance = math.sqrt(np.sum((des - des2)**2))
    kp_Des.append(distance)


for dis in kp_Des:
    print(dis," ")


min_index = np.argmin(kp_Des)

print("\nnearest match: ",names[min_index])

"""