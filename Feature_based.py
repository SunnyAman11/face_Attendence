
# SIFT (Scale Invariant Feature Transform)


import cv2 as cv
import os
import math
import numpy as np
import time


dir = 'Master_fold'
test_gall = 'test'

import numpy as np


#kp = sift.detect(gray_img,None)
#print(kp)
#img=cv.drawKeypoints(gray_img,kp,img)
#img=cv.drawKeypoints(gray_img,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


#print(kp,'\n')



#cv.imshow('image with key points',img)
#cv.imwrite('sift_keypoints.jpg',img)
#cv.waitKey()

# for point in kp[:5]:   # just first 5 for readability
#     print("pt:", point.pt, "size:", point.size, "angle:", point.angle, "response:", point.response)

#print('\n',len(des), '\n',des.shape)

kp_Dis = []
names=[]

matches= {}
test_data={}
inf_time={}

for file_name in os.listdir(test_gall):
    img_read = os.path.join(test_gall, file_name)
    img = cv.imread(img_read)
    img = cv.resize(img, (500, 500))
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    sift = cv.SIFT_create()

    if img is None:
        continue

    t11= time.time()
    kp, des = sift.detectAndCompute(gray_img, None)
    t12= time.time()

    print('inference time for',file_name, ': ',t12 - t11)

    # print(kp,'\n')

    kp_test = []

    matched = []

    matches = {}

    print(des.shape)
    print('\nlenght of test image descriptor for ', file_name, ': ', len(des), '\n')

    img = cv.drawKeypoints(gray_img, kp, img)
    img = cv.drawKeypoints(gray_img, kp, img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    t=0
    test_data[file_name]={}
    inf_time[file_name]={}

    for file in os.listdir(dir):

        dis =0

        t2=0

        matches[file]=[]
        image_path = os.path.join(dir, file)

        image = cv.imread(image_path)

        names.append(file)

        image = cv.resize(image, (500, 500))
        gray_img2 = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        sift = cv.SIFT_create()

        t21= time.time()
        kp2,des2 = sift.detectAndCompute(gray_img2,None)


        bf = cv.BFMatcher()
    #print('\n length of des2 for :',file,' ',len(des2))
        match = bf.knnMatch(des, des2,k=2)


  #  match = sorted(match, key=lambda x: x.distance)
        good = []
        #print('\n', file, len(match) , match, '\n')

        if len(match)>0 :

            for m, n in match:
                if m.distance < 0.75* n.distance:
                    good.append(m)
            #print('m: ',m.distance)
                    dis += (m.distance)**2
                elif n.distance < 0.75* m.distance:
                    good.append(n)
            t22 = time.time()

            t2 = t22 - t21

            if len(good) >0:
                matches[file].append(good)
                dis = math.sqrt(dis)
                test_data[file_name][file]=dis

        inf_time[file_name][file]= t2

for test_name, lst in test_data.items():
    print(test_name,': ')

    for data_nm, dis in lst.items():
        print(data_nm,': ',dis)

for test_name, lst in inf_time.items():
    print(test_name)
    for name, inf_tm in lst.items():
        print(name,': ', inf_tm)

"""
all_dfs = []

import pandas as pd

for test_img, master_dict in test_data.items():
    df = pd.DataFrame(master_dict, index=[test_img])

    all_dfs.append(df)

final_df = pd.concat(all_dfs)
final_df.to_csv("SIFT_.csv")
final_df.to_excel('SIFT_.xlsx')

"""
# Accuracy for sift = 18/49

"""
    matches.append(good)
    #distance = math.sqrt(np.sum((des - des2)**2))
    #kp_Des.append(distance)

kp_near={}

for name in names:
    for good in matches:
        dis=[]
        for kp in good:
            dis.append(kp.distance)
    min_dis = np.argmin(dis)
    kp_near[name]=good[min_dis]

for name, kp in kp_near.items():
    print(name, ': ',kp)

kp_dis=[]
for _,kp in kp_near.items():

    kp_dis.append(kp.distance)

print(kp_dis)

min_dist= np.argmin(kp_dis)

print("\nmatched name : ", names[min_dist], ' ',kp_dis[min_dist])





"""



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