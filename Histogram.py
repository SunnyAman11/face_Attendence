import time

# Histogram Intersection


import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

import math

dir = 'Master_fold'
test_gall = 'test'



#print(hist_test,'\n')


# for chans in channels:
#     print(chans,'\n')

colors = ('b', 'g', 'r')


hist_images = []
names=[]

hist_plot =[]
inf_time={}

Hist_data = {}

sum_data_time=0

"""
for file_name in os.listdir(test_gall):
    img_read = os.path.join(test_gall, file_name)
    img = cv2.imread(img_read)
    if img is None:
        continue
    test_img = cv2.resize(img, (500, 500))
    t11= time.time()
   # if len(test_img.shape) == 3 and test_img.shape[2] == 3:
    hist_test = cv2.calcHist([test_img], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])

    #hist_test = cv2.normalize(hist_test, hist_test).flatten()
    t12= time.time()
    Hist_data[file_name] = {}

    inf_time[file_name]={}
    sum_data_time += t12-t11

    t=0

    for file in os.listdir(dir):
        image_path = os.path.join(dir, file)

        image = cv2.imread(image_path)

        names.append(file)

        image = cv2.resize(image, (500, 500))

        t21= time.time()
        hist_image = cv2.calcHist([image],[0,1,2], None,[256,256,256],[0,256,0,256,0,256])
       # hist_image = cv2.normalize(hist_image, hist_image).flatten()
        t22= time.time()
    # hist_plot.append(hist_image)

        t1=t22 - t21
        similarity = np.sum(np.minimum(hist_test, hist_image))

        inf_time[file_name][file] = t1

        Hist_data[file_name][file] = similarity
"""



sum=0
"""
print('\n\n')
for test_img, master_dict in inf_time.items():
    times=[]
    print("test image: ",test_img,'time ')

    for name,time in master_dict.items():
        print(name,':',time*1000,'millisec')
        #print(val)
        times.append(time)
        sum+=time

print("total time :", sum+ sum_data_time*1000,'milli second')
"""
"""

all_dfs = []

for test_img, master_dict in Hist_data.items():
    df = pd.DataFrame(master_dict, index=[test_img])
    all_dfs.append(df)

final_df = pd.concat(all_dfs)
final_df.to_csv("Hist.csv")
final_df.to_excel('Hist.xlsx')

#df.to_csv('histogram_based_.csv')

#df.to_excel('histogram_based_.xlsx')

#correct matched 34 out of 49
print("\ntest image histogram\n")
"""


# test images histogram
"""
for file in os.listdir(test_gall):

    image_path = os.path.join(test_gall, file)
    image = cv2.imread(image_path)
    image = cv2.resize(image, (500, 500))

    if image is None:
        continue

    channels_data = cv2.split(image)

    for chan, color in zip(channels_data, colors):
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])

        hist = cv2.normalize(hist, hist, alpha=0, beta=0.5, norm_type=cv2.NORM_MINMAX)

        plt.plot(hist, color=color)
        plt.xlim([0, 256])

    plt.title(f"Normalized RGB Histogram of {file}")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Normalized Frequency")
    plt.show()

"""


"""
for chan, color in zip(channels, colors):
    hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
    plt.plot(hist, color=color)
    plt.xlim([0, 256])

plt.title("RGB Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.show()
"""





# Chisquare distance


import pandas as pd
import cv2
import numpy as np
import os

import math

dir = 'Master_fold'
test_gall = 'test'

hist_images = []
names=[]
chi_sq=0

inf_time={}

Hist_data = {}

sum_data_time=0

for file_name in os.listdir(test_gall):
    img_read = os.path.join(test_gall, file_name)
    img = cv2.imread(img_read)
    if img is None:
        continue
    test_img = cv2.resize(img, (500, 500))
    t11= time.time()
   # if len(test_img.shape) == 3 and test_img.shape[2] == 3:
    hist_test = cv2.calcHist([test_img], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])

    hist_test = cv2.normalize(hist_test, hist_test).flatten()
    t12= time.time()
    Hist_data[file_name] = {}

    inf_time[file_name]={}
    sum_data_time += t12-t11


    for file in os.listdir(dir):
        image_path = os.path.join(dir, file)

        image = cv2.imread(image_path)

        names.append(file)

        image = cv2.resize(image, (500, 500))
        t21 = time.time()
        hist_image = cv2.calcHist([image], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    #hist_image = hist_image.reshape(-1).astype(np.float32)
        hist_image= cv2.normalize(hist_image, hist_image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX).flatten()

    # for
        chi_sq = np.sum((hist_test - hist_image)**2 /(hist_test + hist_image+0.05))
        t22 = time.time()
        hist_images.append(chi_sq)
        chi_sq =0
        t1 = t22 - t21

        inf_time[file_name][file] = t1

        #Hist_data[file_name][file] = chi_sq




for test_img, master_dict in inf_time.items():
    times=[]
    print("test image: ",test_img,'time ')

    for name,time in master_dict.items():
        print(name,':',time*1000,'millisec')
        #print(val)
        times.append(time)
        sum+=time


print("total time :", sum+ sum_data_time*1000,'milli second')






#for name, similar in zip(names, hist_images) :
 #   print(name, similar,'\n')

#similar = np.argmin(hist_images)

#print('\nmatch with :', names[similar])
"""
df = pd.DataFrame({
    "name": names,
    "hist_similarity": hist_images,

})

df.to_csv('histogram_chisquare based_.csv')

df.to_excel('histogram_chi_square_based_.xlsx')
"""