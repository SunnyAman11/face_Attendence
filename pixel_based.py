#mean squared error
"""

import cv2
import numpy as np
import os
import pandas as pd
import math

import time

dir = 'Master_fold'
test_gall = 'test'
input_img = 'Apman12.jpg'

img_read = os.path.join(test_gall, input_img)
img = cv2.imread(img_read)
test_img = cv2.resize(img, (500, 500)).reshape(-1).astype(np.float32)

distances = []
names=[]

inf_time={}

MSE_data = {}

for file_name in os.listdir(test_gall):
    img_read = os.path.join(test_gall, file_name)
    img = cv2.imread(img_read)
    if img is None:
        continue
    test_img = cv2.resize(img, (500, 500)).flatten()

    MSE_data[file_name] = {}

    inf_time[file_name]={}

    t=0

    for file in os.listdir(dir):

        image_path = os.path.join(dir, file)

        img_test = cv2.imread(image_path)

        names.append(file)

        img_test = cv2.resize(img_test, (500, 500)).reshape(-1).astype(np.float32)

        t1 = time.time()
        sub = test_img - img_test


        add =0
        for val in sub:
            add += val**2

        mse = math.sqrt(np.mean(add))
        t2 = time.time()

        add=0
        inf_time[file_name][file] = t2 - t1
        MSE_data[file_name][file] = mse

for name, lst in inf_time.items():
    print('test: ',name)
    for key, lst_ in lst.items():
        print(key, 'time: ',lst)
"""
import time

""""
all_dfs = []

for test_img, master_dict in MSE_data.items():
    df = pd.DataFrame(master_dict, index=[test_img])
    all_dfs.append(df)

final_df = pd.concat(all_dfs)
final_df.to_csv("MSE_pixelbased.csv")
final_df.to_excel('MSEpixelbased.xlsx')
"""

"""
for name, distance in zip(names,distances):
    print(name, distance, '\n')


index_min= np.argmin(distances)


print("\nnearest image: ", names[index_min])

"""


# Structural Similarity Index

import numpy as np
import cv2
import os
from skimage.metrics import structural_similarity as ssim
import pandas as pd

dir = 'Master_fold'
test_gall = 'test'
input_img = 'AmanP.jpg'

img_read = os.path.join(test_gall, input_img)


similarity = []
names = []
scores=[]

inf_time={}

SSIM_data = {}

mapping=[]

for file_name in os.listdir(test_gall):
    img_read = os.path.join(test_gall, file_name)
    img = cv2.imread(img_read)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if img is None:
        continue
    test_img = cv2.resize(img, (500, 500)).reshape(-1).astype(np.float32)

    SSIM_data[file_name] = {}

    inf_time[file_name]={}

    t=0
    SSIM_data[file_name] = {}

    inf_time[file_name] = {}


    for file in os.listdir(dir):

        image_path = os.path.join(dir, file)
        image = cv2.imread(image_path)

        names.append(file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image= cv2.resize(image, (500, 500)).reshape(-1).astype(np.float32)

        t1= time.time()
        score, ssim_map = ssim(test_img, image, full=True,data_range=255)
        t2 = time.time()

        inf_time[file_name][file] = t2 - t1
        SSIM_data[file_name][file] = score


sum=0

for name,lst in inf_time.items():
    print("test image:",name)
    for data,latency in lst.items():
        print(data,': ',latency)
        sum+=latency

print("'\ntotal time for the process: ",sum,'seconds')

"""
all_dfs = []

for test_img, master_dict in SSIM_data.items():
    df = pd.DataFrame(master_dict, index=[test_img])

    all_dfs.append(df)

final_df = pd.concat(all_dfs)
final_df.to_csv("SSIM_.csv")
final_df.to_excel('SSIM_.xlsx')
"""

#correct match =29
#Accuracy =29/50


#for name, score,local_map in zip(names, scores,mapping):
 #   print(name, score, ' and local mapping: ',local_map, '\n')


"""
max_index = np.argmax(scores)
print('\n matched name: ',names[max_index])




df = pd.DataFrame({
    "name": names,
    "model distance": scores,

})

df.to_csv('pixel_based_SSIM.csv')

df.to_excel('pixel_based_ssim.xlsx')
"""


