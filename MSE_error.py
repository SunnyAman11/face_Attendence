import cv2
import numpy as np
import os

import time

import pandas as pd

dir = 'Master_fold'
test_gall = 'test'

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
        img_path = os.path.join(dir, file)
        image = cv2.imread(img_path)
        if image is None:
            continue
        image = cv2.resize(image, (500, 500)).flatten()

        t1 =time.time()

        error = np.mean((test_img - image) ** 2)
       # mse = np.sqrt(error)
        mse=error

        t2= time.time()
        inf_time[file_name][file]=t2 - t1

        MSE_data[file_name][file] = mse


master_imag=' '

for file in os.listdir(dir):
    names.append(file)

count=0

for test_img, master_dict in MSE_data.items():

    collect= []
    count+=1
    for _,val in master_dict.items():
        collect.append(val)
        #print(val)

    min_mse = np.argmin(collect)
    print("match image with ",test_img,": ",names[min_mse] )


print('\n',count)

import matplotlib.pyplot as plt
sum=0

print('\n\n')
for test_img, master_dict in inf_time.items():
    times=[]
    print("test image: ",test_img,'time ')

    for name,time in master_dict.items():
        print(name,':',time*1000,'millisec')
        #print(val)
        times.append(time)
        sum+=time
        

print("total time: ", sum)


#Acccuracy = 32/count = 32/50 = 0.64
#

"""
all_dfs = []

for test_img, master_dict in MSE_data.items():
    df = pd.DataFrame(master_dict, index=[test_img])
    all_dfs.append(df)

final_df = pd.concat(all_dfs)
final_df.to_csv("MSE.csv")
final_df.to_excel('MSE.xlsx')
"""


"""
    plt.plot(name,MSE)

    plt.title("graph of mean squared error between test and data images ")
    plt.xlabel("name")
    plt.ylabel("MSE")
    plt.show()
"""