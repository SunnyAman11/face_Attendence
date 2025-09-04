import cv2
import numpy as np
import os

import pandas as pd

dir = 'Master_fold'
test_gall = 'test'

names=[]

MSE_data = {}
for file_name in os.listdir(test_gall):
    img_read = os.path.join(test_gall, file_name)
    img = cv2.imread(img_read)
    if img is None:
        continue
    test_img = cv2.resize(img, (500, 500)).flatten()

    MSE_data[file_name] = {}


    for file in os.listdir(dir):
        img_path = os.path.join(dir, file)
        image = cv2.imread(img_path)
        if image is None:
            continue
        image = cv2.resize(image, (500, 500)).flatten()

        error = np.sum((test_img - image) ** 2)
        mse = np.sqrt(error)

        MSE_data[file_name][file] = mse


master_imag=' '

for file in os.listdir(dir):
    names.append(file)


for test_img, master_dict in MSE_data.items():

    collect= []

    for _,val in master_dict.items():
        collect.append(val)
        print(val)

    min_mse = np.argmin(collect)
    print("match image with ",test_img,": ",names[min_mse] )



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