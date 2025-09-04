#mean squared error

"""
import cv2
import numpy as np
import os

import math

dir = 'Master_fold'
test_gall = 'test'
input_img = 'Apman12.jpg'

img_read = os.path.join(test_gall, input_img)
img = cv2.imread(img_read)
test_img = cv2.resize(img, (500, 500)).reshape(-1).astype(np.float32)

distances = []
names=[]

for file in os.listdir(dir):


    image_path = os.path.join(dir, file)

    img_test = cv2.imread(image_path)

    names.append(file)

    img_test = cv2.resize(img_test, (500, 500)).reshape(-1).astype(np.float32)

    sub = test_img - img_test
    print(sub,'\n')

    add =0
    for val in sub:
        add += val**2

    mse = math.sqrt(np.mean(add))
    add=0

    distances.append( mse)



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
img = cv2.imread(img_read)
img= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
test_img = cv2.resize(img, (500, 500)).reshape(-1).astype(np.float32)

similarity = []
names = []
scores=[]

mapping=[]

for file in os.listdir(dir):


    image_path = os.path.join(dir, file)

    image = cv2.imread(image_path)

    names.append(file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image= cv2.resize(image, (500, 500)).reshape(-1).astype(np.float32)

    score, ssim_map = ssim(test_img, image, full=True,data_range=255)

    scores.append(score)
    mapping.append(ssim_map)


for name, score,local_map in zip(names, scores,mapping):
    print(name, score, ' and local mapping: ',local_map, '\n')


max_index = np.argmax(scores)
print('\n matched name: ',names[max_index])




df = pd.DataFrame({
    "name": names,
    "model distance": scores,

})

df.to_csv('pixel_based_SSIM.csv')

df.to_excel('pixel_based_ssim.xlsx')





