
# Histogram Intersection


import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

import math

dir = 'Master_fold'
test_gall = 'test'
input_img = 'Akshay.png'

img_read = os.path.join(test_gall, input_img)
img = cv2.imread(img_read)
test_img = cv2.resize(img, (500, 500))

hist_test = cv2.calcHist([test_img], [0, 1, 2], None, [256, 256, 256], [0,256, 0,256, 0,256])
hist_test = cv2.normalize(hist_test, hist_test).flatten()


#print(hist_test,'\n')

channels = cv2.split(test_img)

# for chans in channels:
#     print(chans,'\n')

colors = ('b', 'g', 'r')


hist_images = []
names=[]

hist_plot =[]

for file in os.listdir(dir):


    image_path = os.path.join(dir, file)

    image = cv2.imread(image_path)

    names.append(file)

    image = cv2.resize(image, (500, 500))

    hist_image = cv2.calcHist([image],[0,1,2], None,[256,256,256],[0,256,0,256,0,256])
    hist_image = cv2.normalize(hist_image, hist_image).flatten()

    # hist_plot.append(hist_image)

    similarity = np.sum(np.minimum(hist_test, hist_image))

    hist_images.append(similarity)



for name, similar in zip(names, hist_images) :
    print(name, similar,'\n')

similar = np.argmax(hist_images)

print('\nmatch with :', names[similar])

df = pd.DataFrame({
    "name": names,
    "histogram similarity": hist_images,

})

df.to_csv('histogram_based_.csv')

df.to_excel('histogram_based_.xlsx')


print("\ntest image histogram\n")


"""
for file in os.listdir(dir):


    image_path = os.path.join(dir, file)

    image = cv2.imread(image_path)
    image = cv2.resize(image, (500, 500))
    channels_data = cv2.split(image)

    hist_image = cv2.calcHist([image], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    for chan, color in zip(channels, colors):
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        plt.plot(hist, color=color)
        plt.xlim([0, 256])

    plt.title(f"RGB Histogram of {file}")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
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

"""
import pandas as pd
import cv2
import numpy as np
import os

import math

dir = 'Master_fold'
test_gall = 'test'
input_img = 'Akshay.png'

img_read = os.path.join(test_gall, input_img)
img = cv2.imread(img_read)
test_img = cv2.resize(img, (500, 500))

hist_test = cv2.calcHist([test_img], [0, 1, 2], None, [256, 256, 256], [0,256, 0,256, 0,256])
hist_test = cv2.normalize(hist_test, hist_test, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

#hist_test = hist_test.reshape(-1).astype(np.float32)

print(hist_test.size,'\n')

hist_images = []
names=[]
chi_sq=0

for file in os.listdir(dir):


    image_path = os.path.join(dir, file)

    image = cv2.imread(image_path)

    names.append(file)

    image = cv2.resize(image, (500, 500))

    hist_image = cv2.calcHist([image], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    #hist_image = hist_image.reshape(-1).astype(np.float32)
    hist_image= cv2.normalize(hist_image, hist_image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    # for
    chi_sq = np.sum((hist_test - hist_image)**2 /(hist_test + hist_image+0.05))

    hist_images.append(chi_sq)
    chi_sq =0

for name, similar in zip(names, hist_images) :
    print(name, similar,'\n')

similar = np.argmin(hist_images)

print('\nmatch with :', names[similar])

df = pd.DataFrame({
    "name": names,
    "hist_similarity": hist_images,

})

df.to_csv('histogram_chisquare based_.csv')

df.to_excel('histogram_chi_square_based_.xlsx')
"""