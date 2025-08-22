import cv2
import numpy as np
import pandas as pd

import os


target_size= (100,100)

dir = 'expressionFaces'

database_img=[]

label_data=[]

gallery= []

c=0

for name in os.listdir('dir'):
    if name.endswith(('jpg','jpeg','png')):
        img_path = os.path.join(gallery, name)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        norm_img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        resized_img = cv2.resize(img, target_size)

        database_img.append(resized_img)
        label_data.append(c)
        c += 1
        gallery.append(name)
