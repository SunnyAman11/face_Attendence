import cv2
import face_recognition
import os
import numpy as np

import math
import pandas as pd

import openpyxl

dir= 'expressionFaces'

known_encodings= []
known_names= []

results = []


#print(face_locate)

for name in os.listdir(dir):
    if name.endswith(('jpg', 'jpeg','png')):
        img_path = os.path.join(dir, name)
        exp_img = face_recognition.load_image_file(img_path)
        encodings= face_recognition.face_encodings(exp_img)

        if encodings:
            #print(encodings)
            print("\n\n\nencodings", encodings, '\n\n\n')

            known_encodings.append(encodings[0])
            known_names.append(os.path.splitext(name))  # name without extension
        else:
            print(f"No face found in {name}")




def input_image(img_name):
    image_name= img_name
    image = cv2.imread(image_name)
    #face_locate = face_recognition.face_locations(image)
    face_encoding= face_recognition.face_encodings(image)

   # print("\n\n\nfaceencodeinputwala", face_encoding[0], "\n\n") #input image encoding

    face_dis=[]

    face_dis= face_recognition.face_distance(known_encodings,face_encoding[0])
    print("\n face distances:\t",face_dis,'\n')

    best_match_index = np.argmin(face_dis)
    #c = best_match_index
    name = known_names[best_match_index]
    print("name of image match: \t", name[0],"\n")

    return face_dis



def cosine_funct(img_name):
    image_name = img_name
    image = cv2.imread(image_name)
    #face_locate = face_recognition.face_locations(image)
    face_encoding = face_recognition.face_encodings(image)

    length_input_enc = 0

   # print("\n\n\nfaceencodeinputwala", face_encoding[0], "\n\n")
    for element in face_encoding[0]:
        length_input_enc += element * element

    length_input_enc = math.sqrt(length_input_enc)

    thetas = []
    products = []
    for element in known_encodings:
        length_known_enc = 0
        theta = 0
        dot_prod = 0
        for sub_ele, input_encod in zip(element, face_encoding[0]):
            dot_prod += sub_ele * input_encod

            length_known_enc += sub_ele * sub_ele
        length_known_enc = math.sqrt(length_known_enc)
        products.append(dot_prod)
        if length_known_enc == 0 or length_input_enc == 0:
            theta = float('inf')
        else:
            # Cosine angle
            cosine_val = dot_prod / (length_known_enc * length_input_enc)
            cosine_val = max(min(cosine_val, 1.0), -1.0)  # Clamp to avoid acos domain error
            theta = math.acos(cosine_val)
       # theta = math.acos(dot_prod / (abs(length_known_enc) * abs(length_input_enc)))
        thetas.append(theta)

    best_match_index = np.argmin(thetas)
    # c = best_match_index
    name = known_names[best_match_index]
    print("name of image match: \t", name[0], "\n")
    print("\n cosine distances: \t", thetas,'\n')

    return thetas


def eucledean_func(img_name):
    image_name = img_name
    image = cv2.imread(image_name)
    # face_locate = face_recognition.face_locations(image)
    face_encoding = face_recognition.face_encodings(image)

    length_input_enc = 0

    print("\n\n\nfaceencodeinputwala", face_encoding[0], "\n\n")
    diffence_encod = []
    for encodings in known_encodings:

        vec = face_encoding[0] - encodings

        diffence_encod.append(vec)

    dist_vect = []
    for element in diffence_encod:
        magnitude_dis = 0


        for sub_ele in element:
            magnitude_dis += sub_ele * sub_ele
        magnitude_dis = math.sqrt(magnitude_dis)
        dist_vect.append(magnitude_dis)
        #print("difference vector: \t", vec,"\t")
    print("dist vect", dist_vect, "\n")
    best_match_index = np.argmin(dist_vect)
    # c = best_match_index
    name = known_names[best_match_index]
    print("name of image match: \t", name[0], "\n")


    return dist_vect


model_dist= input_image('Aman.jpg')
eucled_dist = eucledean_func('Aman.jpg')

cosine_dist= cosine_funct('Aman.jpg')
# check in test.py







#df.to_excel('image compare.xls')

Name=known_names





df = pd.DataFrame({
    "name": Name,
    "model distance": model_dist,
    "euclidean distance": eucled_dist,
    "cosine distance": cosine_dist
})

df.to_csv('Image compare_new.csv')

df.to_excel('Image compare_new.xlsx')

