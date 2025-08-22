import cv2
import pandas as pd

import numpy as np

import face_recognition


target_size = (100,100)

img= cv2.imread('chaman1.jpg',cv2.IMREAD_GRAYSCALE)

norm_img_input= cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
norm_img_res = cv2.resize(norm_img_input, target_size)




import os

gallery = "expressionFaces"

database_img=[]

label_data=[]


gallery_data=[]

glob_=0
c=0
for name in os.listdir(gallery):
    if name.endswith(('jpg', 'jpeg','png')):
        img_path = os.path.join(gallery, name)

        img=cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        norm_img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        resized_img = cv2.resize(img, target_size)

        database_img.append(resized_img)
        label_data.append(c)
        c+=1
        gallery_data.append(name)
        glob_+=1


for element in database_img:
    print(element,'\n')


test_image= norm_img_res

model = cv2.face.EigenFaceRecognizer_create()
model.train(database_img , np.array(label_data))

predicted = model.predict(norm_img_res)

predicted_label, distance = predicted

h, w = database_img[0].shape

print("\nrows, columns :", h,w, '\n')

eigen_faces = model.getEigenVectors()

print("\neigenface:", eigen_faces)

print(eigen_faces.shape, '\n')
print("eigen row length:", eigen_faces.shape[0])

mean = model.getMean()


test_vector = test_image.flatten().astype(np.float32)

print("test image without mean shift /original: ", test_vector)


#test_centered = (test_vector - mean).reshape(-1)

print("\n test image shape: ", test_vector.shape)


projection = np.dot( test_vector,eigen_faces)  # shape (12,)

test_project = projection


print("\ntest image: ",test_vector)

print("\nprojection of test image on eigen face: ", projection)



#projection of gallery images to eigen faces

projection_gallery=[]

for image in database_img:
    img_vec = image.flatten().astype(np.float32)

    projection = np.dot(img_vec,eigen_faces)
    print('\n projection of images on eigen faces : ', projection)
    projection_gallery.append(projection)



"""
for image in database_img:
    print("\nimage shape: ", image.reshape(-1).shape)
"""




print('\n database image: ', gallery_data[predicted_label],'\n')

print(f"Predicted class = {predicted_label} / distance = {distance}")

#eigen_values_input = model.EigenValues(norm_img_res)

"""

eigen_vectors = model._model['eigenvectors']

print('\n',eigen_vectors)


"""

distances = np.linalg.norm(projection_gallery - test_project, axis=1)  # Euclidean distance
best_match_index = np.argmin(distances)

print(distances)

print("Best match index:", best_match_index)







# this is my trained model file
#model.write("my_trained_model.yml")

