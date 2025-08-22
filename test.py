from __future__ import print_function

import cv2 as cv
import numpy as np
"""
W = 400

def my_ellipse(img, angle):
    thickness = 2
    line_type = 8

    cv.ellipse(img,
                (W // 2, W // 2),
                (W // 4, W // 16),
                angle,
                0,
                360,
                (255, 0, 0),
                thickness,
                line_type)

def my_filled_circle(img, center):
    thickness = -1
    line_type = 8

    cv.circle(img,
               center,
               W // 32,
               (0, 0, 255),
               thickness,
               line_type)

def my_polygon(img):
    line_type = 8

    # Create some points
    ppt = np.array([[W / 4, 7 * W / 8], [ 3*W / 4, 7 * W / 8],
                    [3 * W / 4, 13 * W / 16], [11 * W / 16, 13 * W / 16],
                    [19 * W / 32, 3 * W / 8], [3 * W / 4, 3 * W / 8],
                    [3 * W / 4, W / 8], [26 * W / 40, W / 8],
                    [26 * W / 40, W / 4], [22 * W / 40, W / 4],
                    [22 * W / 40, W / 8], [18 * W / 40, W / 8],
                    [18 * W / 40, W / 4], [14 * W / 40, W / 4],
                    [14 * W / 40, W / 8], [W / 4, W / 8],
                    [W / 4, 3 * W / 8], [13 * W / 32, 3 * W / 8],
                    [5 * W / 16, 13 * W / 16], [W / 4, 13 * W / 16]], np.int32)
    ppt = ppt.reshape((-1, 1, 2))
    cv.fillPoly(img, [ppt], (255, 255, 255), line_type)
    # Only drawind the lines would be:
    # cv.polylines(img, [ppt], True, (255, 0, 255), line_type)

def my_line(img, start, end):
    thickness = 2
    line_type = 8

    cv.line(img,
             start,
             end,
             (0, 0, 0),
             thickness,
             line_type)

atom_window = "Drawing 1: Atom"
rook_window = "Drawing 2: Rook"

# Create black empty images
size = W, W, 3
atom_image = np.zeros(size, dtype=np.uint8)
rook_image = np.zeros(size, dtype=np.uint8)

print(atom_image)
print(rook_image)

# 1.a. Creating ellipses
my_ellipse(atom_image, 90)
#my_ellipse(atom_image, 0)
#my_ellipse(atom_image, 45)
#my_ellipse(atom_image, -45)

# 1.b. Creating circles
my_filled_circle(atom_image, (W // 2, W // 2))

# 2. Draw a rook
# ------------------
# 2.a. Create a convex polygon
my_polygon(rook_image)

cv.rectangle(rook_image,
              (0, 7 * W // 8),
              (W, W),
              (0, 255, 0),
              -1,
              8)

#  2.c. Create a few lines
#my_line(rook_image, (0, 15 * W // 16), (W, 15 * W // 16))
#my_line(rook_image, (W // 4, 7 * W // 8), (W // 4, W))
#my_line(rook_image, (W // 2, 7 * W // 8), (W // 2, W))
#my_line(rook_image, (3 * W // 4, 7 * W // 8), (3 * W // 4, W))

cv.imshow(atom_window, atom_image)
cv.moveWindow(atom_window, 0, 200)
cv.imshow(rook_window, rook_image)
cv.moveWindow(rook_window, W, 200)

cv.waitKey(0)
cv.destroyAllWindows()
"""

"""
import cv2 as cv
import time
# Find the file path (optional step if you know the file is local)
filename = cv.samples.findFile('Aman.jpg')

# Read the image in color
src = cv.imread(filename, cv.IMREAD_COLOR)

# Show the image
cv.imshow('result', src)
#t = round(time.time())

kernel = np.array([[0, -2, 0],
                       [-2, 10, -2],
                       [0, -2, 0]], np.float32)

dst1 = cv.filter2D(src, -1, kernel)

cv.imshow("Output", dst1)

cv.waitKey()
cv.destroyAllWindows()
"""

#from __future__ import print_function


"""

import cv2 as cv

alpha = 0.5

try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3

print(''' Simple Linear Blender
-----------------------
* Enter alpha [0.0-1.0]: ''')
input_alpha = float(raw_input().strip())
if 0 <= alpha <= 1:
    alpha = input_alpha
# [load]
src1 = cv.imread(cv.samples.findFile('ravi.jpg'))
src2 = cv.imread(cv.samples.findFile('sav12.jpg'))

src2_resized = cv.resize(src2, (src1.shape[1], src1.shape[0]))


shape1 = src1.shape
shape2= src2_resized.shape

print(src1.shape,'\n')
print(shape2)
if (shape1 != shape2):
    exit(-1)
# [load]
if src1 is None:
    print("Error loading src1")
    exit(-1)
elif src2 is None:
    print("Error loading src2")
    exit(-1)
# [blend_images]
beta = (1.0 - alpha)
dst = cv.addWeighted(src1, alpha, src2_resized, beta, 0.0)
# [blend_images]
# [display]
cv.imshow('dst', dst)
cv.waitKey(0)
# [display]
cv.destroyAllWindows()
"""
"""
import numpy as np
import cv2 as cv
#from matplotlib import pyplot as plt

img = cv.imread('Aman.jpg', cv.IMREAD_GRAYSCALE)

# Initiate ORB detector
orb = cv.ORB_create()

# find the keypoints with ORB
kp = orb.detect(img,None)

print(kp,'\n')
# compute the descriptors with ORB
kp, des = orb.compute(img, kp)

print('kp', kp, '\n', 'des',des)
# draw only keypoints location,not size and orientation
img2 = cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
cv.imshow('features',img2)
cv.waitKey()
cv.destroyAllWindows()

"""
#cosine functionality
"""
length_input_enc = 0
for element in face_encodings[0]:
    length_input_enc+= element*element

length_input_enc = math.sqrt(length_input_enc)

print("encode for input",length_input_enc, "\n\n")

dis_encod=0
arr_dis=[]
diffence_encod=[]

print("\n\n\n")
for encodings in known_encodings:

    vec= face_encodings[0]- encodings


    diffence_encod.append(vec)
    vec=0

for element in diffence_encod:
    print("hello\n")
    print(element, )





dist_vect=[]
#eucleaden distance

for element in diffence_encod:
    magnitude_dis=0
    for sub_ele in element:
        magnitude_dis+= sub_ele*sub_ele
    magnitude_dis=math.sqrt(magnitude_dis)
    dist_vect.append(magnitude_dis)

print("dist vect", dist_vect,"\n")

# magnitutde of input

length_input_enc =0

for element in face_encodings[0]:
    length_input_enc += element*element







# calculating cosine related


thetas=[]
products=[]
for element in known_encodings:
    magnitude_dis = 0
    theta=0
    dot_prod=0
    for sub_ele, input_encod in zip(element,face_encodings[0]):
        dot_prod += sub_ele*input_encod

        magnitude_dis += sub_ele* sub_ele
        magnitude_dis = math.sqrt(magnitude_dis)
    products.append(dot_prod)
    theta = math.acos(dot_prod/(abs(magnitude_dis) * abs(length_input_enc)))
    thetas.append(theta)


print("dotprodcts: ", products,"\n")
print("\ntheta: ", thetas,"\n\n")

print("face distace from library:", face_dis)
#for element in diffence_encod:
    #dis


print("\n\n\n")
test_image_path = os.path.join(dir, 'ravi bhandari.jpg')
test_image = face_recognition.load_image_file(test_image_path)
test_encod = face_recognition.face_encodings(test_image)
print(test_encod[0])

vec= face_encodings[0] - test_encod[0]

dis = 0

for element in vec:
    dis += element*element

dis = math.sqrt(dis)
print("\nvec distance\t: ",dis)

"""

"""
name='unknown'

print('hel1')
image = 'ravi.jpg'
img = cv2.imread(image)
face_locate = face_recognition.face_locations(img)
#print(face_locate)
face_encodings  = face_recognition.face_encodings(img)

print(face_locate)
print('hello\n')
print("\n\n\nfaceencodeinputwala",face_encodings[0],"\n\n")
c=0
face_dis=[]
for (top,left,right, bottom), face_encoding in zip(face_locate, face_encodings):
    c+=1
    face_dis = face_recognition.face_distance(known_encodings, face_encoding)
    print("facedis\n face dis",face_dis)
    best_match_index = np.argmin(face_dis)
    #print(best_match_index)
    c=best_match_index
    name= known_names[best_match_index]
    print("top \n", top, 'left\n', left, '\nright', right )

   # cv2.rectangle(img, (left,top),(right, bottom), (0, 255, 0), 2)
    #cv2.putText(img, name, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50,225,178),5)

print("\n\n\n",c)
# cv2.imshow("Result", img)
print(name[0])
cv2.waitKey(0)
cv2.destroyAllWindows()

"""
"""
import face_recognition
import cv2

# Load image
image = face_recognition.load_image_file("Aman.jpg")

# Convert RGB (face_recognition uses RGB) to BGR for OpenCV
image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# Detect face landmarks
face_landmarks_list = face_recognition.face_landmarks(image)

# Draw landmarks
for face_landmarks in face_landmarks_list:
    for feature, points in face_landmarks.items():
        # Draw each point on the feature
        for (x, y) in points:
            cv2.circle(image_bgr, (x, y), 2, (0, 255, 0), -1)  # green dot

# Display the image
cv2.imshow("Face with Landmarks", image_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()

#cv2.imshow(f"Face ", image)

# Wait until a key is pressed
#cv2.waitKey(0)
#cv2.destroyAllWindows()
"""

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

# print("\nrows, columns :", h,w, '\n')

print("\n predicted image: ", gallery_data[predicted_label])

eigen_faces = model.getEigenVectors()

#print("\neigenface:", eigen_faces)

#print(eigen_faces.shape, '\n')
#print("eigen row length:", eigen_faces.shape[0])

mean = model.getMean()


test_vector = test_image.flatten().astype(np.float32)

#print("test image without mean shift /original: ", test_vector)


#test_centered = (test_vector - mean).reshape(-1)

#print("\n test image shape: ", test_vector.shape)


projection = np.dot( test_vector,eigen_faces)  # shape (12,)

test_project = projection


#print("\ntest image: ",test_vector)

#print("\nprojection of test image on eigen face: ", projection)



#projection of gallery images to eigen faces

projection_gallery=[]

for image in database_img:
    img_vec = image.flatten().astype(np.float32)

    projection = np.dot(img_vec,eigen_faces)
   # print('\n projection of images on eigen faces : ', projection)
    projection_gallery.append(projection)



"""
for image in database_img:
    print("\nimage shape: ", image.reshape(-1).shape)
"""




print('\n database image: ', gallery_data[predicted_label],'\n')

#print(f"Predicted class = {predicted_label} / distance = {distance}")


"""

eigen_vectors = model._model['eigenvectors']

print('\n',eigen_vectors)


"""

distances = np.linalg.norm(projection_gallery - test_project, axis=1)  # Euclidean distance
best_match_index = np.argmin(distances)

#print(distances)

print("Best match index:", best_match_index)

num_faces = eigen_faces.shape[1]


img_size = int(np.sqrt(eigen_faces.shape[0]))  # assuming square images

import matplotlib.pyplot as plt


for i in range(min(num_faces, 10)):  # display first 10
    eigenface = eigen_faces[:, i].reshape((img_size, img_size))  # reshape to image

    plt.subplot(2, 5, i + 1)
    plt.imshow(eigenface, cmap='jet')  # jet colormap for colorful visualization
    plt.title(f"Eigenface {i + 1}")
    plt.axis('off')

plt.show()

# this is my trained model file
#model.write("my_trained_model.yml")


