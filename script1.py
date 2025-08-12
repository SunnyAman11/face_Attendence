import cv2
import face_recognition
import os
import numpy as np



dir= 'expressionFaces'

known_encodings= []
known_names= []

#print(face_locate)

for name in os.listdir(dir):
    if name.endswith(('jpg', 'jpeg')):
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

name='unknown'

print('hel1')
image = 'sav12.jpg'
img = cv2.imread(image)
face_locate = face_recognition.face_locations(img)
#print(face_locate)
face_encodings  = face_recognition.face_encodings(img)

print(face_locate)
print('hello\n')
print("\n\n\nfaceencodeinputwala",face_encodings[0],"\n\n")
c=0

for (top,left,right, bottom), face_encoding in zip(face_locate, face_encodings):
    c+=1
    face_dis = face_recognition.face_distance(known_encodings, face_encoding)
    print("facedis\n face dis",face_dis)
    best_match_index = np.argmin(face_dis)
    #print(best_match_index)

    name= known_names[best_match_index]
    print("top \n", top, 'left\n', left, '\nright', right )

   # cv2.rectangle(img, (left,top),(right, bottom), (0, 255, 0), 2)
    #cv2.putText(img, name, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50,225,178),5)

print("\n\n\n",c)
# cv2.imshow("Result", img)
print(name[0])
cv2.waitKey(0)
cv2.destroyAllWindows()
