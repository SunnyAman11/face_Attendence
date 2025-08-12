"""
import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video device.")
    exit()

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))  # Save as .avi

print("Recording... Press 'q' to stop.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame.")
        break

    out.write(frame)  # Save frame to file
    cv2.imshow('Recording...', frame)

    if cv2.waitKey(3000) & 0xFF == ord('q'):  # Press 'q' to stop
        break

cap.release()
out.release()
cv2.destroyAllWindows()

import os
print(os.path.abspath("output.avi"))
"""
import time

import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime

known_face_encodings = []
known_face_names = []

start_time = time.time()


for filename in os.listdir("faces"):
        image = face_recognition.load_image_file(f"faces/{filename}")
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):
            encoding = face_recognition.face_encodings(image)
            if encoding:
                known_face_encodings.append(encoding[0])
                known_face_names.append(os.path.splitext(filename)[0])

video_capture = cv2.VideoCapture(0)
start_time = time.time()

marked_names = set()
matched_once = False

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Cannot capture video")
        break

    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        best_match_index = np.argmin(face_distances)
        name = "Unknown"

        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            if name not in marked_names:
                print(name)
                marked_names.add(name)
                with open("attendance.csv", "a") as f:
                    f.write(f"{name}\n")
                print(f"Marked attendance for {name}")

            

        matched_once = True
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)  # Green box
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Attendance Camera", frame)

    end_time = time.time()

    if cv2.waitKey(1) & 0xFF == ord('q') :
        break

video_capture.release()
cv2.destroyAllWindows()

elapsed_time = end_time - start_time

print(f"The code executed in {elapsed_time:.4f} seconds.")
