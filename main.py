import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

video_capture = cv2.VideoCapture(0)

#face registration
ishan_image = face_recognition.load_image_file("faces/ishan.jpeg")
ishan_encoding= face_recognition.face_encodings(ishan_image)[0]

registered_face_encodings=[ishan_encoding, bali_encoding, bindi_encoding]
registered_face_names = ["Ishan"]

students = registered_face_names.copy()

face_location=[]
face_encoding=[]

now = datetime.now()
current_date = now.strftime("%d-%m-%Y")

f=open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    #Face recognition
    face_location = face_recognition.face_locations(rgb_small_frame)
    face_encoding = face_recognition.face_encodings(rgb_small_frame, face_location)

    for face_encoding in face_encoding:
        matches = face_recognition.compare_faces(registered_face_encodings, face_encoding)
        face_similiarity = face_recognition.face_distance(registered_face_encodings, face_encoding)
        best_match_index= np.argmin(face_similiarity)

        if(matches[best_match_index]):
            name = registered_face_names[best_match_index]

        if name in registered_face_names:
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (10, 100)
            fontScale = 1.5
            fontColor = (0, 0, 0)
            thickness = 3
            lineType = 2
            cv2.putText(frame, name + " is Present", bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)

        if name in students:
            students.remove(name)
            current_time = now.strftime("%H-%M-%S")
            lnwriter.writerow([name, current_time])

        cv2.imshow("Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord("Q"):
            break


video_capture.release()
cv2.destroyAllWindows()
f.close()
