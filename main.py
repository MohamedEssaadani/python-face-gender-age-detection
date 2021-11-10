import os
from datetime import datetime

import cv2
import face_recognition
import numpy as np
import pandas as pd

# **   ANTI SPOOFING       ** #
# Anti spoofing directories
train_dir = "antispoofing_dataset/train"
test_dir = "antispoofing_dataset/test"

# Dataset Exploration
categories = ["real", "spoof"]
print("---------------------Exploring Training Datasets--------------------")
for category in categories:
    path = os.path.join(train_dir, category)
    if category == 'real':
        r1 = len(os.listdir(path))
    else:
        s1 = len(os.listdir(path))
    print("There are {} images in {} directory".format(len(os.listdir(path)), category))
print("There are {} total images in training directory".format(r1 + s1))

print("-----------------------Exploring Testing Datasets-------------------------")
for category in categories:
    path = os.path.join(test_dir, category)
    if category == 'real':
        r2 = len(os.listdir(path))
    else:
        s2 = len(os.listdir(path))
    print("There are {} images in {} directory".format(len(os.listdir(path)), category))
print("There are {} total images in testing directory".format(r2 + s2))


def faceBox(faceNet, frame):
    # get frame height
    frameHeight = frame.shape[0]
    # get frame width
    frameWidth = frame.shape[1]
    # creates 4-dimensional blob from image
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], swapRB=False)
    #  pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detection = faceNet.forward()
    bboxs = []

    # loop over the detections
    for i in range(detection.shape[2]):
        # extract the confidence (i.e., probability) associated with the prediction
        confidence = detection[0, 0, i, 2]
        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.7:
            x1 = int(detection[0, 0, i, 3] * frameWidth)
            y1 = int(detection[0, 0, i, 4] * frameHeight)
            x2 = int(detection[0, 0, i, 5] * frameWidth)
            y2 = int(detection[0, 0, i, 6] * frameHeight)
            bboxs.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
    return frame, bboxs


faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# create vid object
video = cv2.VideoCapture(0)

padding = 20

while True:
    # start the video camera & get the frame & the ret(true or false)
    ret, frame = video.read()
    frame, bboxs = faceBox(faceNet, frame)
    for bbox in bboxs:
        # face=frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        face = frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
               max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPred = genderNet.forward()
        gender = genderList[genderPred[0].argmax()]

        ageNet.setInput(blob)
        agePred = ageNet.forward()
        age = ageList[agePred[0].argmax()]

        # Load a sample picture and learn how to recognize it.
        essaadani_image = face_recognition.load_image_file("photos_5iir/Mohamed Essaadani.jpg")
        essaadani_face_encoding = face_recognition.face_encodings(essaadani_image)[0]

        # Load a second sample picture and learn how to recognize it.
        sakout_image = face_recognition.load_image_file("photos_5iir/Reda Sakout.jpg")
        sakout_face_encoding = face_recognition.face_encodings(sakout_image)[0]

        # Create arrays of known face encodings and their names
        known_face_encodings = [
            essaadani_face_encoding,
            sakout_face_encoding
        ]
        known_face_names = [
            "ESSAADANI Mohamed",
            "SAKOUT Reda"
        ]

        # Initialize some variables
        face_locations = []
        face_encodings = []
        face_names = []
        process_this_frame = True

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    # write name to csv
                    names = pd.DataFrame([[name]],
                                         columns=['Name'])
                    now = datetime.now()

                    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

                    names.to_csv('attendance.csv')

                face_names.append(name)

        process_this_frame = not process_this_frame

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            # cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Draw a label with a name below the face
            if name == "Unknown":
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            else:
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        label = "{},{}".format(gender, age)
        cv2.rectangle(frame, (bbox[0], bbox[1] - 30), (bbox[2], bbox[1]), (0, 255, 0), -1)
        cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                    cv2.LINE_AA)
    cv2.imshow("Age-Gender", frame)

    k = cv2.waitKey(1)
    if k == ord('q'):
        break
video.release()
cv2.destroyAllWindows()
