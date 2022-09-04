import cv2
import dlib
import numpy as np

age_weights = "age_deploy.prototxt"
age_config = "age_net.caffemodel"
age_Net = cv2.dnn.readNet(age_config, age_weights)

gender_weights = "gender_deploy.prototxt"
gender_config = "gender_net.caffemodel"
gender_Net = cv2.dnn.readNet(gender_config, gender_weights)

ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
           '(25-32)', '(38-43)', '(48-53)', '(60-100)']
model_mean = (78.4263377603, 87.7689143744, 114.895847746)

genList = ['Male', 'Female']

Boxes = []

text = 'Age'

def face_detect(frame):

    faces = face_detector(frame)
    results = []

    for face in faces:
        left = face.left()  # extracting the face coordinates
        top = face.top()
        right = face.right()
        bottom = face.bottom()

        box = [left, top, right, bottom]
        Boxes.append(box)

        for box in Boxes:
            face = frame[box[1]:box[3], box[0]:box[2]]
        
        blob = cv2.dnn.blobFromImage(face, 1.0 , (227, 227), model_mean, swapRB=False)

        age_Net.setInput(blob)
        age_preds = age_Net.forward()
        age = ageList[age_preds[0].argmax()]

        gender_Net.setInput(blob)
        gender_preds = gender_Net.forward()
        gender = genList[gender_preds[0].argmax()]
        
        results.append([top, right, bottom, left, age, gender])
        
    return results


vid = cv2.VideoCapture(0)

while(True):

    ret, frame = vid.read()
    
    face_detector = dlib.get_frontal_face_detector()
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detect(img_gray)

    for top, right, bottom, left, age, gender in results:
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, f'{gender}', (left, top), cv2.FONT_HERSHEY_COMPLEX, 0.5, (36,255,12), 1)
        cv2.putText(frame, f'{text}:{age}', (left, top-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 1)

    cv2.imshow('Age Detection', frame)


    k = cv2.waitKey(30) & 0xff
    if k==27:
        break

vid.release()

cv2.destroyAllWindows()

