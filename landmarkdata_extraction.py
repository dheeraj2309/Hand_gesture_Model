import numpy as np
import tensorflow as tf
import mediapipe as mp
import os
import cv2 as cv


mp_hands=mp.solutions.hands
hands=mp_hands.Hands(static_image_mode=True,max_num_hands=1,min_detection_confidence=0.5)

photos = 'pics/'

data=[]
labels=[]
for gesture in os.listdir(photos):
    class_path = os.path.join(photos,gesture)


    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path,img_name)
        
        image=cv.imread(img_path)
        
        rgb_image=cv.cvtColor(image,cv.COLOR_BGR2RGB)
        result=hands.process(rgb_image)
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                landmarks=[]
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x,lm.y,lm.z])
                    # landmarks = np.array(landmarks).flatten() / np.linalg.norm(landmarks)
                data.append(landmarks)
                labels.append(gesture)
        else:
            print(f"the image was not fed:{img_path}")

np.savez("training.npz",landmarks=np.array(data),labels=np.array(labels))

