import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

data=np.load('training.npz')
landmarks=data['landmarks']
labels=data['labels']

label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_onehot = to_categorical(labels_encoded)

# class_mapping = {index: label for index, label in enumerate(label_encoder.classes_)}
# print(class_mapping)
X_train, X_val, y_train, y_val = train_test_split(landmarks, labels_onehot, test_size=0.2, random_state=0)

model=Sequential([
    Dense(128,activation='relu',input_shape=(63,)),
    Dense(32,activation='relu'),
    # Dense(32,activation='relu'),
    Dense(6,activation='softmax'),
])
model.compile(optimizer='adam',loss=tf.keras.losses.CategoricalCrossentropy(),metrics=['accuracy'])
model.fit(X_train,y_train,validation_data=(X_val,y_val),epochs=25,batch_size=16)

model.save("gesture_recog.h5")