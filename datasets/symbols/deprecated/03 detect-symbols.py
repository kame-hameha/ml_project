#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By:     Kai Metzger
# Created School: Franz-Oberthuer-Schule Wuerzburg
# Created Email:  metzgerkai@franz-oberthuer-schule.de
# Created Date:   Mon April 07 23:00:00 UTC 2025
# Version:        1.0.1
# =============================================================================
"""This script loads the pretrained weights into a (64,32,16,4)-layer neuronal
   network that can distinguish the 4 symbols (circle, cross, square, 
   triangle). 
   The model was trained on some images that where recorded with 640x480 pixels 
   and downsized to 32 x 32 pixels to train the model on a Raspberry Pi 5 with
   4-8 GB RAM.
   """

# =============================================================================
# Imports
# =============================================================================

import cv2
import os
import numpy as np
from tensorflow.keras import layers, models, utils
import keras
import glob
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# Declare variables
# =============================================================================
img_size_x = 32
img_size_y = 32
img_dim = img_size_x * img_size_y

# Change the following paths to your dataset path
home_dir = os.path.expanduser("~")
dataset_for_training = "dataset1"
img_dir = home_dir + "/ml_project/datasets/symbols/" + dataset_for_training + "/data"
gt_dir = home_dir + "/ml_project/datasets/symbols/" + dataset_for_training + "/gt"
checkpoint_filepath = home_dir + "/ml_project/datasets/symbols/" + dataset_for_training + "/chpt/"

# =============================================================================
# Settings for image recording
# =============================================================================
cam = cv2.VideoCapture(0)
cam.set(3,640) # set Width
cam.set(4,480) # set Height
cv2.namedWindow("camera")

# =============================================================================
# # Define functions
# =============================================================================
# Change image size and convert to grayscale images
def pic_prep (image, x, y):
  image = cv2.resize(image, (y,x)) #größe ändern
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #in graustufen ändern
  image = image / 255 #normierung
  return image

# =============================================================================
# Create neural network with 4 layers and (64, 32, 16, 4) neurons per layer.
# =============================================================================
model = models.Sequential()
model.add(layers.Dense(128,input_dim=img_dim,activation='relu'))
model.add(layers.Dense(64,input_dim=img_dim,activation='relu'))
model.add(layers.Dense(8,input_dim=img_dim,activation='relu'))
model.add(layers.Dense(4,activation='sigmoid'))
opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='mean_squared_error', 
              optimizer=opt, 
              metrics=['accuracy'])


# =============================================================================
# Load pretrained dataset weights to e.g. test on new (unseen) data.
# =============================================================================
print(checkpoint_filepath + "chpt.keras")
model.load_weights(checkpoint_filepath + "chpt.keras")


# =============================================================================
# Main
# =============================================================================
while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k == 32:
        # SPACE pressed

        # read frame
        data_pred = np.zeros((1, img_size_x, img_size_y), dtype=float)
        img_pred = frame
        img_pred = pic_prep(img_pred, img_size_x, img_size_y)
        data_pred[0,:,:] = img_pred
        data_pred = data_pred.reshape(1,img_dim)

        # prediction
        result = model.predict(data_pred)
        result = np.round(result, decimals=2)

        # output of class that has hightest probability
        max_res = 0
        res_index = 4
        for i in range(0, 4, 1):
            if result[0,i] > max_res:
                max_res = result[0,i]
                res_index = i

        result = np.round(result * 100, decimals=2)

        if res_index == 0:
            print('Probability of ' + str(result[0,0]) + '%f or a cross!')
        elif res_index == 1:
            print('Probability of ' + str(result[0,1]) + '%f or a circle!')
        elif res_index == 2:
            print('Probability of ' + str(result[0,2]) + '%f or a triangle!')
        elif res_index == 3:
            print('Probability of ' + str(result[0,3]) + '%f or a square!')
        elif res_index == 4:
            print('Error!')

        print (result[0, :])


cam.release()
cv2.destroyAllWindows()