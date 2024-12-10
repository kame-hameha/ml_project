#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By:     Kai Metzger
# Created School: Franz-Oberthuer-Schule Wuerzburg
# Created Email:  metzgerkai@franz-oberthuer-schule.de
# Created Date:   Fri February 23 07:31:00 UTC 2024
# Version:        1.0
# =============================================================================
"""The Module has been build for creating a dataset with images + ground truth
   on a Raspberry Pi 4 with a standard USB camera. An image with a resolution 
   of 640px x 480px is recorded and you can control image recording plus ground 
   truth creation via pressing the following keys on the keyboard:
   - ESC:       Quit
   - SPACE:     Take picture (*without pressing BACKSPACE before) and save into 
                folders data/ground_truth. Increment counter by + 1.
   - BACKSPACE  Search folders and start image/gt file counters with highest 
                count (i. e. already taken 100 images --> images names 0 - 99, 
                next image with <100.png> and gt with 100.txt).
   - 0          Use label 0 for grount truth and write it to .txt file.
   - ...        "
   - 3          "

   You can change the script to fit your needs (i. e. create more classes,
   choose different keys, etc.)

   Camera window has to be active for user input (add new data, abort/exit, 
   etc.)
   """

# =============================================================================
# Imports
# =============================================================================
import numpy as np
import cv2
import glob
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# Config
# =============================================================================
# Create folders in dataset path for data, gt (ground truth) and 
# chpt (checkpoint) folder
dataset_path = "/home/pi/images/dataset"

# =============================================================================
# Camera setup
# =============================================================================
# Settings for image recording
cam = cv2.VideoCapture(0)
cam.set(3,640) # set Width
cam.set(4,480) # set Height       
cv2.namedWindow("camera")

# =============================================================================
# Variables
# =============================================================================
img_counter = 0

# =============================================================================
# Main loop to record new images, abort with CTRL+C
# =============================================================================
try:
    while True:
        # Image related stuff
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("camera", frame)

        # Press some keys to record images, and then press another key, 
        # i. e. 0 to write the first index into the text file.
        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        if k%256 == 8:
            # BACKSPACE pressed
            # Search folder for already recorded images&ground truth data
            list_data = glob.glob(dataset_path +"/data/*")
            list_gt = glob.glob(dataset_path + "/gt/*")
            #print(list_data)
            img_counter = len(list_data)
            gt_counter = len(list_gt)
            #print(count_gt
            
            # image count = GT-Anzahl?
            if (img_counter != gt_counter):
                print("Images and annotated data are not equal!")
                
            print("Continue with :" + str(img_counter))
        elif k%256 == 32:
            # SPACE pressed
            img_name = dataset_path + "/data/{}.png".format(img_counter)
            gt_name = dataset_path + "/gt/{}.txt".format(img_counter)

            # Save image ave in folder /dataset/<number+1>.png
            cv2.imwrite(img_name, frame)
            #print(frame.shape)
            print("{} written!".format(img_name))
            
            class_label = None
            print("Enter class label for current image, press ...\n \
                  0 = cross \n \
                  1 = circle \n \
                  2 = square \n \
                  3 = triangle")
            # Enter class label (0:circle or 1:rectangle or ...)
            # on keyboard, caution: Num-Pad does not work here!
            c = cv2.waitKey(-1)
            if c == 48: # ASCII 48 = key 0 on keyboard
                class_label = 0
            if c == 49: # ASCII 49 = key 1 on keyboard
                class_label = 1
            if c == 50: # ASCII 50 = key 2 on keyboard
                class_label = 2      
            if c == 51: # ASCII 51 = key 3 on keyboard
                class_label = 3
            print("Class = ", class_label)
            
            # GT in folder and /dataset/ground_truth/<number+1>.txt
            with open(gt_name, "w") as text_file:
                    text_file.write(str(class_label))
            print("{} written!".format(gt_name))

            # Increment by 1 for image and ground truth
            img_counter += 1

except KeyboardInterrupt:
    print("Programmabbruch!")
finally:
    # =========================================================================
    # Clean exit
    # =========================================================================
    cam.release()
    cv2.destroyAllWindows()