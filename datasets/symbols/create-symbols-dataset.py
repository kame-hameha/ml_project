#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By:     Kai Metzger
# Created School: Franz-Oberthuer-Schule Wuerzburg
# Created Email:  metzgerkai@franz-oberthuer-schule.de
# Created Date:   Fri February 23 07:31:00 UTC 2025
# Changed Date:   Wed November 12 09:41 UTC 2025
# Version:        1.0.2
# =============================================================================
"""The Module has been build for creating a dataset with images + ground truth
   on a Raspberry Pi 4 with a standard USB camera. An image with a resolution m
   of 640px x 480px is recorded and you can control image recording plus ground 
   truth creation via pressing the following keys on the keyboard:
   - ESC:       Quit
   - SPACE:     Take picture (*without pressing BACKSPACE before) and save into 
                folders data/ground_truth. Increment counter by + 1.
   - BACKSPACE  Search folders and start image/gt file counters with highest 
                count (i. e. already taken 100 images --> images names 0 - 99, 
                next image with <100.png> and gt with 100.txt).
   - TAB:       Enter number where to start counting from (i. e. 10, 360, ...).
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

# =============================================================================
# Imports and other stuff you could remove
# =============================================================================
import warnings
import os
warnings.filterwarnings("ignore")

# =============================================================================
# Config
# =============================================================================
# Create folders in dataset path for data, gt (ground truth) and 
# chpt (checkpoint) folder
dataset_path = "/home/pi/ml_project/datasets/symbols/dataset9"
hd_camera = True

# Check if dataset path exists, if not create it
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)
    os.makedirs(os.path.join(dataset_path, "data"))
    os.makedirs(os.path.join(dataset_path, "gt"))

# =============================================================================
# Camera setup
# =============================================================================
# Settings for image recording
cam = cv2.VideoCapture(0) 
# command to get USB cam formats:
# v4l2-ctl -d /dev/video0 --list-formats-ext

if hd_camera:
    print("Using hd camera!")
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    # Set properties. Each returns === True on success (i.e. correct resolution)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
else:
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Check image size:
#width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
#height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
#print(f"Frame Width: {width}, Frame Height: {height}")

cv2.namedWindow("camera")

# =============================================================================
# Functions
# =============================================================================
def read_number_with_cv2(window_name='Type number', prompt='Type number (Enter to finish, ESC to cancel): '):
    number_str = ''
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    base_img = np.zeros((120,600,3), dtype=np.uint8)

    while True:
        img = base_img.copy()
        text = prompt + number_str
        cv2.putText(img, text, (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,220,255), 2)
        cv2.imshow(window_name, img)

        key = cv2.waitKey(0)
        if key == -1:
            continue

        # Useful for debugging unknown codes - uncomment when needed:
        # print("raw key:", key, " hex:", hex(key))

        low = key & 0xFF  # low 8 bits - usually ascii for normal keys

        # Accept digits from ASCII low byte
        if ord('0') <= low <= ord('9'):
            number_str += chr(low)
            continue

        # Enter (CR/LF) handling: some platforms return 13, 10 or 13+256 etc.
        if low in (10, 13):
            break

        # ESC to cancel
        if low == 27:
            number_str = ''
            break

        # If the keypad produces non-ASCII codes on your platform (e.g., you saw 1360),
        # map them manually to digits using values you observed with the debug snippet:
        keypad_map = {
            # example entries (replace with numbers you saw while debugging)
            # 1360: '0', 1361: '1', ... 
        }
        if key in keypad_map:
            number_str += keypad_map[key]
            continue

        # ignore other keys
        # optionally show a brief flash or sound for invalid key

    cv2.destroyWindow(window_name)
    return int(number_str) if number_str else None

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
        if k == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        if k == 9:
            # Check if TAB is pressed on keyboard
            img_counter = read_number_with_cv2()
            print("Continue with :" + str(img_counter))
        if k == 8:
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
        elif k == 32:
            # SPACE pressed
            img_name = dataset_path + "/data/{}.png".format(img_counter)
            gt_name = dataset_path + "/gt/{}.txt".format(img_counter)

            # Save image ave in folder /dataset/<number+1>.png
            cv2.imwrite(img_name, frame)
            #print(frame.shape)
            print("{} written!".format(img_name))
            
            class_label = None
            print("Enter class label for current image, press ...\n \
                  0 = cross = Kreuz \n \
                  1 = circle = Kreis \n \
                  2 = triangle = Dreieck \n \
                  3 = square = Viereck")
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
    print("Program aborted!")
finally:
    # =========================================================================
    # Clean exit
    # =========================================================================
    cam.release()
    cv2.destroyAllWindows()
