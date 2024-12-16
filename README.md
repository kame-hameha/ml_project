# ml_project

This project involves setting up a Raspberry Pi (model 5) with necessary hardware and downloading the Raspbian OS. Users should have basic knowledge of Linux, Terminal, and Python. The project includes cloning a repository, installing software, creating and training datasets with a camera, and running inference on new images. Optionally, multiple datasets can be trained sequentially.

If you need further clarification or assistance, feel free to ask! 😊

## 0. Hardware requirements
- Get a Raspberry Pi (model 5)
- Power plug
- SD-card (minimum 32 GB)
- USB-camera 
- (Optional: case and fan)
Average price: Raspi5/plug/SD-card/case&fan = 80+5+10+5+10 € = ~120 €)

## 1. Download image and unzip it from: 

https://www.raspberrypi.com/software/

Tested Version:
2024-07-04-raspios-bookworm-arm64-full

## 2. (Optional) 
- Basic knowledge in Linux, Terminal (Konsole), und Python.
- AI/Machine Learning skills

## 3. Clone repository and change script/s to fit your needs:
### 3.1 Install software: 
Description: 
- Flash image to SD-card,
- update system,
- install software, libs and APIs as described in the file

File: software.bash 

(this is not a shell script that installs everything by running it, just copy&paste line-by-line and hit enter to install stuff)

### 3.2 Create a dataset of your choice

File: create-symbols-dataset.py

### 3.3 Training of recorded dataset (from 3.2)

File: training-symbols-dataset.ipynb (=train dataset)

### 3.4 (Optional: train multiple datasets, one after each other)

File: training-symbols-dataset-comparison.ipynb

### 3.5 Run inference on new images

File: detect-symbols.py

### 3.6 Download a dataset from dropbox:

https://www.dropbox.com/scl/fi/qyr4uz39u8aghfcbs9x8c/symbols.zip?rlkey=w93kuktegathzdp8lwpyau57q&st=qq2v7iuw&dl=0
