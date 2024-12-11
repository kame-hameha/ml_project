# ml_project

## 0. Hardware requirements
- Get a Raspberry Pi (model 5)
- Power plug
- SD-card (minimum 32 GB)
- (Optional: case and fan)
Average price: Raspi5/plug/SD-card/case&fan = 80+5+10+5 € = 100 €)

## 1. Download image and unzip it from: https://www.raspberrypi.com/software/
Tested Version:
2024-07-04-raspios-bookworm-arm64-full

## 2. (Optional) 
- Basic knowledge in Linux, Terminal (Konsole), und Python.
- AI/Machine Learning skills

## 3. Clone repository and change script/s to fit your needs:
### 3.1 Install software: 
Description: Flash image to SD-card, update system, install software, libs and APIs as described in the file

File: software.bash 

### 3.2 Create a dataset of your choice

File: create-symbols-dataset.py

### 3.3 Training of recorded dataset (from 3.2)

File: training-symbols-dataset.ipynb (=train dataset)

### 3.4 (Optional: train multiple datasets, one after each other)

File: training-symbols-dataset-comparison.ipynb

### 3.5 Run inference on new images

File: detect-symbols.py
