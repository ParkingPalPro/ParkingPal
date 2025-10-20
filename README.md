# ParkingPal 
### Smart Parking and License Plate Recognition System

This project demonstrates two main computer vision tasks using **OpenCV** and **Tesseract OCR**:

1. **License plate recognition** – detecting and reading car license plates from images or video frames.
2. **Parking space occupancy detection** – tracking which parking spaces are occupied or free using video analysis.

The goal is to build an automated parking management system for a Raspberry Pi–based setup.

---

## Table of Contents
- [Project Background](#project-background)
- [Project Structure](#project-structure)
- [Installation Guide](#installation-guide)
  - [Clone the Repository](#clone-the-repository)
  - [Create a Virtual Environment](#create-virtual-environment)
  - [Install Dependencies](#install-dependencies)
  - [Install Tesseract-OCR](#install-tesseract-ocr)
    - [For Windows](#for-windows)
    - [For macOS](#for-macos)
    - [For Linux](#for-linux)
  - [Add Tesseract to Environment Variables](#add-tesseract-to-the-environment-variables)
    - [Windows Setup](#for-windows-1)
    - [macOS/Linux Setup](#for-macos-and-linux)
  - [Verify Tesseract Installation](#verify-tesseract-installation)
- [License Plate Recognition (`car_plate.py`)](#license-plate-recognition-car_platepy)
  - [Example Usage](#example-usage)
- [Parking Space Occupancy Detection (`parking_space_det.py`)](#parking-space-occupancy-detectionparking_space_detpy)
  - [Example Usage](#example-usage-1)

---

---

## Project Background

Since we did not receive the correct **camera adapter cable** for the **Raspberry Pi 5**, testing was performed using **recorded videos and sample images**.  

In the final version:
- **One camera** will take pictures of cars entering through the **entry gate** to read their **license plates**.
- **Other cameras** will monitor **parking space occupancy** to detect which spaces are free or occupied.

---

## Project Structure
```bash
.
├── car_plate.py # License plate detection and OCR
├── parking_space_det.py # Parking space occupancy detection
├── requirements.txt # Python dependencies
├── images/ or videos/ # Test images and videos
└── README.md
```

---
## Installation Guide

- Clone the repository
    ```bash
    git clone git@github.com:ParkingPalPro/ParkingPal.git
    ```
  ```bash
  cd ParkingPal/
  ```
- Create virtual environment (recommended)
    ```bash
    python -m venv venv
    source venv/bin/activate      # Linux / macOS
    venv\Scripts\activate         # Windows
    ```
  
- Install dependencies
    ```bash
    pip install -r requirements.txt 
  ```

### Download and Install Tesseract-OCR

Tesseract is required for license plate recognition in car_plate.py.

#### For Windows:
Go to the official Tesseract GitHub release page: https://github.com/UB-Mannheim/tesseract/wiki
Download the Windows installer (tesseract-ocr-setup.exe) from the releases section.
Run the installer and complete the installation process.

<img width="1351" height="418" alt="image" src="https://github.com/user-attachments/assets/94b284be-75e8-4879-818a-b9e20f98ea61" />


#### For macOS:
```bash
brew install tesseract
```

#### For Linux:
```bash
sudo apt update
sudo apt install tesseract-ocr
```

### Add Tesseract to the Environment Variables

#### For Windows:
1. Open the Start Menu, search for Environment Variables, and select Edit the system environment variables. 
2. In the System Properties window, click the Environment Variables button. 
3. Under System variables, find and select the Path variable, then click Edit. 
4. Click New and add the path to the tesseract.exe file (usually located in C:\Program Files\Tesseract-OCR\). 
5. Click OK to save changes.

#### For macOS and Linux:
1. Open a terminal and add the Tesseract path to the shell configuration file (`.bashrc`, `.zshrc`, or `.bash_profile`):
    ```bash
    export PATH="/usr/local/bin/tesseract:$PATH"
    ```
2. Save the file and run the following command to apply changes:
    ```bash
    source ~/.bashrc  # or source ~/.zshrc
    ```
### Verify Tesseract Installation
After adding Tesseract to our environment variables, open a terminal (or Command Prompt on Windows) and type:
```bash
  tesseract --version
```

## License Plate Recognition (`car_plate.py`)

This script:
- Loads an image of a car.
- Detects rectangular contours that resemble license plates.
- Crops the detected plate.
- Uses **Tesseract OCR** to extract the text.


### Example usage

```bash
python3 car_plate.py
```

## Parking Space Occupancy Detection(`parking_space_det.py`)



This script:
- Uses background subtraction and edge detection to detect motion or car presence.

- Allows you to interactively define parking spaces on the first frame.

- Saves parking area positions in parking_positions.pkl.

- Detects occupied vs. free spaces in real time.

### Example usage

```bash
python parking_space_det.py
```

Then a new window will open where you can add or remove paring spaces using 4 dots.
A simple interactions will be printed to console 
Example
```bash
Loaded 12 parking spaces

=== Parking Space Setup Mode ===
LEFT CLICK to add points (4 points per space)
Points should be in order: top-left, top-right, bottom-right, bottom-left
LEFT CLICK to cancel current drawing
CLICK on existing space to REMOVE it
Press 's' to SAVE and 'q' to QUIT setup
================================
```
**Set up mode**
<img width="2367" height="1335" alt="image" src="https://github.com/user-attachments/assets/5c48d7e4-6d60-4307-a41b-70c5935fd0ea" />

After you finish the setup click "s" to save and "q" to quit the set up mode then the two windos may appear one with the mask that uses edge detection and background substraction and the second may obser below:
**Mask**
<img width="2542" height="1380" alt="image" src="https://github.com/user-attachments/assets/14f497f4-93e4-45e1-a0b4-61fdf635605d" />
**Result**
<img width="2407" height="1319" alt="image" src="https://github.com/user-attachments/assets/9acda549-757f-448f-81d1-8025f0ff47c9" />


