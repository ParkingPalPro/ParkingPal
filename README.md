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
- [Parking Space Occupancy Detection (`parking_detector.py`)](#parking-space-occupancy-detectionparking_space_detpy)
  - [Example Usage](#example-usage-1)
- [Server](#server)

---

---

## Project Background

ParkingPal is a real-time parking management system that integrates computer vision-based parking space detection with a web interface for monitoring and session management.

---

## Project Structure
```bash
.
├── README.md
├── edge_device   # Directory with code for edge devices 
│   └── car_plate.py    # License plate detection and OCR
│   └── parking_detector.py    # Parking space occupancy detection
│   └── test_videos   # Videos that could be used for testing
│       ├── entry.mp4
│       ├── exit.mp4
│       └── parking_uia.mp4
├── requirements.txt    # Python dependencies
└── server    # Flask server
    ├── config.py
    ├── instance
    │   └── app.db # application database
    ├── main.py
    ├── parkingpal
    │   ├── __init__.py
    │   ├── blueprints
    │   │   ├── __init__.py
    │   │   ├── admin
    │   │   │   ├── __init__.py
    │   │   │   └── routes.py
    │   │   ├── api   # api endpoint for edge devices
    │   │   │   ├── __init__.py
    │   │   │   ├── cameras.py  # camera configs, snapshots, get available monitoring cameras 
    │   │   │   ├── parking_sessions.py   # receive hashed plate number with entry and leave timestamp 
    │   │   │   └── parking_spaces.py   # get polygon (ROI) from the server, updated parking space availability, get status
    │   │   ├── auth
    │   │   │   ├── __init__.py
    │   │   │   └── routes.py
    │   │   └── user
    │   │       ├── __init__.py
    │   │       └── routes.py
    │   ├── extensions.py
    │   ├── models    # database models
    │   │   ├── __init__.py
    │   │   ├── camera_config.py
    │   │   ├── parking_session.py
    │   │   ├── parking_space.py
    │   │   └── user.py
    │   ├── static
    │   │   └── images
    │   ├── templates   # HTML templates
    │   │   ├── admin.html
    │   │   ├── base.html
    │   │   ├── dashboard.html
    │   │   ├── login.html
    │   │   ├── parking_session.html
    │   │   └── register.html
    │   └── utils.py
    └── run.py    # Entry point for running the server

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
- Update system (Raspberry pi)
    ```bash
    sudo apt update 
    sudo apt full-upgrade
    ```
- Install global dependencies (Raspberry pi)
    ```bash
    sudo apt install -y python3-picamera2 --no-install-recommends
    ```
    ```bash
    sudo apt install -y python3-picamera2
    ```
- Test if installed correctly (Raspberry pi)
    ```bash
    python3 -c "from picamera2 import Picamera2; print('Picamera2 OK')"
    ```
  Should print Picamera2 OK, if not try fix the installation with `sudo apt remove -y` and `sudo apt --fix-broken install -y`

- Create virtual environment (recommended)
    ```bash
    python -m venv .venv --system-site-packages
    source .venv/bin/activate      # Linux / macOS
    .venv\Scripts\activate         # Windows
    ```
  
- Install dependencies
    ```bash
    pip install -r requirements.txt 
  ```
- If you on other system then Raspberry you may install numpy as well
    ```bash
    pip install numpy
    ```

### Raspberry Pi camera
The current setup was tested on Raspberry Pi 5B with 4GB memory storage and Raspberry Pi Camera Module 3 Wide.
It may have some issues on earlier versions of Raspberry and may not work with earlier versions of camera.  

If you installed all this should be enough to run code in ```edge_device``` directory.
If that did not work follow the official Raspberry manual for installing dependencies:
https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf

https://www.raspberrypi.com/documentation/computers/camera_software.html#rpicam-apps


However, in some cases that might not be enough, you may that clone and build ```libcam``` on your raspberry. 
Raspberry provides their own fork: 
https://github.com/raspberrypi/libcamera

These steps might resolve most of the issues. 
If you still run on some issues you can as well run edge device code on other platforms, the code uses the ```open-cv```
video capture, just do **NOT**  include ```--picamera``` flag when running the script. 
You can as well provide video file as source then you do not need to use web camera at all.  


### Raspberry Pi camera
The current setup was tested on Raspberry Pi 5B with 4GB memory storage and Raspberry Pi Camera Module 3 Wide.
It may have some issues on earlier versions of Raspberry and may not work with earlier versions of camera.  

The ```requirements.txt``` includes picamera2 package, this should be enough to run code in ```edge_device``` directory.
If that did not work follow the official Raspberry manual for installing dependencies:
https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf

https://www.raspberrypi.com/documentation/computers/camera_software.html#rpicam-apps


However, in some cases that might not be enough, you may that clone and build ```libcam``` on your raspberry. 
Raspberry provides their own fork: 
https://github.com/raspberrypi/libcamera

These steps might resolve most of the issues. 
If you still run on some issues you can as well run edge device code on other platforms, the code uses the ```open-cv```
video capture, just do **NOT**  include ```--picamera``` flag when running the script. 
You can as well provide video file as source then you do not need to use web camera at all.  


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

## License Plate Recognition (`edge_device/car_plate.py`)

This script:
- Detects rectangular contours that resemble license plates.
- Crops the detected plate.
- Uses **Tesseract OCR** to extract the text.


### Example usage
On Windows
```bash
# Using webcam
python car_plate.py --camera-id CAM_ENTRANCE_01 --role entrance --server http://localhost:5000

# Using video file
python car_plate.py --source entry.mp4 --role entrance
```
On Raspberry Pi (Picamera2)
```bash
python car_plate.py --camera-id CAM_ENTRANCE_01 --role entrance --picamera --server http://192.168.1.100:5000
```
#### Flags
`--camera-id`
Unique ID for the camera. Example: CAM_ENTRANCE_01.

`--role`
Defines the camera’s purpose.
Options: entrance or exit.

`--server`
URL of the backend server that receives the detections with `http://`.

`--source`
Video input.
0 = default webcam
Or path to a video file.

`--picamera`
Enables Raspberry Pi Picamera2 instead of a normal webcam.

## Parking Space Occupancy Detection(`edge_device/parking_api.py`)



This script:
- Uses background subtraction and edge detection to detect motion or car presence.

- Detects occupied vs. free spaces in real time.

### Example usage
Uses default webcam, default camera ID, and local server.
```bash
python parking_api.py
```
Please ensure you are running the server first and raspberry pi can access it. The detection will not work without config from the server.
Set camera id, source video and server to send data to
```bash
python3 parking_detector.py  --camera-id CAM_EAST_WING --source ./test_videos/parking_uia.mp4 --server http://192.168.0.50:5000
```

Set camera id, uses picamera and server to send data to
```bash
python3 parking_detector.py  --camera-id CAM_EAST_WING --picamera --server http://192.168.0.50:5000
```

You can as well adjust the default values in the code directly, you will found it at the end of the file
```python
# Create detector
detector = HybridParkingDetector(
    camera_id=args.camera_id, # string
    video_source=video_source, # 0 or path to video file
    use_picamera=args.picamera, # True or False
    server_url=args.server, # string
    snapshot_interval=args.snapshot_interval, # integer
    config_check_interval=args.config_interval # integer
    )
```

The code have threshold that can be adjusted depending on the environment.
```python
OCCUPATION_THRESHOLD = 0.28
```


#### Command-line flags
`--camera-id`
Unique identifier for this camera.
Used so the server knows which parking lot the data belongs to.
Default: CAM_PARKING_01

`--server` URL of the backend server that receives snapshots and space status.
Default: http://localhost:5000

`--source`
Video input.
0 → default webcam
Or path to a video file (e.g. /home/pi/video.mp4)

`--picamera`
Enables the Raspberry Pi Picamera2 instead of USB webcam.
Works only if Picamera2 is installed.

`--snapshot-interval`
How often (in seconds) a frame is uploaded to the server for monitoring (admin UI).
Default: 10

`--config-interval`
How often (in seconds) the camera fetches updated parking-space polygons from the server.
Default: 5

## Server
Move to `server/` directory:
```bash
cd server/
```

Start the server
```bash
python run.py
```

The server runs on `localhost:5000`

you can access the admin panel with the following credentials on `/admin`
```python
admin_user = User(
    username='admin',
    email='admin@example.com',
    password=hash_password('Password1.'),
    is_admin=True
)
```
When the monitoring camera is connected to the server, it sends snapshots to the server. You can create or adjust existing ROI polygons from the admin panel.

### Admin Interface (`/admin`)

Purpose: Configure parking space detection zones
Features:

- View live camera snapshots from detection systems
- Draw polygons to define parking space boundaries
- Assign space numbers manually
- Save configuration to camera devices
- Multi-camera support


Access: Admin privileges required

<img width="720" alt="image" src="https://github.com/user-attachments/assets/bdaabbea-88f1-460c-938d-12b99a2940f2" />
<img width="720" alt="image" src="https://github.com/user-attachments/assets/c9bc2f63-05c0-44ae-85a6-e5eb8302b6b3" />
<img width="720" alt="image" src="https://github.com/user-attachments/assets/f28b8919-5719-479a-9cdc-3cfeb7fae0db" />



### Dashboard (`/dashboard`)

Purpose: Real-time parking lot overview
Features:

- Live statistics (total spaces, available, occupied)
- Visual grid showing each parking space status
- Color-coded spaces: green (available), orange (occupied)
- Auto-refresh every 3 seconds
- Animated transitions when space status changes


Access: Login required

<img width="720" alt="image" src="https://github.com/user-attachments/assets/0a9e6373-e756-4b6a-95a8-9de010cc1772" />


### Parking Session Retrieval (`/parking-session`)

Purpose: View parking session details by license plate
Features:

- Enter license plate number to lookup session
- Displays entry time, exit time, and duration
- Shows "Still Active" for ongoing sessions (no exit time)
- Automatically deletes completed sessions after retrieval
- Active sessions remain in database for future lookup


Access: Login required
<img width="720" alt="image" src="https://github.com/user-attachments/assets/2b8dc8a1-5dd6-4d17-a409-79f8b6346d4a" />


