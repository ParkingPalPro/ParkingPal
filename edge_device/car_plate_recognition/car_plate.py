import cv2
import numpy as np
import pytesseract
import re
import hashlib
import requests
from datetime import datetime
from enum import Enum


class CameraRole(Enum):
    ENTRANCE = "entrance"
    EXIT = "exit"
    MONITORING = "monitoring"


class ParkingCamera:
    def __init__(self, camera_id, role, server_url, salt_key):
        """
        Initialize parking camera with specific role

        Args:
            camera_id: Unique identifier for this camera
            role: CameraRole enum (ENTRANCE, EXIT, or MONITORING)
            server_url: URL of the parking management server
            salt_key: Secret salt for hashing plate numbers
        """
        self.camera_id = camera_id
        self.role = role
        self.server_url = server_url
        self.salt_key = salt_key

    def hash_plate(self, plate_number):
        """
        Hash the plate number with salt using SHA-256
        """
        salted_plate = f"{plate_number}{self.salt_key}"
        return hashlib.sha256(plate_number.encode()).hexdigest()

    def send_to_server(self, plate_number):
        """
        Send hashed plate data to server with timestamp and camera role
        """
        hashed_plate = self.hash_plate(plate_number)
        timestamp = datetime.now().isoformat()

        payload = {
            "camera_id": self.camera_id,
            "role": self.role.value,
            "hashed_plate": hashed_plate,
            "timestamp": timestamp,
            "original_plate": plate_number  # Only for entrance to associate email
        }

        try:
            response = requests.post(
                f"{self.server_url}/plate_event",
                json=payload,
                timeout=5
            )

            if response.status_code == 200:
                print(f"✓ Sent to server: {plate_number} -> {hashed_plate[:16]}...")
                result = response.json()
                if result.get("message"):
                    print(f"  Server: {result['message']}")
                return True
            else:
                print(f"✗ Server error: {response.status_code}")
                return False

        except requests.exceptions.RequestException as e:
            print(f"✗ Connection error: {e}")
            return False

    def is_valid_plate_text(self, text):
        """Validate if text looks like a license plate"""
        if not text or len(text) < 3:
            return False

        clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())

        if len(clean_text) < 3:
            return False

        letters = sum(c.isalpha() for c in clean_text)
        numbers = sum(c.isdigit() for c in clean_text)

        return letters >= 1 and numbers >= 1

    def is_valid_plate_shape(self, contour, frame_shape):
        """Check if contour has valid license plate proportions"""
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)

        if aspect_ratio < 1.5 or aspect_ratio > 6:
            return False

        if w < 60 or h < 20:
            return False

        frame_area = frame_shape[0] * frame_shape[1]
        contour_area = cv2.contourArea(contour)
        if contour_area > frame_area * 0.5:
            return False

        return True

    def detect_plate(self, frame):
        """Detect and read license plate from a frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(bfilter, 30, 200)

        keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = keypoints[0]
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]

        location = None
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)

            if len(approx) == 4 and self.is_valid_plate_shape(contour, gray.shape):
                location = approx
                break

        if location is None:
            return None, None, None

        mask = np.zeros(gray.shape, dtype="uint8")
        cv2.drawContours(mask, [location], 0, 255, -1)

        (y, x) = np.where(mask == 255)
        if len(x) == 0 or len(y) == 0:
            return None, None, None

        x1, y1 = np.min(x), np.min(y)
        x2, y2 = np.max(x), np.max(y)

        cropped = frame[y1:y2 + 1, x1:x2 + 1]

        cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        cropped_gray = cv2.bilateralFilter(cropped_gray, 11, 17, 17)

        _, cropped_thresh = cv2.threshold(cropped_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        text = pytesseract.image_to_string(
            cropped_thresh,
            config='--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        )
        text = text.strip()

        if not self.is_valid_plate_text(text):
            return None, None, None

        text = re.sub(r'[^A-Z0-9]', '', text.upper())

        return cropped, text, location

    def process_video(self, source=0, skip_frames=2, send_interval=30):
        """
        Process video from file or webcam

        Args:
            source: 0 for webcam, or path to video file
            skip_frames: process every Nth frame
            send_interval: minimum frames between sending same plate
        """
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            print("Error: Could not open video source")
            return

        print(f"\n{'=' * 60}")
        print(f"  PARKING CAMERA - {self.role.value.upper()}")
        print(f"  Camera ID: {self.camera_id}")
        print(f"  Server: {self.server_url}")
        print(f"{'=' * 60}")
        print("\nControls:")
        print("  'q' - Quit")
        print("  's' - Save current detection")
        print("\nStatus: Running...\n")

        frame_count = 0
        last_valid_detection = None
        detection_confidence = 0
        sent_plates = {}  # Track sent plates to avoid duplicates

        while True:
            ret, frame = cap.read()

            if not ret:
                print("End of video or cannot read frame")
                break

            frame_count += 1
            display_frame = frame.copy()

            if frame_count % skip_frames == 0:
                cropped, text, location = self.detect_plate(frame)

                if text and location is not None:
                    last_valid_detection = (cropped, text, location)
                    detection_confidence = 10

                    # Send to server if not recently sent
                    if text not in sent_plates or (frame_count - sent_plates[text]) > send_interval:
                        self.send_to_server(text)
                        sent_plates[text] = frame_count

            if last_valid_detection is not None and detection_confidence > 0:
                cropped, text, location = last_valid_detection

                cv2.drawContours(display_frame, [location], -1, (0, 255, 0), 3)

                cv2.putText(display_frame, text,
                            (location[0][0][0], location[0][0][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                if cropped is not None and cropped.size > 0:
                    cv2.imshow("Detected Plate", cropped)

                detection_confidence -= 1

            # Display camera info and status
            role_color = (0, 255, 255) if self.role == CameraRole.ENTRANCE else \
                (0, 165, 255) if self.role == CameraRole.EXIT else (255, 255, 0)

            cv2.putText(display_frame, f"Role: {self.role.value.upper()}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, role_color, 2)

            status = f"Plate: {last_valid_detection[1] if last_valid_detection else 'Searching...'}"
            cv2.putText(display_frame, status, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow(f"Camera {self.camera_id} - {self.role.value}", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and last_valid_detection is not None:
                cropped, text, _ = last_valid_detection
                filename = f"plate_{self.camera_id}_{text}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(filename, cropped)
                print(f"Saved: {filename}")

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Configuration
    SERVER_URL = "http://localhost:5000"  # Change to your server URL
    SALT_KEY = "your_secret_salt_key_here_change_this"  # IMPORTANT: Change this!

    # Example: Entrance Camera
    camera = ParkingCamera(
        camera_id="CAM_ENTRANCE_01",
        role=CameraRole.EXIT,
        server_url=SERVER_URL,
        salt_key=SALT_KEY
    )

    # For video file:
    camera.process_video('./videos/entry.mp4')

    # For webcam:
    # camera.process_video(0)

    # Example: Exit Camera
    # camera = ParkingCamera(
    #     camera_id="CAM_EXIT_01",
    #     role=CameraRole.EXIT,
    #     server_url=SERVER_URL,
    #     salt_key=SALT_KEY
    # )
    # camera.process_video(0)

    # Example: Monitoring Camera
    # camera = ParkingCamera(
    #     camera_id="CAM_MONITOR_01",
    #     role=CameraRole.MONITORING,
    #     server_url=SERVER_URL,
    #     salt_key=SALT_KEY
    # )
    # camera.process_video(0)