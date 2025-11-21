import cv2
import numpy as np
import pytesseract
import re
import hashlib
import requests
import time
from datetime import datetime
from enum import Enum
import argparse

# Try to import Picamera2, fallback to regular camera if not available
try:
    from picamera2 import Picamera2

    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False
    print("Picamera2 not available, using standard camera")


class CameraRole(Enum):
    ENTRANCE = "entrance"
    EXIT = "exit"
    MONITORING = "monitoring"


class ParkingCamera:
    def __init__(self, camera_id, role, server_url, salt_key, use_picamera=False):
        """
        Initialize parking camera with specific role

        Args:
            camera_id: Unique identifier for this camera
            role: CameraRole enum (ENTRANCE, EXIT, or MONITORING)
            server_url: URL of the parking management server
            salt_key: Secret salt for hashing plate numbers
            use_picamera: Whether to use Picamera2 (Raspberry Pi camera)
        """
        self.camera_id = camera_id
        self.role = role
        self.server_url = server_url
        self.salt_key = salt_key
        self.use_picamera = use_picamera and PICAMERA_AVAILABLE
        self.picam2 = None

    def init_picamera(self):
        """Initialize Picamera2"""
        if not self.use_picamera:
            return False

        try:
            self.picam2 = Picamera2()
            config = self.picam2.create_preview_configuration(
                main={"format": 'BGR888', "size": (1920, 1080)},
                controls={"FrameRate": 30}
            )
            self.picam2.configure(config)
            self.picam2.start()
            time.sleep(2)
            print("Picamera2 initialized")
            return True
        except Exception as e:
            print(f"Error initializing Picamera2: {e}")
            return False

    def capture_frame(self, cap=None):
        """Capture frame from camera"""
        if self.use_picamera and self.picam2:
            frame = self.picam2.capture_array()
            # Picamera2 returns RGB, convert to BGR for OpenCV
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif cap:
            ret, frame = cap.read()
            return frame if ret else None
        return None

    def hash_plate(self, plate_number):
        """
        Hash the plate number with salt using SHA-256
        """
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
        }

        try:
            response = requests.post(
                f"{self.server_url}/api/register_plate",
                json=payload,
                timeout=5
            )

            if response.status_code == 200:
                print(f"✓ Sent to server: {hashed_plate[:16]}...")
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
        # Initialize camera
        cap = None
        if self.use_picamera:
            if not self.init_picamera():
                print("Failed to initialize Picamera")
                return
        else:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                print("Error: Could not open video source")
                return

        print(f"\n{'=' * 60}")
        print(f"  PARKING CAMERA - {self.role.value.upper()}")
        print(f"  Camera ID: {self.camera_id}")
        print(f"  Server: {self.server_url}")
        if self.use_picamera:
            print(f"  Camera: Picamera2 (Raspberry Pi)")
        else:
            print(f"  Camera: OpenCV VideoCapture")
        print(f"{'=' * 60}")
        print("\nControls:")
        print("  'q' - Quit")
        print("  's' - Save current detection")
        print("\nStatus: Running...\n")

        frame_count = 0
        last_valid_detection = None
        detection_confidence = 0
        sent_plates = {}  # Track sent plates to avoid duplicates

        try:
            while True:
                # Capture frame
                frame = self.capture_frame(cap)

                if frame is None:
                    if cap:
                        # If using video file, loop back to start
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

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

        finally:
            # Cleanup
            if self.use_picamera and self.picam2:
                self.picam2.stop()
            elif cap:
                cap.release()

            cv2.destroyAllWindows()
            print("\nCamera stopped")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='License plate detection with Picamera2 support')
    parser.add_argument('--camera-id', default='CAM_ENTRANCE_01', help='Camera identifier')
    parser.add_argument('--role', default='entrance', choices=['entrance', 'exit', 'monitoring'],
                        help='Camera role')
    parser.add_argument('--server', default='http://localhost:5000', help='Server URL')
    parser.add_argument('--source', default=0, help='Video source (0 for webcam, or video file path)')
    parser.add_argument('--picamera', action='store_true', help='Use Raspberry Pi camera')
    parser.add_argument('--salt', default='your_secret_salt_key_here_change_this',
                        help='Salt key for hashing')
    parser.add_argument('--skip-frames', type=int, default=2, help='Process every Nth frame')
    parser.add_argument('--send-interval', type=int, default=30,
                        help='Minimum frames between sending same plate')

    args = parser.parse_args()

    # Parse video source
    try:
        video_source = int(args.source)
    except ValueError:
        video_source = args.source

    # Map role string to enum
    role_map = {
        'entrance': CameraRole.ENTRANCE,
        'exit': CameraRole.EXIT,
        'monitoring': CameraRole.MONITORING
    }

    # Create camera
    camera = ParkingCamera(
        camera_id=args.camera_id,
        role=role_map[args.role],
        server_url=args.server,
        salt_key=args.salt,
        use_picamera=args.picamera
    )

    # Process video
    camera.process_video(
        source=video_source,
        skip_frames=args.skip_frames,
        send_interval=args.send_interval
    )