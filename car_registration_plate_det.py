import cv2
import numpy as np
import pytesseract
import re
from datetime import datetime
import json
from pathlib import Path


class CarRegistrationDetector:
    """
    Simple license plate OCR with entry/exit timestamp logging
    """

    def __init__(self, gate_type='entry', log_file='vehicle_log.json'):
        """
        Args:
            gate_type: 'entry' or 'exit'
            log_file: JSON file to store timestamps
        """
        self.gate_type = gate_type
        self.log_file = log_file
        self.vehicle_log = self.load_log()

        # Cooldown to avoid duplicate detections (in frames)
        self.recent_plates = {}
        self.cooldown_frames = 90  # ~3 seconds at 30fps

    def load_log(self):
        """Load existing log"""
        if Path(self.log_file).exists():
            with open(self.log_file, 'r') as f:
                return json.load(f)
        return []

    def save_log(self):
        """Save log to file"""
        with open(self.log_file, 'w') as f:
            json.dump(self.vehicle_log, f, indent=2)

    def detect_plate_region(self, frame):
        """Find potential plate region using edge detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.bilateralFilter(gray, 11, 17, 17)
        edges = cv2.Canny(blur, 30, 200)

        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

            if len(approx) == 4:  # Rectangular
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / float(h)

                # Typical plate aspect ratio and size
                if 2.0 < aspect_ratio < 6.0 and 2000 < w * h < 100000:
                    return frame[y:y + h, x:x + w], (x, y, w, h)

        return None, None

    def preprocess_plate(self, plate_img):
        """Prepare plate image for OCR"""
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

        # Upscale for better OCR
        gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Denoise
        gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

        # Threshold
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)

        return thresh

    def read_plate(self, plate_img):
        """Perform OCR on plate"""
        try:
            processed = self.preprocess_plate(plate_img)

            # Tesseract config for plates
            config = '--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            text = pytesseract.image_to_string(processed, config=config)

            # Clean text
            text = re.sub(r'[^A-Z0-9]', '', text.strip().upper())

            # Validate (4-8 characters, has letters and numbers)
            if 4 <= len(text) <= 8 and any(c.isalpha() for c in text) and any(c.isdigit() for c in text):
                return text

        except Exception as e:
            print(f"OCR Error: {e}")

        return None

    def log_vehicle(self, plate_text):
        """Log vehicle with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        entry = {
            'plate': plate_text,
            'gate': self.gate_type,
            'timestamp': timestamp
        }

        self.vehicle_log.append(entry)
        self.save_log()

        print(f"{'ENTRY' if self.gate_type == 'entry' else 'EXIT'}: {plate_text} at {timestamp}")

    def update_cooldown(self):
        """Remove expired cooldowns"""
        expired = [plate for plate, frames in self.recent_plates.items() if frames <= 0]
        for plate in expired:
            del self.recent_plates[plate]

        for plate in self.recent_plates:
            self.recent_plates[plate] -= 1

    def process_video(self, video_path):
        """Process video file for plate detection"""
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return

        print(f"\n{'=' * 50}")
        print(f"Processing {self.gate_type.upper()} gate: {video_path}")
        print(f"{'=' * 50}\n")

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Process every 5th frame for performance
            if frame_count % 5 != 0:
                continue

            # Update cooldowns
            self.update_cooldown()

            # Detect plate region
            plate_img, bbox = self.detect_plate_region(frame)

            if plate_img is not None:
                # Read plate text
                plate_text = self.read_plate(plate_img)

                if plate_text and plate_text not in self.recent_plates:
                    # Log it
                    self.log_vehicle(plate_text)

                    # Add cooldown
                    self.recent_plates[plate_text] = self.cooldown_frames

        cap.release()
        print(f"\nProcessing complete. Total detections: {len(self.vehicle_log)}")

    def calculate_duration(self, plate_number):
        """Calculate parking duration for a plate number"""
        entries = [e for e in self.vehicle_log if e['plate'] == plate_number and e['gate'] == 'entry']
        exits = [e for e in self.vehicle_log if e['plate'] == plate_number and e['gate'] == 'exit']

        if not entries:
            return None

        entry_time = datetime.strptime(entries[-1]['timestamp'], "%Y-%m-%d %H:%M:%S")

        if exits:
            exit_time = datetime.strptime(exits[-1]['timestamp'], "%Y-%m-%d %H:%M:%S")
            duration = exit_time - entry_time
            return duration

        # Still parked
        return None


def calculate_all_durations(log_file='vehicle_log.json'):
    """Calculate parking durations for all vehicles"""
    if not Path(log_file).exists():
        print(f"Log file {log_file} not found")
        return

    with open(log_file, 'r') as f:
        logs = json.load(f)

    # Group by plate
    plates = {}
    for entry in logs:
        plate = entry['plate']
        if plate not in plates:
            plates[plate] = {'entries': [], 'exits': []}

        if entry['gate'] == 'entry':
            plates[plate]['entries'].append(entry['timestamp'])
        else:
            plates[plate]['exits'].append(entry['timestamp'])

    print("\n" + "=" * 60)
    print("PARKING DURATION REPORT")
    print("=" * 60 + "\n")

    for plate, data in plates.items():
        print(f"Plate: {plate}")

        for i, entry_time_str in enumerate(data['entries']):
            entry_time = datetime.strptime(entry_time_str, "%Y-%m-%d %H:%M:%S")

            if i < len(data['exits']):
                exit_time = datetime.strptime(data['exits'][i], "%Y-%m-%d %H:%M:%S")
                duration = exit_time - entry_time

                hours = duration.total_seconds() / 3600
                print(f"  Entry: {entry_time_str}")
                print(f"  Exit:  {data['exits'][i]}")
                print(f"  Duration: {duration} ({hours:.2f} hours)")
            else:
                print(f"  Entry: {entry_time_str}")
                print(f"  Exit:  Still parked")

            print()

    print("=" * 60 + "\n")


# Example usage
if __name__ == "__main__":
    # Set Tesseract path if needed (Windows)
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    # Process entry gate video. Here we can plug the entry_gate footage.
    entry_detector = CarRegistrationDetector(gate_type='entry', log_file='vehicle_log.json')
    entry_detector.process_video('license_plate_detection.mp4')

    # Process exit gate video. Here we can plug the exit_gate footage.
    exit_detector = CarRegistrationDetector(gate_type='exit', log_file='vehicle_log.json')
    exit_detector.process_video('license_plate_detection.mp4')

    # Calculate durations
    calculate_all_durations('vehicle_log.json')