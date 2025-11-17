import cv2
import numpy as np
import pickle
from pathlib import Path
from picamera2 import Picamera2
import time
import requests
import json
from threading import Thread, Lock


class ParkingSpaceDetector:
    """
    Parking space occupancy detection with server integration
    """

    def __init__(self, video_source, parking_positions_file='parking_positions.pkl',
                 use_picamera=False, server_url='http://localhost:5000', max_display_width=1280,
                 max_display_height=720):
        self.video_source = video_source
        self.parking_positions_file = parking_positions_file
        self.parking_spaces = []
        self.use_picamera = use_picamera
        self.picam2 = None
        self.server_url = server_url

        # Display settings
        self.max_display_width = max_display_width
        self.max_display_height = max_display_height
        self.scale_factor = 1.0
        self.original_size = None

        # Drawing state
        self.drawing = False
        self.points = []
        self.temp_point = None

        # No background subtractor needed anymore - we use static edge detection

        # Server communication
        self.last_status = {}
        self.status_lock = Lock()
        self.send_interval = 2.0  # Send updates every 2 seconds
        self.last_send_time = 0

        # Load parking positions if file exists
        self.load_positions()

    def calculate_scale_factor(self, frame):
        """Calculate scale factor to fit frame on screen"""
        height, width = frame.shape[:2]
        self.original_size = (width, height)

        # Calculate scale factors for width and height
        width_scale = self.max_display_width / width
        height_scale = self.max_display_height / height

        # Use the smaller scale to ensure the entire frame fits
        self.scale_factor = min(width_scale, height_scale, 1.0)  # Don't upscale

        if self.scale_factor < 1.0:
            new_width = int(width * self.scale_factor)
            new_height = int(height * self.scale_factor)
            print(f"Scaling video: {width}x{height} -> {new_width}x{new_height} (scale: {self.scale_factor:.2f})")
        else:
            print(f"Video fits screen: {width}x{height} (no scaling needed)")

    def scale_frame(self, frame):
        """Scale frame for display"""
        if self.scale_factor < 1.0:
            new_width = int(frame.shape[1] * self.scale_factor)
            new_height = int(frame.shape[0] * self.scale_factor)
            return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return frame

    def scale_point(self, point):
        """Scale a point from display coordinates to original frame coordinates"""
        if self.scale_factor < 1.0:
            return (int(point[0] / self.scale_factor), int(point[1] / self.scale_factor))
        return point

    def scale_parking_spaces_for_display(self):
        """Scale parking space coordinates for display"""
        if self.scale_factor >= 1.0:
            return self.parking_spaces

        scaled_spaces = []
        for space in self.parking_spaces:
            scaled_space = space.copy()
            scaled_space['points'] = [
                (int(x * self.scale_factor), int(y * self.scale_factor))
                for x, y in space['points']
            ]
            scaled_spaces.append(scaled_space)
        return scaled_spaces

    def load_positions(self):
        """Load parking space positions from file"""
        if Path(self.parking_positions_file).exists():
            with open(self.parking_positions_file, 'rb') as f:
                self.parking_spaces = pickle.load(f)
            print(f"Loaded {len(self.parking_spaces)} parking spaces")

    def save_positions(self):
        """Save parking space positions to file"""
        with open(self.parking_positions_file, 'wb') as f:
            pickle.dump(self.parking_spaces, f)
        print(f"Saved {len(self.parking_spaces)} parking spaces")

    def send_status_to_server(self, occupied_spaces, free_spaces, force=False):
        """
        Send parking status to server
        Only sends if status changed or force=True
        """
        current_time = time.time()

        # Throttle updates unless forced
        if not force and (current_time - self.last_send_time) < self.send_interval:
            return

        # Build status dictionary
        current_status = {}
        for i in free_spaces:
            current_status[i] = True  # is_free = True
        for i in occupied_spaces:
            current_status[i] = False  # is_free = False

        # Check if status changed
        with self.status_lock:
            if current_status == self.last_status and not force:
                return

            self.last_status = current_status.copy()

        # Prepare data
        spaces_data = [
            {'space_number': space_num, 'is_free': is_free}
            for space_num, is_free in current_status.items()
        ]

        data = {'spaces': spaces_data}

        # Send to server in background thread
        def send_request():
            try:
                response = requests.post(
                    f"{self.server_url}/api/parking/update",
                    json=data,
                    timeout=5
                )

                if response.status_code == 200:
                    print(f"✓ Status sent: {len(free_spaces)} free, {len(occupied_spaces)} occupied")
                else:
                    print(f"✗ Server error: {response.status_code}")

            except requests.exceptions.RequestException as e:
                print(f"✗ Connection error: {e}")

        # Run in background thread to not block detection
        thread = Thread(target=send_request, daemon=True)
        thread.start()

        self.last_send_time = current_time

    def setup_parking_spaces(self, frame):
        """
        Interactive setup to define parking spaces with adjustable angles
        Click 4 points to define a quadrilateral for each parking space
        """
        # Calculate scale factor for this frame
        self.calculate_scale_factor(frame)

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # Convert display coordinates to original frame coordinates
                original_point = self.scale_point((x, y))

                # Check if clicking on existing space (to remove)
                for i, space in enumerate(self.parking_spaces):
                    if cv2.pointPolygonTest(np.array(space['points'], np.int32), original_point, False) >= 0:
                        self.parking_spaces.pop(i)
                        print(f"Removed parking space {i}")
                        return

                # Add point for new parking space (in original coordinates)
                self.points.append(original_point)
                print(f"Point {len(self.points)}: {original_point}")

                # If 4 points are selected, create parking space
                if len(self.points) == 4:
                    space = {
                        'points': self.points.copy(),
                        'id': len(self.parking_spaces)
                    }
                    self.parking_spaces.append(space)
                    print(f"Created parking space {space['id']}")
                    self.points = []

            elif event == cv2.EVENT_MOUSEMOVE:
                self.temp_point = (x, y)  # Store in display coordinates

            elif event == cv2.EVENT_RBUTTONDOWN:
                # Right click to cancel current drawing
                if len(self.points) > 0:
                    print(f"Cancelled drawing (had {len(self.points)} points)")
                    self.points = []

        cv2.namedWindow('Setup Parking Spaces')
        cv2.setMouseCallback('Setup Parking Spaces', mouse_callback)

        print("\n=== Parking Space Setup Mode ===")
        print("LEFT CLICK to add points (4 points per space)")
        print("Points should be in order: top-left, top-right, bottom-right, bottom-left")
        print("RIGHT CLICK to cancel current drawing")
        print("CLICK on existing space to REMOVE it")
        print("Press 's' to SAVE and 'q' to QUIT setup")
        print("================================\n")

        while True:
            # Scale frame for display
            img = self.scale_frame(frame.copy())

            # Draw existing parking spaces (scaled for display)
            scaled_spaces = self.scale_parking_spaces_for_display()
            for i, space in enumerate(scaled_spaces):
                pts = np.array(space['points'], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(img, [pts], True, (255, 0, 255), 2)

                # Fill with semi-transparent color
                overlay = img.copy()
                cv2.fillPoly(overlay, [pts], (255, 0, 255))
                cv2.addWeighted(overlay, 0.2, img, 0.8, 0, img)

                # Draw space number in center
                M = cv2.moments(pts)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    cv2.putText(img, str(i), (cx - 10, cy + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # Draw current points being placed (scale for display)
            if len(self.points) > 0:
                scaled_points = [
                    (int(x * self.scale_factor), int(y * self.scale_factor))
                    for x, y in self.points
                ]

                for i, pt in enumerate(scaled_points):
                    cv2.circle(img, pt, 5, (0, 255, 0), -1)
                    cv2.putText(img, str(i + 1), (pt[0] + 10, pt[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Draw lines between points
                if len(scaled_points) > 1:
                    for i in range(len(scaled_points) - 1):
                        cv2.line(img, scaled_points[i], scaled_points[i + 1], (0, 255, 0), 2)

                # Draw preview line to mouse cursor
                if self.temp_point:
                    cv2.line(img, scaled_points[-1], self.temp_point, (0, 255, 0), 1)

                    # If 3 points placed, show closing line
                    if len(scaled_points) == 3:
                        cv2.line(img, self.temp_point, scaled_points[0], (0, 255, 0), 1)

            # Display instructions
            instruction = f"Spaces: {len(self.parking_spaces)} | Points: {len(self.points)}/4"
            cv2.putText(img, instruction, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img, "s=save | q=quit | right-click=cancel", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow('Setup Parking Spaces', img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                self.save_positions()
                print(f"Saved {len(self.parking_spaces)} parking spaces")
            elif key == ord('q'):
                if len(self.points) > 0:
                    print("Warning: Incomplete parking space discarded")
                    self.points = []
                break

        cv2.destroyWindow('Setup Parking Spaces')

    def preprocess_frame(self, frame):
        """
        Preprocess frame with edge detection
        This method doesn't adapt to stationary objects
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Increase contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

        # Use Canny edge detection (more reliable than adaptive threshold)
        edges = cv2.Canny(blurred, 50, 150)

        # Dilate to connect nearby edges
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)

        return dilated

    def get_color_variance(self, frame, space):
        """
        Calculate color variance in parking space
        Empty spaces have lower variance (uniform pavement)
        Occupied spaces have higher variance (car details, shadows, etc.)
        """
        pts = np.array(space['points'], np.int32)
        x, y, w, h = cv2.boundingRect(pts)

        frame_h, frame_w = frame.shape[:2]
        x = max(0, x)
        y = max(0, y)
        x2 = min(frame_w, x + w)
        y2 = min(frame_h, y + h)
        w = x2 - x
        h = y2 - y

        if w <= 0 or h <= 0:
            return 0

        # Create mask for the polygon
        mask = np.zeros((h, w), dtype=np.uint8)
        shifted_pts = pts - [x, y]
        cv2.fillPoly(mask, [shifted_pts], 255)

        # Extract ROI
        roi = frame[y:y + h, x:x + w]

        if roi.shape[0] != mask.shape[0] or roi.shape[1] != mask.shape[1]:
            return 0

        # Calculate standard deviation of pixel intensities
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        masked_pixels = gray_roi[mask > 0]

        if len(masked_pixels) == 0:
            return 0

        return np.std(masked_pixels)

    def check_parking_space(self, edge_frame, color_frame, space):
        """
        Check if a parking space is occupied using multiple methods:
        1. Edge density (primary - doesn't adapt to stationary objects)
        2. Color variance (secondary - distinguishes cars from pavement)
        """
        pts = np.array(space['points'], np.int32)
        x, y, w, h = cv2.boundingRect(pts)

        frame_h, frame_w = edge_frame.shape[:2]
        x = max(0, x)
        y = max(0, y)
        x2 = min(frame_w, x + w)
        y2 = min(frame_h, y + h)
        w = x2 - x
        h = y2 - y

        if w <= 0 or h <= 0:
            return False

        # Create mask for the polygon
        mask = np.zeros((h, w), dtype=np.uint8)
        shifted_pts = pts - [x, y]
        cv2.fillPoly(mask, [shifted_pts], 255)

        # Extract edge ROI
        roi = edge_frame[y:y + h, x:x + w]

        if roi.shape[0] != mask.shape[0] or roi.shape[1] != mask.shape[1]:
            return False

        # Apply mask to edges
        masked_roi = cv2.bitwise_and(roi, roi, mask=mask)
        edge_pixel_count = cv2.countNonZero(masked_roi)
        mask_area = cv2.countNonZero(mask)

        if mask_area == 0:
            return False

        # Calculate edge density
        edge_ratio = edge_pixel_count / mask_area

        # Get color variance
        color_variance = self.get_color_variance(color_frame, space)

        # Decision logic:
        # High edge density OR high color variance = occupied
        # Use lower threshold for edge ratio since we're not using background subtraction
        is_occupied = edge_ratio > 0.28 or color_variance > 38

        return is_occupied

    def detect_occupancy(self, frame):
        """
        Main detection method - returns list of occupied spaces
        Uses edge detection (doesn't adapt) + color variance
        """
        # Get edge-based detection (primary method)
        edge_frame = self.preprocess_frame(frame)

        occupied_spaces = []
        free_spaces = []

        for i, space in enumerate(self.parking_spaces):
            is_occupied = self.check_parking_space(edge_frame, frame, space)
            if is_occupied:
                occupied_spaces.append(i)
            else:
                free_spaces.append(i)

        return occupied_spaces, free_spaces, edge_frame

    def draw_spaces(self, frame, occupied_spaces, free_spaces):
        """Draw parking spaces on frame with color coding"""
        img = frame.copy()

        # Draw free spaces in green
        for i in free_spaces:
            space = self.parking_spaces[i]
            pts = np.array(space['points'], np.int32)
            pts = pts.reshape((-1, 1, 2))

            overlay = img.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
            cv2.polylines(img, [pts], True, (0, 255, 0), 2)

            M = cv2.moments(pts)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.putText(img, str(i), (cx - 10, cy + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw occupied spaces in red
        for i in occupied_spaces:
            space = self.parking_spaces[i]
            pts = np.array(space['points'], np.int32)
            pts = pts.reshape((-1, 1, 2))

            overlay = img.copy()
            cv2.fillPoly(overlay, [pts], (0, 0, 255))
            cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
            cv2.polylines(img, [pts], True, (0, 0, 255), 2)

            M = cv2.moments(pts)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.putText(img, str(i), (cx - 10, cy + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Display statistics
        total = len(self.parking_spaces)
        free = len(free_spaces)
        occupied = len(occupied_spaces)

        cv2.putText(img, f"Total: {total}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f"Free: {free}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(img, f"Occupied: {occupied}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return img

    def init_picamera(self):
        """Initialize Picamera2"""
        try:
            self.picam2 = Picamera2()
            camera_props = self.picam2.camera_properties
            print(f"Camera: {camera_props}")

            config = self.picam2.create_preview_configuration(
                main={"format": 'BGR888', "size": (2304, 1296)},
                controls={"FrameRate": 30}
            )

            self.picam2.configure(config)
            self.picam2.start()
            time.sleep(2)

            print(f"Picamera2 initialized successfully")
            print(f"Resolution: {config['main']['size']}")
            return True
        except Exception as e:
            print(f"Error initializing Picamera2: {e}")
            return False

    def capture_frame_picamera(self):
        """Capture frame from Picamera2"""
        if self.picam2 is None:
            return None

        frame = self.picam2.capture_array()
        frame = frame[:, :, ::-1]  # BGR -> RGB
        return frame

    def run(self, setup_mode=False):
        """Run the parking space detection"""
        cap = None

        # Initialize camera
        if self.use_picamera:
            if not self.init_picamera():
                print("Failed to initialize Picamera2")
                return
        else:
            cap = cv2.VideoCapture(self.video_source)
            if not cap.isOpened():
                cap = cv2.VideoCapture(self.video_source, cv2.CAP_V4L2)

            if not cap.isOpened():
                print("Error: Cannot open video source")
                return

        # Read first frame for setup
        if self.use_picamera:
            frame = self.capture_frame_picamera()
        else:
            ret, frame = cap.read()
            if not ret:
                print("Error: Cannot read frame")
                return

        # Calculate scale factor based on first frame
        self.calculate_scale_factor(frame)

        # Setup parking spaces if needed
        if setup_mode or len(self.parking_spaces) == 0:
            self.setup_parking_spaces(frame)

        print("Press 'q' to quit, 's' to enter setup mode")
        print(f"Sending updates to: {self.server_url}")

        frame_count = 0

        while True:
            # Capture frame
            if self.use_picamera:
                frame = self.capture_frame_picamera()
                if frame is None:
                    print("Error: Cannot capture frame")
                    break
            else:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

            # Detect occupancy (on original frame)
            occupied, free, processed = self.detect_occupancy(frame)

            # Send status to server (first frame or periodic updates)
            force_send = (frame_count == 0)  # Force send on first frame
            self.send_status_to_server(occupied, free, force=force_send)

            # Draw results (on original frame)
            result = self.draw_spaces(frame, occupied, free)

            # Scale frames for display
            result_display = self.scale_frame(result)
            processed_display = self.scale_frame(processed)

            # Display frames
            cv2.imshow('Parking Space Detection', result_display)
            cv2.imshow('Processed Frame', processed_display)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.setup_parking_spaces(frame)
                # Send updated configuration immediately
                if len(self.parking_spaces) > 0:
                    occupied, free, _ = self.detect_occupancy(frame)
                    self.send_status_to_server(occupied, free, force=True)

            frame_count += 1

        # Cleanup
        if self.use_picamera and self.picam2:
            self.picam2.stop()
        elif cap:
            cap.release()
        cv2.destroyAllWindows()


# Example usage
if __name__ == "__main__":
    cv2.setNumThreads(16)

    # Configure your server URL
    SERVER_URL = 'http://192.168.1.127:5000'  # Change to your server IP

    # For Raspberry Pi Camera
    # detector = ParkingSpaceDetector(0, use_picamera=True, server_url=SERVER_URL)

    # For USB webcam
    # detector = ParkingSpaceDetector(0, use_picamera=False, server_url=SERVER_URL)

    # For video file
    detector = ParkingSpaceDetector('timeline.mp4', use_picamera=False, server_url=SERVER_URL)

    # Run with setup mode first time
    detector.run(setup_mode=True)
