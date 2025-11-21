import cv2
import numpy as np
import time
import requests
from threading import Thread, Lock
import io

OCCUPATION_THRESHOLD = 0.28

# Try to import Picamera2, fallback to regular camera if not available
try:
    from picamera2 import Picamera2

    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False
    print("Picamera2 not available, using standard camera")

def resize_frame_keep_ratio(frame, target_width=360):
    h, w = frame.shape[:2]
    scale = target_width / float(w)
    target_height = int(h * scale)
    resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
    return resized, scale



class HybridParkingDetector:
    """
    Parking detector with:
    - Web-based configuration management (pulls from API)
    - Hybrid detection (background subtraction + edge detection)
    - Snapshot uploads
    - Proper space ID mapping from API
    """

    def __init__(self, camera_id='CAM_PARKING_01', video_source=0,
                 use_picamera=False, server_url='http://localhost:5000',
                 snapshot_interval=10, config_check_interval=5):

        self.camera_id = camera_id
        self.video_source = video_source
        self.use_picamera = use_picamera and PICAMERA_AVAILABLE
        self.server_url = server_url
        self.picam2 = None

        # Parking spaces configuration (from API)
        # Format: [{'id': 4, 'points': [[x,y], ...]}, ...]
        self.parking_spaces = []
        self.config_lock = Lock()

        # Background subtraction
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=40,
            detectShadows=True
        )

        # Timing
        self.snapshot_interval = snapshot_interval
        self.config_check_interval = config_check_interval
        self.last_snapshot_time = 0
        self.last_config_check = 0

        # Status tracking - keyed by space ID (not index)
        self.last_status = {}
        self.status_lock = Lock()

        # Background threads
        self.running = False
        self.config_thread = None

        print(f"Initialized {self.camera_id}")
        print(f"Server: {self.server_url}")

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

    def send_snapshot(self, frame):
        """Send snapshot to server for admin interface"""
        current_time = time.time()

        if (current_time - self.last_snapshot_time) < self.snapshot_interval:
            return

        def upload():
            try:
                # Encode frame as JPEG
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

                # Send to server
                files = {'image': ('snapshot.jpg', io.BytesIO(buffer), 'image/jpeg')}
                response = requests.post(
                    f"{self.server_url}/api/camera/{self.camera_id}/snapshot",
                    files=files,
                    timeout=5
                )

                if response.status_code == 200:
                    print(f"✓ Snapshot sent to server")
                else:
                    print(f"✗ Snapshot upload failed: {response.status_code}")

            except Exception as e:
                print(f"✗ Snapshot upload error: {e}")

        Thread(target=upload, daemon=True).start()
        self.last_snapshot_time = current_time

    def fetch_configuration(self):
        """Fetch parking space configuration from server"""
        try:
            response = requests.get(
                f"{self.server_url}/api/camera/{self.camera_id}/config",
                timeout=5
            )

            if response.status_code == 200:
                data = response.json()
                spaces = data.get('spaces', [])

                with self.config_lock:
                    if spaces != self.parking_spaces:
                        self.parking_spaces = spaces
                        space_ids = [s['id'] for s in spaces]
                        print(f"✓ Configuration updated: {len(spaces)} parking spaces")
                        print(f"  Space IDs: {space_ids}")
                        return True
                return False
            else:
                print(f"✗ Config fetch failed: {response.status_code}")
                return False

        except Exception as e:
            print(f"✗ Config fetch error: {e}")
            return False

    def config_sync_loop(self):
        """Background thread to periodically check for configuration updates"""
        while self.running:
            current_time = time.time()

            if (current_time - self.last_config_check) >= self.config_check_interval:
                self.fetch_configuration()
                self.last_config_check = current_time

            time.sleep(1)

    def send_status_to_server(self, occupied_space_ids, free_space_ids):
        """
        Send parking status to server
        Uses space IDs (not indices)
        """
        current_status = {}
        for space_id in free_space_ids:
            current_status[space_id] = True
        for space_id in occupied_space_ids:
            current_status[space_id] = False

        # Check if status changed
        with self.status_lock:
            if current_status == self.last_status:
                return
            self.last_status = current_status.copy()

        # Prepare data
        spaces_data = [
            {'space_number': space_id, 'is_free': is_free}
            for space_id, is_free in current_status.items()
        ]

        data = {
            'camera_id': self.camera_id,
            'spaces': spaces_data
        }

        def send():
            try:
                response = requests.post(
                    f"{self.server_url}/api/parking/update",
                    json=data,
                    timeout=5
                )

                if response.status_code == 200:
                    print(f"✓ Status update: {len(free_space_ids)} free, {len(occupied_space_ids)} occupied")
                    print(f"  Free IDs: {free_space_ids}")
                    print(f"  Occupied IDs: {occupied_space_ids}")

            except Exception as e:
                print(f"✗ Status update error: {e}")

        Thread(target=send, daemon=True).start()

    def preprocess_frame_edges(self, frame):
        """Edge detection preprocessing"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Blur and edge detection
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Dilate edges
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)

        return dilated

    def preprocess_frame_bg_sub(self, frame):
        """Background subtraction preprocessing"""
        fg_mask = self.bg_subtractor.apply(frame)

        # Remove shadows (value 127 in MOG2)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Noise reduction
        kernel = np.ones((3, 3), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        return fg_mask

    def get_color_variance(self, frame, space):
        """Calculate color variance in parking space"""
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

        mask = np.zeros((h, w), dtype=np.uint8)
        shifted_pts = pts - [x, y]
        cv2.fillPoly(mask, [shifted_pts], 255)

        roi = frame[y:y + h, x:x + w]

        if roi.shape[0] != mask.shape[0] or roi.shape[1] != mask.shape[1]:
            return 0

        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        masked_pixels = gray_roi[mask > 0]

        if len(masked_pixels) == 0:
            return 0

        return np.std(masked_pixels)

    @staticmethod
    def check_parking_space(edge_frame, bg_mask, color_frame, space):
        """
        Check if parking space is occupied using hybrid approach:
        1. Combine edge detection and background subtraction with bitwise OR
        2. Analyze combined pixel density
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

        mask_area = cv2.countNonZero(mask)
        if mask_area == 0:
            return False

        # Extract ROIs
        bg_roi = bg_mask[y:y + h, x:x + w]
        edge_roi = edge_frame[y:y + h, x:x + w]

        if (bg_roi.shape[0] != mask.shape[0] or bg_roi.shape[1] != mask.shape[1] or
                edge_roi.shape[0] != mask.shape[0] or edge_roi.shape[1] != mask.shape[1]):
            return False

        # Combine edge detection and background subtraction using bitwise OR
        combined = cv2.bitwise_or(edge_roi, bg_roi)

        # Apply mask to combined frame
        masked_combined = cv2.bitwise_and(combined, combined, mask=mask)

        # Count pixels in combined detection
        pixel_count = cv2.countNonZero(masked_combined)
        pixel_ratio = pixel_count / mask_area

        # Decision: occupied if combined pixel ratio exceeds threshold
        is_occupied = pixel_ratio > OCCUPATION_THRESHOLD

        return is_occupied

    def detect_occupancy(self, frame):
        """
        Detect occupancy for all parking spaces
        Returns space IDs (not indices)
        """
        with self.config_lock:
            if len(self.parking_spaces) == 0:
                return [], [], None, None, None

            spaces = self.parking_spaces.copy()

        # Generate both detection frames
        edge_frame = self.preprocess_frame_edges(frame)
        bg_mask = self.preprocess_frame_bg_sub(frame)

        # Combine using bitwise OR
        combined_frame = cv2.bitwise_or(edge_frame, bg_mask)

        occupied_space_ids = []
        free_space_ids = []

        for space in spaces:
            space_id = space['id']
            is_occupied = self.check_parking_space(edge_frame, bg_mask, frame, space)

            if is_occupied:
                occupied_space_ids.append(space_id)
            else:
                free_space_ids.append(space_id)

        return occupied_space_ids, free_space_ids, edge_frame, bg_mask, combined_frame

    def draw_spaces(self, frame, occupied_space_ids, free_space_ids):
        """Draw parking spaces on frame using space IDs"""
        img = frame.copy()

        with self.config_lock:
            spaces = self.parking_spaces.copy()

        # Draw free spaces (green)
        for space in spaces:
            space_id = space['id']
            pts = np.array(space['points'], np.int32).reshape((-1, 1, 2))

            if space_id in free_space_ids:
                color = (0, 255, 0)  # Green
                overlay = img.copy()
                cv2.fillPoly(overlay, [pts], color)
                cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
                cv2.polylines(img, [pts], True, color, 2)

                # Label with space ID
                M = cv2.moments(pts)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    cv2.putText(img, str(space_id), (cx - 10, cy + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Draw occupied spaces (red)
        for space in spaces:
            space_id = space['id']
            pts = np.array(space['points'], np.int32).reshape((-1, 1, 2))

            if space_id in occupied_space_ids:
                color = (0, 0, 255)  # Red
                overlay = img.copy()
                cv2.fillPoly(overlay, [pts], color)
                cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
                cv2.polylines(img, [pts], True, color, 2)

                # Label with space ID
                M = cv2.moments(pts)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    cv2.putText(img, str(space_id), (cx - 10, cy + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Statistics
        total = len(spaces)
        free = len(free_space_ids)
        occupied = len(occupied_space_ids)

        cv2.putText(img, f"Camera: {self.camera_id}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f"Total: {total} | Free: {free} | Occupied: {occupied}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return img

    def run(self):
        """Main detection loop"""
        # Initialize camera
        cap = None
        if self.use_picamera:
            if not self.init_picamera():
                print("Failed to initialize Picamera")
                return
        else:
            cap = cv2.VideoCapture(self.video_source)
            if not cap.isOpened():
                print("Error: Cannot open camera")
                return

        # Start configuration sync thread
        self.running = True
        self.config_thread = Thread(target=self.config_sync_loop, daemon=True)
        self.config_thread.start()

        # Initial config fetch
        print("Fetching initial configuration...")
        self.fetch_configuration()

        print("\n" + "=" * 60)
        print(f"  HYBRID PARKING DETECTOR - {self.camera_id}")
        print("=" * 60)
        print(f"Server: {self.server_url}")
        print(f"Snapshot interval: {self.snapshot_interval}s")
        print(f"Config check interval: {self.config_check_interval}s")
        print(f"Detection: Edge Detection OR Background Subtraction (bitwise OR)")
        print("\nPress 'q' to quit, 'r' to force config refresh")
        print("=" * 60 + "\n")

        frame_count = 0

        try:
            while True:
                # Capture frame
                frame = self.capture_frame(cap)
                if frame is None:
                    if cap:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue


                # Send snapshot to server periodically
                self.send_snapshot(frame)

                # Detect occupancy (returns space IDs, not indices)
                occupied_ids, free_ids, edge_frame, bg_mask, combined_frame = self.detect_occupancy(frame)

                # Send status to server
                if len(self.parking_spaces) > 0:
                    self.send_status_to_server(occupied_ids, free_ids)

                # Draw results
                result = self.draw_spaces(frame, occupied_ids, free_ids)
                result = cv2.resize(result, (720, int(frame.shape[0] * (720 / frame.shape[1]))))

                # Display
                cv2.imshow(f'Parking Detector - {self.camera_id}', result)
                #if edge_frame is not None:
                #    cv2.imshow('Edge Detection', edge_frame)
                #if bg_mask is not None:
                #    cv2.imshow('Background Subtraction', bg_mask)
                if combined_frame is not None:
                    combined_frame = cv2.resize(combined_frame, (720, int(frame.shape[0] * (720 / frame.shape[1]))))
                    cv2.imshow('Combined (Edge OR BG)', combined_frame)

                # Handle keyboard input
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    print("Force refreshing configuration...")
                    self.fetch_configuration()

                frame_count += 1

        finally:
            self.running = False
            if self.config_thread:
                self.config_thread.join(timeout=2)

            if self.use_picamera and self.picam2:
                self.picam2.stop()
            elif cap:
                cap.release()

            cv2.destroyAllWindows()
            print("\nDetector stopped")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Hybrid web-managed parking detector')
    parser.add_argument('--camera-id', default='CAM_PARKING_01', help='Camera identifier')
    parser.add_argument('--server', default='http://localhost:5000', help='Server URL')
    parser.add_argument('--source', default=0, help='Video source (0 for webcam, or video file path)')
    parser.add_argument('--picamera', action='store_true', help='Use Raspberry Pi camera')
    parser.add_argument('--snapshot-interval', type=int, default=10, help='Snapshot upload interval (seconds)')
    parser.add_argument('--config-interval', type=int, default=5, help='Config check interval (seconds)')

    args = parser.parse_args()

    # Parse video source
    try:
        video_source = int(args.source)
    except ValueError:
        video_source = args.source

    # Create detector
    detector = HybridParkingDetector(
        camera_id=args.camera_id,
        video_source=video_source,
        use_picamera=args.picamera,
        server_url=args.server,
        snapshot_interval=args.snapshot_interval,
        config_check_interval=args.config_interval
    )

    # Set OpenCV thread count for performance
    cv2.setNumThreads(16)

    # Run
    detector.run()
