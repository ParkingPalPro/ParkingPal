import cv2
import numpy as np
import pickle
from pathlib import Path


class ParkingSpaceDetector:
    """
    Parking space occupancy detection using edge detection and background subtraction
    """

    def __init__(self, video_source, parking_positions_file='parking_positions.pkl'):
        self.video_source = video_source
        self.parking_positions_file = parking_positions_file
        self.parking_spaces = []  # List of parking space dictionaries with points and angle

        # Drawing state
        self.drawing = False
        self.points = []  # For polygon drawing
        self.temp_point = None  # Current mouse position

        # Background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True
        )

        # Load parking positions if file exists
        self.load_positions()

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

    def setup_parking_spaces(self, frame):
        """
        Interactive setup to define parking spaces with adjustable angles
        Click 4 points to define a quadrilateral for each parking space
        """

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # Check if clicking on existing space (to remove)
                for i, space in enumerate(self.parking_spaces):
                    if cv2.pointPolygonTest(np.array(space['points'], np.int32), (x, y), False) >= 0:
                        self.parking_spaces.pop(i)
                        print(f"Removed parking space {i}")
                        return

                # Add point for new parking space
                self.points.append((x, y))
                print(f"Point {len(self.points)}: ({x}, {y})")

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
                self.temp_point = (x, y)

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
            img = frame.copy()

            # Draw existing parking spaces
            for i, space in enumerate(self.parking_spaces):
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

            # Draw current points being placed
            if len(self.points) > 0:
                for i, pt in enumerate(self.points):
                    cv2.circle(img, pt, 5, (0, 255, 0), -1)
                    cv2.putText(img, str(i + 1), (pt[0] + 10, pt[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Draw lines between points
                if len(self.points) > 1:
                    for i in range(len(self.points) - 1):
                        cv2.line(img, self.points[i], self.points[i + 1], (0, 255, 0), 2)

                # Draw preview line to mouse cursor
                if self.temp_point:
                    cv2.line(img, self.points[-1], self.temp_point, (0, 255, 0), 1)

                    # If 3 points placed, show closing line
                    if len(self.points) == 3:
                        cv2.line(img, self.temp_point, self.points[0], (0, 255, 0), 1)

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
        Preprocess frame with edge detection and background subtraction
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 25, 10
        )

        # Apply median blur to remove small noise
        median = cv2.medianBlur(thresh, 5)

        # Dilate to connect edges
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(median, kernel, iterations=1)

        return dilated

    def apply_background_subtraction(self, frame):
        """
        Apply background subtraction to detect changes
        """
        fg_mask = self.bg_subtractor.apply(frame)

        # Remove shadows (value 127 in MOG2)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Morphological operations to clean up mask
        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        return fg_mask

    def check_parking_space(self, processed_frame, space):
        """
        Check if a parking space is occupied using polygon-based ROI
        Returns True if occupied, False if free
        """
        pts = np.array(space['points'], np.int32)

        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(pts)

        # Create mask for the polygon
        mask = np.zeros((h, w), dtype=np.uint8)

        # Shift points to local coordinates
        shifted_pts = pts - [x, y]
        cv2.fillPoly(mask, [shifted_pts], 255)

        # Extract ROI from processed frame
        roi = processed_frame[y:y + h, x:x + w]

        # Apply mask to ROI
        masked_roi = cv2.bitwise_and(roi, roi, mask=mask)

        # Count non-zero pixels only within the polygon
        pixel_count = cv2.countNonZero(masked_roi)

        # Calculate ratio of occupied pixels (only counting masked area)
        mask_area = cv2.countNonZero(mask)
        if mask_area == 0:
            return False

        occupied_ratio = pixel_count / mask_area

        # Threshold for occupancy (tune this value based on your needs)
        return occupied_ratio > 0.25

    def detect_occupancy(self, frame):
        """
        Main detection method - returns list of occupied spaces
        """
        # Preprocess frame
        processed = self.preprocess_frame(frame)

        # Apply background subtraction
        bg_mask = self.apply_background_subtraction(frame)

        # Combine both methods
        combined = cv2.bitwise_or(processed, bg_mask)

        occupied_spaces = []
        free_spaces = []

        for i, space in enumerate(self.parking_spaces):
            is_occupied = self.check_parking_space(combined, space)

            if is_occupied:
                occupied_spaces.append(i)
            else:
                free_spaces.append(i)

        return occupied_spaces, free_spaces, combined

    def draw_spaces(self, frame, occupied_spaces, free_spaces):
        """
        Draw parking spaces on frame with color coding
        Green = Free, Red = Occupied
        """
        img = frame.copy()

        # Draw free spaces in green
        for i in free_spaces:
            space = self.parking_spaces[i]
            pts = np.array(space['points'], np.int32)
            pts = pts.reshape((-1, 1, 2))

            # Draw filled polygon with transparency
            overlay = img.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)

            # Draw border
            cv2.polylines(img, [pts], True, (0, 255, 0), 2)

            # Draw space number in center
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

            # Draw filled polygon with transparency
            overlay = img.copy()
            cv2.fillPoly(overlay, [pts], (0, 0, 255))
            cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)

            # Draw border
            cv2.polylines(img, [pts], True, (0, 0, 255), 2)

            # Draw space number in center
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

    def run(self, setup_mode=False):
        """
        Run the parking space detection
        """
        cap = cv2.VideoCapture(self.video_source)

        if not cap.isOpened():
            print("Error: Cannot open video source")
            return

        # Read first frame for setup if needed
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame")
            return

        # Setup parking spaces if in setup mode or no positions exist
        if setup_mode or len(self.parking_spaces) == 0:
            self.setup_parking_spaces(frame)

        # Reset video to beginning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        print("Press 'q' to quit, 's' to enter setup mode")

        while True:
            ret, frame = cap.read()

            # Loop video if it ends
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                    history=500, varThreshold=16, detectShadows=True
                )
                continue

            # Detect occupancy
            occupied, free, processed = self.detect_occupancy(frame)

            # Draw results
            result = self.draw_spaces(frame, occupied, free)

            # Display frames
            cv2.imshow('Parking Space Detection', result)
            cv2.imshow('Processed Frame', processed)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.setup_parking_spaces(frame)

        cap.release()
        cv2.destroyAllWindows()


# Example usage
if __name__ == "__main__":
    cv2.setNumThreads(16)
    # For video file
    detector = ParkingSpaceDetector('easy1.mp4')

    # For webcam
    #detector = ParkingSpaceDetector(0)

    # For image (will process as static image)
    # detector = ParkingSpaceDetector('Solution/Datasets/parking_rois_gopro/images/GOPR0090.jpg')

    # Run with setup mode first time
    detector.run(setup_mode=True)

    # Subsequent runs
    # detector.run(setup_mode=False)