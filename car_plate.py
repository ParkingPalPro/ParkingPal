import cv2
import numpy as np
import pytesseract
import re


def is_valid_plate_text(text):
    """
    Validate if text looks like a license plate
    Returns True if text contains at least 2 letters or numbers
    """
    if not text or len(text) < 3:
        return False

    # Remove special characters and spaces
    clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())

    # Check if we have at least 3 alphanumeric characters
    if len(clean_text) < 3:
        return False

    # Count letters and numbers
    letters = sum(c.isalpha() for c in clean_text)
    numbers = sum(c.isdigit() for c in clean_text)

    # Valid plate should have at least some letters AND numbers
    return letters >= 1 and numbers >= 1


def is_valid_plate_shape(contour, frame_shape):
    """
    Check if contour has valid license plate proportions
    """
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / float(h)

    # License plates are typically wider than tall (aspect ratio 2-5)
    if aspect_ratio < 1.5 or aspect_ratio > 6:
        return False

    # Check minimum size (at least 60 pixels wide)
    if w < 60 or h < 20:
        return False

    # Check maximum size (not the entire frame)
    frame_area = frame_shape[0] * frame_shape[1]
    contour_area = cv2.contourArea(contour)
    if contour_area > frame_area * 0.5:  # Not more than 50% of frame
        return False

    return True


def detect_plate(frame):
    """
    Detect and read license plate from a frame
    Returns the cropped plate image and detected text
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Enhanced preprocessing
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(bfilter, 30, 200)

    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = keypoints[0]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]

    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)

        # Check for rectangle and valid shape
        if len(approx) == 4 and is_valid_plate_shape(contour, gray.shape):
            location = approx
            break

    if location is None:
        return None, None, None

    # Create mask and extract region
    mask = np.zeros(gray.shape, dtype="uint8")
    cv2.drawContours(mask, [location], 0, 255, -1)

    (y, x) = np.where(mask == 255)
    if len(x) == 0 or len(y) == 0:
        return None, None, None

    x1, y1 = np.min(x), np.min(y)
    x2, y2 = np.max(x), np.max(y)

    cropped = frame[y1:y2 + 1, x1:x2 + 1]

    # Enhance cropped image for better OCR
    cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    cropped_gray = cv2.bilateralFilter(cropped_gray, 11, 17, 17)

    # Apply threshold
    _, cropped_thresh = cv2.threshold(cropped_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # OCR with better config
    text = pytesseract.image_to_string(cropped_thresh,
                                       config='--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    text = text.strip()

    # Validate text
    if not is_valid_plate_text(text):
        return None, None, None

    # Clean up text
    text = re.sub(r'[^A-Z0-9]', '', text.upper())

    return cropped, text, location


def process_video(source=0, skip_frames=2):
    """
    Process video from file or webcam
    source: 0 for webcam, or path to video file
    skip_frames: process every Nth frame (higher = faster but less responsive)
    """
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Error: Could not open video source")
        return

    print("Press 'q' to quit, 's' to save current detection")
    print("Filtering enabled: Only valid plate text will be shown")

    frame_count = 0
    last_valid_detection = None
    detection_confidence = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("End of video or cannot read frame")
            break

        frame_count += 1
        display_frame = frame.copy()

        # Process every Nth frame to reduce noise
        if frame_count % skip_frames == 0:
            cropped, text, location = detect_plate(frame)

            # Only update if we got a valid detection
            if text and location is not None:
                last_valid_detection = (cropped, text, location)
                detection_confidence = 10  # Reset confidence counter

        # Display last valid detection with fade out
        if last_valid_detection is not None and detection_confidence > 0:
            cropped, text, location = last_valid_detection

            # Draw rectangle
            cv2.drawContours(display_frame, [location], -1, (0, 255, 0), 3)

            # Display text on frame
            cv2.putText(display_frame, text, (location[0][0][0], location[0][0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Show cropped plate
            if cropped is not None and cropped.size > 0:
                cv2.imshow("Detected Plate", cropped)

            detection_confidence -= 1

        # Display FPS and detection status
        status = f"Detection: {last_valid_detection[1] if last_valid_detection else 'Searching...'}"
        cv2.putText(display_frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("License Plate Detection", display_frame)

        # Key controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and last_valid_detection is not None:
            cropped, text, _ = last_valid_detection
            cv2.imwrite("detected_plate.jpg", cropped)
            print(f"Saved plate image with text: {text}")


if __name__ == "__main__":
    # Choose your source:

    # For webcam (default camera):
    #process_video(0)

    # For video file:
    process_video('exit.mp4')

    # For different webcam (if you have multiple):
    # process_video(1)  # or 2, 3, etc.