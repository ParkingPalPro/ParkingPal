import cv2
import numpy as np
import pytesseract

img = cv2.imread('images/image2.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

bfilber = cv2.bilateralFilter(gray, 11, 17, 17) # noize redaction
edged = cv2.Canny(bfilber, 30, 200)

keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = keypoints[0]
contours = sorted(contours, key=cv2.contourArea, reverse=True)

location = None
for contour in contours:
    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
    if len(approx) == 4:
        location = approx
        break

mask = np.zeros(gray.shape[:2], dtype="uint8")
new_image = cv2.drawContours(mask, [location], 0, 255, 2)
new_image = cv2.bitwise_and(img, img, mask=mask)


(x, y) = np.where(mask == 255)
x1 = np.min(x)
y1 = np.min(y)
x2 = np.max(x)
y2 = np.max(y)
cropped = img[x1:x2+1, y1:y2+1]

read = pytesseract.image_to_string(cropped)
print(read)
cv2.imshow("cropped", cropped)
cv2.waitKey(0)
