import cv2
import numpy as np

# Load the large image and the template (smaller image)
large_image = cv2.imread('full.jpg')
template = cv2.imread('ball.jpg')

# Convert images to grayscale
large_gray = cv2.cvtColor(large_image, cv2.COLOR_BGR2GRAY)
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# Get the dimensions of the template image
w, h = template_gray.shape[::-1]

# Perform template matching
result = cv2.matchTemplate(large_gray, template_gray, cv2.TM_CCOEFF_NORMED)

# Set a higher threshold to avoid false positives
threshold = 0.95  # Increase threshold to reduce false positives
loc = np.where(result >= threshold)

# If you want to find the single best match, use minMaxLoc
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# Print the single best match found
if max_val >= threshold:
    print(f"Best match found at location: {max_loc}, with confidence: {max_val}")
else:
    print("No match found with sufficient confidence.")
