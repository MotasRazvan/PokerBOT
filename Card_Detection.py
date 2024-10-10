# Author: George Motas
# Image detection for screenshot
# to be an app in the future
# TO DO:
#       - print when object appears on the screen or display something
#       - minimalistic gui maybe with Tk


import cv2
import numpy as np
import pyautogui

# Take a screenshot of the entire screen
screenshot = pyautogui.screenshot()

# Convert the screenshot to a format compatible with OpenCV (BGR)
main_image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

# Load the template image
template = cv2.imread('ball.jpg')

# Convert both images to grayscale
main_gray = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# Define the scales to iterate over
scales = np.linspace(0.2, 5.0, 50)  # Adjust the range and number of scales as needed

for scale in scales:
    # Resize the template image according to the current scale
    resized_template = cv2.resize(template_gray, None, fx=scale, fy=scale)
    
    # Get the dimensions of the resized template
    w, h = resized_template.shape[::-1]
    
    # Perform template matching
    res = cv2.matchTemplate(main_gray, resized_template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.95  # Adjust this threshold as needed
    loc = np.where(res >= threshold)
    
    # Draw rectangles around the matched areas
    for pt in zip(*loc[::-1]):
        cv2.rectangle(main_image, pt, (pt[0] + int(w/scale), pt[1] + int(h/scale)), (0, 255, 0), 2)

# Display the result
cv2.imshow('Result', main_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
