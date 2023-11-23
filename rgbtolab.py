# chat gpt
# Write a python code to convert an RGB image to L*a*b* image and display the three channels separtely using opencv and numpy
# End of Prompt

import cv2
import numpy as np

# Load the RGB image
rgb_image = cv2.imread('Tall_Grass.jpg')

# Convert the image from BGR to RGB
rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

# Convert RGB image to L*a*b*
lab_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)

# Extract L*, a*, and b* channels
l_channel, a_channel, b_channel = cv2.split(lab_image)

# Display the original RGB image
cv2.imshow('RGB Image', rgb_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Display the L* channel
cv2.imshow('L* Channel', l_channel)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Display the a* channel
cv2.imshow('a* Channel', a_channel)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Display the b* channel
cv2.imshow('b* Channel', b_channel)
cv2.waitKey(0)
cv2.destroyAllWindows()
