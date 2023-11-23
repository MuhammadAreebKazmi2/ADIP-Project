import cv2
import converter


# # Import picture & create HSI copy using algorithm
# img = cv2.imread('Tall_Grass.jpg', 1)
# hsi = converter.RGB_TO_HSI(img)

# # Display HSV Image
# cv2.imshow('HSI Image', hsi)

# # The three value channels
# cv2.imshow('H Channel', hsi[:, :, 0])
# cv2.imshow('S Channel', hsi[:, :, 1])
# cv2.imshow('I Channel', hsi[:, :, 2])

# # Wait for a key press and then terminate the program
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# # Import picture & create HSV copy using OpenCV
# bgr_img = cv2.imread('Tall_Grass.jpg')

# # Convert the BGR image to HSV Image
# hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
# cv2.imwrite('hsv_image.jpg', hsv_img)

# # Display the HSV image
# cv2.imshow('HSV image', hsv_img)
# cv2.imshow('H Channel', hsv_img[:, :, 0])
# cv2.imshow('S Channel', hsv_img[:, :, 1])
# cv2.imshow('V Channel', hsv_img[:, :, 2])
# cv2.waitKey(0)
# cv2.destroyAllWindows()

