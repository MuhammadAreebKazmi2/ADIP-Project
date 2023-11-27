import cv2
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz

def apply_fuzzy_c_means_to_image(image_path):
    # Load the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_shape = img.shape

    # Rescale pixel values to the range [0, 1]
    pixels = img / 255.0

    # Reshape the image to a 2D array of pixels
    pixels = np.reshape(pixels, (img_shape[0] * img_shape[1], img_shape[2]))

    # FCM clustering using scikit-fuzzy
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
        pixels.T, 3, 2, error=0.005, maxiter=1000, init=None
    )

    # Assign each pixel to the cluster with the highest membership value
    labels = np.argmax(u, axis=0)

    # Replace pixel values with cluster center values
    segmented_img = cntr[labels].reshape(img_shape)

    # Display the original and segmented images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(segmented_img)
    plt.title('Segmented Image (Fuzzy C-Means)')
    plt.axis('off')

    plt.show()

# Example usage:
if __name__ == "__main__":
    # Image path
    image_path = 'PIA11420~orig.jpg'

    apply_fuzzy_c_means_to_image(image_path)
