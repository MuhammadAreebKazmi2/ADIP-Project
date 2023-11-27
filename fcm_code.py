import cv2
import numpy as np
import matplotlib.pyplot as plt

def initialize_membership_matrix(n_samples, n_clusters):
    membership_matrix = np.random.rand(n_samples, n_clusters)
    membership_matrix = np.divide(membership_matrix, np.sum(membership_matrix, axis=1, keepdims=True))
    return membership_matrix

def update_cluster_centers(X, membership_matrix, fuzziness):
    numerator = np.dot(membership_matrix.T ** fuzziness, X)
    denominator = np.sum(membership_matrix.T ** fuzziness, axis=1, keepdims=True)
    new_centers = numerator / denominator
    return new_centers

def update_membership_matrix(X, cluster_centers, fuzziness):
    distances = np.linalg.norm(X[:, np.newaxis] - cluster_centers, axis=2)
    distances = np.fmax(distances, np.finfo(np.float64).eps)  # Avoid division by zero
    membership_matrix = 1.0 / distances ** (2 / (fuzziness - 1))
    membership_matrix = np.divide(membership_matrix, np.sum(membership_matrix, axis=1, keepdims=True))
    return membership_matrix

def fuzzy_c_means(X, n_clusters, fuzziness, max_iter=100, tol=1e-4, verbose=False):
    n_samples, n_features = X.shape

    # Initialize cluster centers randomly
    cluster_centers = X[np.random.choice(n_samples, n_clusters, replace=False)]

    # Initialize membership matrix randomly
    membership_matrix = initialize_membership_matrix(n_samples, n_clusters)

    for iteration in range(max_iter):
        # Update cluster centers
        new_cluster_centers = update_cluster_centers(X, membership_matrix, fuzziness)

        # Update membership matrix
        new_membership_matrix = update_membership_matrix(X, new_cluster_centers, fuzziness)

        # Calculate the change in cluster centers and membership matrix
        center_change = np.linalg.norm(new_cluster_centers - cluster_centers)
        membership_change = np.linalg.norm(new_membership_matrix - membership_matrix)

        # Update cluster centers and membership matrix
        cluster_centers = new_cluster_centers
        membership_matrix = new_membership_matrix

        if verbose:
            print(f"Iteration {iteration + 1}: Center Change = {center_change}, Membership Change = {membership_change}")

        # Check for convergence
        if center_change < tol and membership_change < tol:
            break

    return cluster_centers, membership_matrix

def apply_fuzzy_c_means_to_image(image_path, n_clusters, fuzziness):
    # Load the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    img_shape = img_lab.shape

    # Extract the L channel
    L = img_lab[:, :, 0]

    # Rescale pixel values to the range [0, 1]
    pixels = L / 255.0

    # Reshape the L channel to a 1D array
    pixels = pixels.reshape(-1, 1)

    # Run Fuzzy C-Means clustering
    cluster_centers, membership_matrix = fuzzy_c_means(pixels, n_clusters, fuzziness, verbose=True)

    # Assign each pixel to the cluster with the highest membership value
    labels = np.argmax(membership_matrix, axis=1)

    # Replace pixel values with cluster center values
    segmented_img = cluster_centers[labels].reshape(img_shape[0], img_shape[1])

    # Display the original and segmented images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(segmented_img, cmap='gray')
    plt.title('Segmented Image (Fuzzy C-Means)')
    plt.axis('off')

    plt.show()

# Example usage:
if __name__ == "__main__":
    np.random.seed(42)

    # Image path
    image_path = 'PIA11420~orig.jpg'

    # Number of clusters
    n_clusters = 3

    # Fuzziness parameter (greater values make the clusters fuzzier)
    fuzziness = 2.0

    apply_fuzzy_c_means_to_image(image_path, n_clusters, fuzziness)
