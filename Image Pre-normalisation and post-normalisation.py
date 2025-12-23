#Importing the necessary libraries
import cv2
import matplotlib.pyplot as plt

# Random Image selection from mat_images FOLDER
img_path = r"C:/Users/peter/brain_tumor_project/data/raw/mat_images/1000_img.png"

# Load image
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError(f"Could not load: {img_path}")

# Normalize image (Min-Max scaling)
norm_img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

# Plot comparison
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title("Pre-Normalization")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(norm_img, cmap='gray')
plt.title("Post-Normalization")
plt.axis('off')

plt.tight_layout()
plt.savefig("normalization_comparison.png", dpi=300)
plt.show()
