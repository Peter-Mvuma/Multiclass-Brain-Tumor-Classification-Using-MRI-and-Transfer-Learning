
#Importing the necessary libries
import pandas as pd
import matplotlib.pyplot as plt
import cv2

df = pd.read_csv("data/raw/metadata/mat_metadata.csv")

# Select one sample per class
samples = df.groupby("tumor_type").first()

plt.figure(figsize=(12,4))
for i, (tumor, row) in enumerate(samples.iterrows(), 1):
    img = cv2.imread(row["image_path"])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(1, 3, i)
    plt.imshow(img); plt.title(tumor); plt.axis('off')

plt.tight_layout()
plt.show()
