# Import libraries for data handling, visualization, deep learning, and evaluation.
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report

# Define project paths and create output directories
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_CSV = os.path.join(PROJECT_ROOT, "data", "processed_mat", "train_metadata.csv")
VAL_CSV   = os.path.join(PROJECT_ROOT, "data", "processed_mat", "val_metadata.csv")

MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
FIGURES_DIR  = os.path.join(PROJECT_ROOT, "results", "figures")
METRICS_DIR  = os.path.join(PROJECT_ROOT, "results", "metrics")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# Load training and validation metadata from CSV files.
train_df = pd.read_csv(TRAIN_CSV)
val_df   = pd.read_csv(VAL_CSV)


# Configure image data generators and build training/validation iterators.
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_dataframe(
    train_df,
    x_col="image_path",
    y_col="tumor_type",
    target_size=IMG_SIZE,
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_gen = val_datagen.flow_from_dataframe(
    val_df,
    x_col="image_path",
    y_col="tumor_type",
    target_size=IMG_SIZE,
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Derive ordered class names from the generator's class_indices mapping
class_indices = train_gen.class_indices  
# dict: {'class_name': idx}
idx_to_class = {v: k for k, v in class_indices.items()}
class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

print("Class mapping:", class_indices)

# Build and compile the ResNet-50 model with frozen backbone (stage 1).
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224,224,3))
base_model.trainable = False  # Stage 1: Train head only

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.3)(x)
output = Dense(3, activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# Stage 1: Train only the classification head.
print("===== Stage 1: Training classifier head only =====")
history1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=5
)


# Stage 2: Unfreeze top layers and fine-tune with lower learning rate.
for layer in base_model.layers[-50:]:  
    layer.trainable = True

model.compile(
    optimizer=Adam(1e-5),    
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("===== Stage 2: Fine-tuning top layers =====")
history2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15
)


# Merge training histories from both stages for plotting
def merge_histories(h1, h2, key):
    return h1.history.get(key, []) + h2.history.get(key, [])

acc = merge_histories(history1, history2, "accuracy")
val_acc = merge_histories(history1, history2, "val_accuracy")
loss = merge_histories(history1, history2, "loss")
val_loss = merge_histories(history1, history2, "val_loss")

epochs_range = range(1, len(acc) + 1)


# Plot and save training/validation accuracy curve
plt.figure()
plt.plot(epochs_range, acc, label="Train Accuracy")
plt.plot(epochs_range, val_acc, label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy (ResNet-50)")
plt.legend()
acc_path = os.path.join(FIGURES_DIR, "resnet50_accuracy_curve.png")
plt.savefig(acc_path, bbox_inches="tight")
plt.close()
print("Saved accuracy curve to:", acc_path)

# Plot and save training/validation loss curve.

plt.figure()
plt.plot(epochs_range, loss, label="Train Loss")
plt.plot(epochs_range, val_loss, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss (ResNet-50)")
plt.legend()
loss_path = os.path.join(FIGURES_DIR, "resnet50_loss_curve.png")
plt.savefig(loss_path, bbox_inches="tight")
plt.close()
print("Saved loss curve to:", loss_path)


# Generate validation predictions and compute confusion matrix with heatmap and report.

print("===== Evaluating on validation set for confusion matrix =====")
val_steps = len(val_gen)
y_prob = model.predict(val_gen, steps=val_steps)
y_pred = np.argmax(y_prob, axis=1)
y_true = val_gen.classes  # integer labels in same order as class_indices

cm = confusion_matrix(y_true, y_pred)
print("Confusion matrix:\n", cm)

# Save confusion matrix as CSV.
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
cm_csv_path = os.path.join(METRICS_DIR, "resnet50_confusion_matrix.csv")
cm_df.to_csv(cm_csv_path)
print("Saved confusion matrix CSV to:", cm_csv_path)

# Plot confusion matrix heatmap with cell annotations
plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix (Validation)")
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)

# Add numbers on cells
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j, i, format(cm[i, j], "d"),
            ha="center",
            va="center",
            color="white" if cm[i, j] > thresh else "black"
        )

plt.ylabel("True label")
plt.xlabel("Predicted label")
cm_fig_path = os.path.join(FIGURES_DIR, "resnet50_confusion_matrix_heatmap.png")
plt.tight_layout()
plt.savefig(cm_fig_path, bbox_inches="tight")
plt.close()
print("Saved confusion matrix heatmap to:", cm_fig_path)

# Generate and save classification report
report = classification_report(y_true, y_pred, target_names=class_names)
print("Classification report:\n", report)

report_path = os.path.join(METRICS_DIR, "resnet50_classification_report.txt")
with open(report_path, "w") as f:
    f.write(report)
print("Saved classification report to:", report_path)

# Save the final fine-tuned model
OUT_PATH = os.path.join(MODELS_DIR, "resnet50_finetuned.h5")
model.save(OUT_PATH)
print("Fine-tuned model saved to:", OUT_PATH)
