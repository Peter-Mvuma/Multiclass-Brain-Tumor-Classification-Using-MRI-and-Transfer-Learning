# Import libraries for data handling, visualization, deep learning and evaluation.
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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.metrics import confusion_matrix, classification_report

# Setting random seeds to improve reproducibility of model training
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Define project paths for data, models, figures, and metrics and create required directories
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_CSV = os.path.join(PROJECT_ROOT, "data", "processed_mat", "train_metadata.csv")
VAL_CSV   = os.path.join(PROJECT_ROOT, "data", "processed_mat", "val_metadata.csv")

MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
FIGURES_DIR  = os.path.join(PROJECT_ROOT, "results", "figures")
METRICS_DIR  = os.path.join(PROJECT_ROOT, "results", "metrics")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

# Specify core image size and batch configuration for the data pipeline
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# Load training and validation metadata from CSV files into pandas DataFrames
train_df = pd.read_csv(TRAIN_CSV)
val_df   = pd.read_csv(VAL_CSV)

# Configure image data generators for preprocessing and augmentation of training and validation data
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
    shuffle=True,
    seed=SEED
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

# Derive an ordered list of class names from the generator’s class_indices mapping
class_indices = train_gen.class_indices  
# dict: {'class_name': idx}
idx_to_class = {v: k for k, v in class_indices.items()}
class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
print("Class mapping:", class_indices)

# Construct and compile a ResNet-50–based transfer learning model for multi-class tumor classification.
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  

# Stage 1: Add and compile the classification head on top of the frozen ResNet-50 backbone.

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.3)(x)
output = Dense(len(class_names), activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Configure callbacks to save the best model and control training via validation loss.
best_model_path = os.path.join(MODELS_DIR, "resnet50_best.keras")

checkpoint_cb = ModelCheckpoint(
    best_model_path,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

earlystop_cb = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
    verbose=1
)

lr_cb = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=2,
    min_lr=1e-7,
    verbose=1
)

callbacks_list = [checkpoint_cb, earlystop_cb, lr_cb]

# Stage 1: Train classifier head only (frozen backbone)
print("===== Stage 1: Training classifier head only =====")
history1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=5,
    callbacks=callbacks_list
)

# Stage 2: Fine-tuning top layers (increase epochs to ~50 total)
# unfreezing the top 50 layers
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
    epochs=45,  
    callbacks=callbacks_list
)

# Combine stage-1 and stage-2 training histories to form continuous accuracy and loss curves
def merge_histories(h1, h2, key):
    return h1.history.get(key, []) + h2.history.get(key, [])

acc      = merge_histories(history1, history2, "accuracy")
val_acc  = merge_histories(history1, history2, "val_accuracy")
loss     = merge_histories(history1, history2, "loss")
val_loss = merge_histories(history1, history2, "val_loss")

epochs_range = range(1, len(acc) + 1)

# Smooth accuracy and loss curves using a moving average
WINDOW = 3  
acc_s      = pd.Series(acc).rolling(WINDOW, center=True, min_periods=1).mean()
val_acc_s  = pd.Series(val_acc).rolling(WINDOW, center=True, min_periods=1).mean()
loss_s     = pd.Series(loss).rolling(WINDOW, center=True, min_periods=1).mean()
val_loss_s = pd.Series(val_loss).rolling(WINDOW, center=True, min_periods=1).mean()

# Plot and save the training and validation accuracy curves 
plt.figure()
plt.plot(epochs_range, acc_s, label="Train Accuracy (smoothed)")
plt.plot(epochs_range, val_acc_s, label="Val Accuracy (smoothed)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy (ResNet-50)")
plt.legend()
acc_path = os.path.join(FIGURES_DIR, "resnet50_accuracy_curve.png")
plt.savefig(acc_path, bbox_inches="tight")
plt.close()
print("Saved accuracy curve to:", acc_path)

# Plot and save the training and validation loss curves.
plt.figure()
plt.plot(epochs_range, loss_s, label="Train Loss (smoothed)")
plt.plot(epochs_range, val_loss_s, label="Val Loss (smoothed)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss (ResNet-50)")
plt.legend()
loss_path = os.path.join(FIGURES_DIR, "resnet50_loss_curve.png")
plt.savefig(loss_path, bbox_inches="tight")
plt.close()
print("Saved loss curve to:", loss_path)

# ----------------------
# Load BEST model weights before evaluation
# ----------------------
if os.path.exists(best_model_path):
    print(f"Loading best model weights from: {best_model_path}")
    best_model = tf.keras.models.load_model(best_model_path)
else:
    print("Best model file not found, using current model weights.")
    best_model = model

# Generate validation predictions to compute the confusion matrix and classification report
print("===== Evaluating on validation set for confusion matrix =====")
val_steps = len(val_gen)
y_prob = best_model.predict(val_gen, steps=val_steps)
y_pred = np.argmax(y_prob, axis=1)
y_true = val_gen.classes  

# Compute and display the confusion matrix for validation predictions.
cm = confusion_matrix(y_true, y_pred)
print("Confusion matrix:\n", cm)

# Save the confusion matrix as a labeled CSV file.
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
cm_csv_path = os.path.join(METRICS_DIR, "resnet50_confusion_matrix.csv")
cm_df.to_csv(cm_csv_path)
print("Saved confusion matrix CSV to:", cm_csv_path)

# Plot confusion matrix heatmap
plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix (Validation)")
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)

# Annotate the confusion matrix heatmap with cell counts and save to disk.
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

# Generate and save the text classification report.
report = classification_report(y_true, y_pred, target_names=class_names)
print("Classification report:\n", report)

report_path = os.path.join(METRICS_DIR, "resnet50_classification_report.txt")
with open(report_path, "w") as f:
    f.write(report)
print("Saved classification report to:", report_path)

# Save the final fine-tuned model (best weights) in Keras format for later use.
final_model_path = os.path.join(MODELS_DIR, "resnet50_finetuned_best.keras")
best_model.save(final_model_path)
print("Best fine-tuned model saved to:", final_model_path)
