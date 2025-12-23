# Import libraries for data handling, visualization, deep learning, and evaluation
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout,
    GlobalMaxPooling2D, Reshape, Activation,
    Add, Multiply, Lambda, Concatenate, Conv2D
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize

# Setting random seeds to improve reproducibility of model training and evaluation
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Define project paths for data, models, figures, and Grad-CAM outputs, and create required directories
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_CSV = os.path.join(PROJECT_ROOT, "data", "processed_mat", "train_metadata.csv")
VAL_CSV   = os.path.join(PROJECT_ROOT, "data", "processed_mat", "val_metadata.csv")

MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
FIGURES_DIR  = os.path.join(PROJECT_ROOT, "results", "figures")
METRICS_DIR  = os.path.join(PROJECT_ROOT, "results", "metrics")
GRADCAM_DIR  = os.path.join(FIGURES_DIR, "gradcam")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(GRADCAM_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 16


# Load training and validation metadata from CSV files into pandas DataFrames
train_df = pd.read_csv(TRAIN_CSV)
val_df   = pd.read_csv(VAL_CSV)

# Configure image data generators and build training/validation iterators with on-the-fly preprocessing and augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

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

# Derive the ordered class names and number of classes from the generator mapping
class_indices = train_gen.class_indices  # dict: {'glioma': 0, ...}
idx_to_class = {v: k for k, v in class_indices.items()}
class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
num_classes = len(class_names)
print("Class mapping:", class_indices)

# Define the CBAM attention module combining channel and spatial attention on convolutional feature maps
def cbam_block(input_tensor, reduction_ratio=8, name_prefix="cbam"):
    """Convolutional Block Attention Module (CBAM) applied to a feature map."""
    channel = input_tensor.shape[-1]

    # Channel attention MLP
    shared_dense_one = Dense(
        channel // reduction_ratio,
        activation="relu",
        kernel_initializer="he_normal",
        use_bias=True
    )
    shared_dense_two = Dense(
        channel,
        kernel_initializer="he_normal",
        use_bias=True
    )

    # Global average and max pooling
    avg_pool = GlobalAveragePooling2D()(input_tensor)
    max_pool = GlobalMaxPooling2D()(input_tensor)

    avg_pool = Reshape((1, 1, channel))(avg_pool)
    max_pool = Reshape((1, 1, channel))(max_pool)

    mlp_avg = shared_dense_two(shared_dense_one(avg_pool))
    mlp_max = shared_dense_two(shared_dense_one(max_pool))

    # Names for clarity
    channel_sigmoid_name = f"{name_prefix}_channel_sigmoid"
    channel_mul_name = f"{name_prefix}_channel_mul"
    spatial_conv_name = f"{name_prefix}_spatial_conv"
    spatial_mul_name = f"{name_prefix}_spatial_mul"

    # Channel attention
    channel_attention = Activation(
        "sigmoid",
        name=channel_sigmoid_name
    )(Add()([mlp_avg, mlp_max]))

    x = Multiply(name=channel_mul_name)([input_tensor, channel_attention])

    # Spatial attention: avg + max over channels
    avg_pool_spatial = Lambda(
        lambda z: tf.reduce_mean(z, axis=3, keepdims=True)
    )(x)
    max_pool_spatial = Lambda(
        lambda z: tf.reduce_max(z, axis=3, keepdims=True)
    )(x)
    concat = Concatenate(axis=3)([avg_pool_spatial, max_pool_spatial])

    spatial_attention = Conv2D(
        filters=1,
        kernel_size=7,
        padding="same",
        activation="sigmoid",
        kernel_initializer="he_normal",
        use_bias=False,
        name=spatial_conv_name
    )(concat)

    x = Multiply(name=spatial_mul_name)([x, spatial_attention])
    return x

# Initialize the ResNet-50 backbone for CBAM and freeze its weights (stage 1)
base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)

base_model.trainable = False  

# Build and compile the ResNet-50 + CBAM classifier on top of the frozen backbone
x = base_model.output
x = cbam_block(x, name_prefix="cbam_last")

x = GlobalAveragePooling2D(name="gap")(x)
x = Dropout(0.3, name="dropout")(x)
output = Dense(num_classes, activation="softmax", name="predictions")(x)

model = Model(inputs=base_model.input, outputs=output, name="resnet50_cbam")

model.compile(
    optimizer=Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Configure callbacks to save the best weights and adapt training based on validation loss
best_weights_path = os.path.join(MODELS_DIR, "resnet50_attention_best.weights.h5")

checkpoint_cb = ModelCheckpoint(
    best_weights_path,
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=True,   
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

# Stage 1: Train only the classification head with the ResNet-50 + CBAM backbone frozen
print("===== Stage 1: Training classifier head only (frozen ResNet + CBAM) =====")
history1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=5,
    callbacks=callbacks_list
)

# Stage 2: Unfreeze and fine-tune the top ResNet-50 layers plus CBAM with a lower learning rate
print("===== Stage 2: Fine-tuning top ResNet layers + CBAM =====")
# Unfreeze last 50 layers of the base model
for layer in base_model.layers[-50:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15,
    callbacks=callbacks_list
)

# Merge training histories from both stages and plot training/validation accuracy and loss curves
def merge_histories(h1, h2, key):
    return h1.history.get(key, []) + h2.history.get(key, [])

acc = merge_histories(history1, history2, "accuracy")
val_acc = merge_histories(history1, history2, "val_accuracy")
loss = merge_histories(history1, history2, "loss")
val_loss = merge_histories(history1, history2, "val_loss")

epochs_range = range(1, len(acc) + 1)

# Accuracy curve
plt.figure()
plt.plot(epochs_range, acc, label="Train Accuracy")
plt.plot(epochs_range, val_acc, label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy (ResNet-50 + CBAM)")
plt.legend()
acc_path = os.path.join(FIGURES_DIR, "resnet50_attention_accuracy_curve.png")
plt.savefig(acc_path, bbox_inches="tight")
plt.close()
print("Saved accuracy curve to:", acc_path)

# Loss curve
plt.figure()
plt.plot(epochs_range, loss, label="Train Loss")
plt.plot(epochs_range, val_loss, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss (ResNet-50 + CBAM)")
plt.legend()
loss_path = os.path.join(FIGURES_DIR, "resnet50_attention_loss_curve.png")
plt.savefig(loss_path, bbox_inches="tight")
plt.close()
print("Saved loss curve to:", loss_path)

# Load the best validation-loss weights into the current model before evaluation
print("===== Loading best weights into current model =====")
if os.path.exists(best_weights_path):
    model.load_weights(best_weights_path)
    print(f"Loaded best weights from: {best_weights_path}")
else:
    print("Best weights file not found, using current model weights.")

best_model = model  # alias for clarity

# Evaluate on the validation set and save the confusion matrix, heatmap, and classification report
print("===== Evaluating on validation set for confusion matrix & report =====")
val_steps = len(val_gen)
y_prob = best_model.predict(val_gen, steps=val_steps)
y_pred = np.argmax(y_prob, axis=1)
y_true = val_gen.classes  

# integer labels in same order as class_indices
cm = confusion_matrix(y_true, y_pred)
print("Confusion matrix:\n", cm)

# Save confusion matrix CSV
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
cm_csv_path = os.path.join(METRICS_DIR, "resnet50_attention_confusion_matrix.csv")
cm_df.to_csv(cm_csv_path)
print("Saved confusion matrix CSV to:", cm_csv_path)

# Plot confusion matrix heatmap
plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix (Validation) - ResNet50 + CBAM")
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)

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
cm_fig_path = os.path.join(FIGURES_DIR, "resnet50_attention_confusion_matrix_heatmap.png")
plt.tight_layout()
plt.savefig(cm_fig_path, bbox_inches="tight")
plt.close()
print("Saved confusion matrix heatmap to:", cm_fig_path)

# Classification report
report = classification_report(y_true, y_pred, target_names=class_names)
print("Classification report:\n", report)

report_path = os.path.join(METRICS_DIR, "resnet50_attention_classification_report.txt")
with open(report_path, "w") as f:
    f.write(report)
print("Saved classification report to:", report_path)

# Compute one-vs-rest ROC curves and AUC scores (micro, macro, and per-class) and save the ROC plot
print("===== Computing ROC curves and AUC scores =====")
y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))  

# shape: (N, num_classes)
fpr = {}
tpr = {}
roc_auc = {}

for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# micro-average
fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# macro-average
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(num_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= num_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label=f"micro-average ROC (AUC = {roc_auc['micro']:.2f})")
plt.plot(fpr["macro"], tpr["macro"],
         label=f"macro-average ROC (AUC = {roc_auc['macro']:.2f})")

for i, name in enumerate(class_names):
    plt.plot(fpr[i], tpr[i], lw=1,
             label=f"ROC {name} (AUC = {roc_auc[i]:.2f})")

plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - ResNet50 + CBAM")
plt.legend(loc="lower right", fontsize="small")
roc_path = os.path.join(FIGURES_DIR, "resnet50_attention_roc_curves.png")
plt.savefig(roc_path, bbox_inches="tight")
plt.close()
print("Saved ROC curves to:", roc_path)

# Define helper functions to compute and save Grad-CAM heatmaps and overlays
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    img_array: shape (1, H, W, 3) scaled to [0,1]
    """
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array, training=False)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    conv_outputs = conv_outputs * pooled_grads

    heatmap = tf.reduce_mean(conv_outputs, axis=-1)

    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def save_gradcam_for_batch(model, generator, last_conv_layer_name, out_dir, num_images=6):
    """
    Saves Grad-CAM overlays for a small subset of images from a generator.
    """
    batch_imgs, batch_labels = next(generator)
    batch_size = batch_imgs.shape[0]
    n = min(num_images, batch_size)

    for i in range(n):
        img = batch_imgs[i]          # (H, W, 3)
        label_idx = np.argmax(batch_labels[i])
        true_label = class_names[label_idx]

        img_input = np.expand_dims(img, axis=0)

        heatmap = make_gradcam_heatmap(
            img_input, model, last_conv_layer_name
        )

        # Resize heatmap to image size
        heatmap_resized = tf.image.resize(
            np.expand_dims(heatmap, axis=-1),
            (img.shape[0], img.shape[1])
        ).numpy().squeeze()

        # Overlay heatmap on original image
        plt.figure(figsize=(5, 2.5))

        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title(f"Original ({true_label})")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(img)
        plt.imshow(heatmap_resized, cmap="jet", alpha=0.4)
        plt.title("Grad-CAM overlay")
        plt.axis("off")

        out_path = os.path.join(out_dir, f"gradcam_{i}_{true_label}.png")
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print("Saved Grad-CAM image to:", out_path)

# Generate and save Grad-CAM visualizations for selected validation images using the best model
print("===== Generating Grad-CAM visualizations =====")
last_conv_layer_name = "conv5_block3_out"

val_cam_gen = val_datagen.flow_from_dataframe(
    val_df,
    x_col="image_path",
    y_col="tumor_type",
    target_size=IMG_SIZE,
    class_mode="categorical",
    batch_size=8,
    shuffle=False
)

save_gradcam_for_batch(best_model, val_cam_gen, last_conv_layer_name, GRADCAM_DIR, num_images=6)

# Save the final fine-tuned model weights and mark the pipeline as complete
final_weights_path = os.path.join(MODELS_DIR, "resnet50_attention_finetuned_best.weights.h5")
best_model.save_weights(final_weights_path)
print("Best fine-tuned attention model weights saved to:", final_weights_path)
print("===== Model 4 (ResNet50 + CBAM + ROC + GradCAM) pipeline complete. =====")
