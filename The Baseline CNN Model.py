# Importing the required Libraries
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Define project root and data file paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_CSV = os.path.join(PROJECT_ROOT, "data", "processed_mat", "train_metadata.csv")
VAL_CSV   = os.path.join(PROJECT_ROOT, "data", "processed_mat", "val_metadata.csv")

# Training hyperparameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4

# Load train/validation metadata from CSVs
train_df = pd.read_csv(TRAIN_CSV)
val_df   = pd.read_csv(VAL_CSV)

# Training Data Generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
# Validation data generator
val_datagen = ImageDataGenerator(rescale=1./255)

# Build training iterator
train_gen = train_datagen.flow_from_dataframe(
    train_df,
    x_col="image_path",
    y_col="tumor_type",
    target_size=IMG_SIZE,
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=True
)

# Build validation iterator
val_gen = val_datagen.flow_from_dataframe(
    val_df,
    x_col="image_path",
    y_col="tumor_type",
    target_size=IMG_SIZE,
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Build Model to be used 
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224,224,3))
base_model.trainable = False  

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.3)(x)
output = Dense(3, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(LR),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Train the Model
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

# Save the trained model
OUT_PATH = os.path.join(PROJECT_ROOT, "models", "resnet50_classifier.h5")
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
model.save(OUT_PATH)

print("Model saved to:", OUT_PATH)
