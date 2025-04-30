# === 1. IMPORT LIBRARIES ===
import numpy as np
import cv2 
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import streamlit as st

# === 2. LOAD DATASET ===
IMAGE_DIR = "E:\\Aswin Docs\\Code\\images"
MASK_DIR = "E:\\Aswin Docs\\Code\\masks"

IMG_HEIGHT = 256
IMG_WIDTH = 256

# Function to load images
def load_data():
    images = []
    masks = []
    for img_name in os.listdir(IMAGE_DIR):
        img_path = os.path.join(IMAGE_DIR, img_name)
        mask_path = os.path.join(MASK_DIR, img_name)

        try:
            img = load_img(img_path, color_mode="grayscale", target_size=(IMG_HEIGHT, IMG_WIDTH))
            mask = load_img(mask_path, color_mode="grayscale", target_size=(IMG_HEIGHT, IMG_WIDTH))

            img = img_to_array(img) / 255.0
            mask = img_to_array(mask) / 255.0

            images.append(img)
            masks.append(mask)
        except Exception as e:
            print(f"Error loading {img_name}: {e}")
            continue

    return np.array(images, dtype=np.float32), np.array(masks, dtype=np.float32)

images, masks = load_data()

# === 3. DEFINE U-NET MODEL ===
def unet_model():
    inputs = layers.Input((IMG_HEIGHT, IMG_WIDTH, 1))

    c1 = layers.Conv2D(16, (3,3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(16, (3,3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2,2))(c1)

    c2 = layers.Conv2D(32, (3,3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(32, (3,3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2,2))(c2)

    c3 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2,2))(c3)

    c4 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(c4)

    u5 = layers.UpSampling2D((2,2))(c4)
    u5 = layers.concatenate([u5, c3])
    c5 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(u5)
    c5 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(c5)

    u6 = layers.UpSampling2D((2,2))(c5)
    u6 = layers.concatenate([u6, c2])
    c6 = layers.Conv2D(32, (3,3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(32, (3,3), activation='relu', padding='same')(c6)

    u7 = layers.UpSampling2D((2,2))(c6)
    u7 = layers.concatenate([u7, c1])
    c7 = layers.Conv2D(16, (3,3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(16, (3,3), activation='relu', padding='same')(c7)

    outputs = layers.Conv2D(1, (1,1), activation='sigmoid')(c7)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

model = unet_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# === 4. TRAIN-TEST SPLIT ===
X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=0.2, random_state=42)

# === 5. MODEL TRAINING ===
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=16)

# === 6. STREAMLIT APP ===
st.title("Brain Tumor Segmentation")
st.write("Upload an MRI Image to detect and segment tumor regions.")

uploaded_file = st.file_uploader("Choose an MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        img = load_img(uploaded_file, color_mode="grayscale", target_size=(IMG_HEIGHT, IMG_WIDTH))
        img_array = img_to_array(img) / 255.0
        img_input = np.expand_dims(img_array, axis=0).astype(np.float32)

        # Uncomment below if you have a trained model
        model = tf.keras.models.load_model('brain_tumor_segmentation.h5')
        pred_mask = model.predict(img_input)[0, :, :, 0]
        st.image(pred_mask, caption='Predicted Tumor Segmentation', use_column_width=True)

        # For visualization only (no prediction)
        st.image(img_array.squeeze(), caption='Uploaded MRI Image', use_column_width=True)
    except Exception as e:
        st.error(f"Error processing image: {e}")
