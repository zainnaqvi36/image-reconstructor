import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model
model = load_model('cnn_inpainting_model.h5')

# Function to preprocess the image
def preprocess_image(image):
    image = np.array(image)
    image = cv2.resize(image, (128, 128))  # Resizing to match model input size
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert from RGB to BGR for model input
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Normalize to [0, 1]
    return image

# Function to add random occlusion to an image
def add_occlusion(image, occlusion_size=(32, 32)):
    h, w, _ = image.shape
    x = np.random.randint(0, w - occlusion_size[1])
    y = np.random.randint(0, h - occlusion_size[0])
    occluded_image = image.copy()
    occluded_image[y:y + occlusion_size[0], x:x + occlusion_size[1]] = 0  # Add occlusion
    return occluded_image

# Streamlit app UI
st.title('CNN Inpainting with Streamlit')
st.write("Upload an image to apply occlusion and generate the inpainted image using the trained model.")

# Upload an image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load and preprocess the image
    img = Image.open(uploaded_file)
    img = np.array(img)

    # Apply occlusion
    occluded_img = add_occlusion(img)

    # Preprocess the images for prediction
    original_img = preprocess_image(img)
    occluded_img = preprocess_image(occluded_img)

    # Predict using the trained CNN model
    reconstructed_img = model.predict(occluded_img)
    reconstructed_img = (reconstructed_img[0] * 255).astype(np.uint8)

    original_img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert original image to RGB
    occluded_img_rgb = cv2.cvtColor((occluded_img[0] * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)  # Convert occluded image to RGB
    reconstructed_img_rgb = cv2.cvtColor(reconstructed_img, cv2.COLOR_BGR2RGB)  # Convert generated image to RGB

    # Set the small dimensions for imagesco
    small_dim = (150, 150)

    # Display images in a 3x3 grid layout
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(original_img_rgb, caption="Original Image", width=small_dim[0])

    with col2:
        st.image(occluded_img_rgb, caption="Occluded Image", width=small_dim[0])

    with col3:
        st.image(reconstructed_img_rgb, caption="Generated Image", width=small_dim[0])
