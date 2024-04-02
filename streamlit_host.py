import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobilenet_v2_preprocess_input

# Load the trained model
model_path = r"C:\Users\Rishav\Desktop\projects\dataset_pbl\cnnModel.hdf5"
model = tf.keras.models.load_model(model_path)

# Define class labels
class_labels = {0: 'No fire', 1: 'Fire'}

# Function to preprocess the image
def preprocess_image(img):
    # Resize the image to match the input size of MobileNetV2
    resized_img = cv2.resize(img, (256, 256))  # Resize
    # Preprocess the image using MobileNetV2 preprocess_input
    preprocessed_img = mobilenet_v2_preprocess_input(resized_img)
    # Expand dimensions to match the model's input shape
    preprocessed_img = np.expand_dims(preprocessed_img, axis=0)
    return preprocessed_img


from tensorflow.keras.preprocessing import image

# Streamlit app
def main():
    st.title("Fire Detection App")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image as numpy array
        img = Image.open(uploaded_file)
        img = img.resize((256, 256))  # Resize the image to match the model's input size
        img = np.array(img)  # Convert image to numpy array
        img = img / 255.0  # Normalize the pixel values

        # Display the uploaded image
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Predict button
        if st.button("Predict"):
            # Preprocess the image
            processed_image = preprocess_image(img)
            # Make prediction
            prediction = model.predict(processed_image)
            predicted_class = np.argmax(prediction)
            st.write(f"Predicted class: {class_labels[predicted_class]}")
if __name__ == '__main__':
    main()            
