import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
from tempfile import NamedTemporaryFile
from ultralytics import YOLO
import os
import glob
import cv2

# Function to preprocess the image
def preprocess_image(image, model_type):
    if model_type == "CNN Model":
        resized_image = image.resize((350, 350))  # Resize the image to match the input shape of the first model
    else:
        resized_image = image.resize((224, 224))  # Resize the image to match the input shape of the second model
    image_array = np.array(resized_image)  # Convert image to array
    image_array = image_array / 255.0  # Normalize pixel values to the range [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Load the pre-trained models
@st.cache(allow_output_mutation=True)
def load_pretrained_model(model_type):
    if model_type == "CNN Model":
        return load_model(r'C:\Users\Rishav\Desktop\me\code\projects\dataset_pbl\first_model.hdf5')
    else:
        return load_model(r'C:\Users\Rishav\Desktop\me\code\projects\dataset_pbl\second_model.hdf5')

# Function to get the class name from the predicted index
def get_class_name(index):
    # Replace this with your class names or class labels
    class_names = ['No Wildfire detected', 'Wildfire detected']  # Example class names
    return class_names[index]

# Function to get the latest video file in the output directory
def get_latest_detected_video():
    # Specify the directory where the detected videos are stored
    detection_dir = r'C:\Users\Rishav\Desktop\me\code\projects\dataset_pbl\runs\detect'
    # Use glob to find all directories with name starting with 'predict'
    detection_dirs = glob.glob(os.path.join(detection_dir, 'predict*'))
    # Sort the directories based on their creation time
    detection_dirs.sort(key=os.path.getmtime, reverse=True)
    # Get the latest directory
    latest_dir = detection_dirs[0] if detection_dirs else None
    # Get the path of the detected video within the latest directory
    detected_video_path = None
    if latest_dir:
        video_files = glob.glob(os.path.join(latest_dir, '*.avi'))
        detected_video_path = video_files[0] if video_files else None
    return detected_video_path

# Main function to run the Streamlit app
def main():
    st.title("Wildfire Prediction App")
    
    # Upload an image or video for prediction
    uploaded_file = st.file_uploader("Choose an image or video...", type=["jpg", "jpeg", "png", "mp4"])
    
    # If an image or video is uploaded, display it and allow prediction
    if uploaded_file is not None:
        # Check if the uploaded file is a video
        if uploaded_file.type == 'video/mp4':
            # Display the uploaded video
            st.video(uploaded_file)
            
            # Dropdown menu to select the model
            video_model_type = st.selectbox("Select Model", ["YOLOv8 Model", "YOLOv9 Model"])
            
            # Button for making detection on the video
            if st.button("Detect on Video"):
                # Save the video to a temporary file
                with NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
                    temp_file.write(uploaded_file.read())
                    video_path = temp_file.name
                    print("Video path:", video_path)  # Debugging line
                
                # Load the selected model
                if video_model_type == "YOLOv8 Model":
                    yolo_model = YOLO(r'C:\Users\Rishav\Desktop\me\code\projects\dataset_pbl\best.pt')
                else:  # YOLOv9 Model
                    yolo_model = YOLO(r'C:\Users\Rishav\Desktop\me\code\projects\dataset_pbl\YoloV9.pt')
                
                # Perform detection on the video
                yolo_model.predict(source=video_path, imgsz=640, conf=0.20, save=True)
                
                # Get the path of the detected video
                detected_video_path = get_latest_detected_video()
                
                # Display the detected video path
                if detected_video_path:
                    st.write("Detected video saved at:", detected_video_path)
                else:
                    st.write("No detected video found.")
        
        # If an image is uploaded
        else:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            # Dropdown menu to select the model
            image_model_type = st.selectbox("Select Model", ["CNN Model", "RestNet50 Model"])
            
            # Preprocess the image
            preprocessed_image = preprocess_image(image, image_model_type)
            
            # Load the selected model
            model = load_pretrained_model(image_model_type)
            
            # Predict button
            if st.button("Predict"):
                # Perform prediction
                predictions = model.predict(preprocessed_image)
                predicted_index = np.argmax(predictions)
                
                # Get the predicted class name
                predicted_class = get_class_name(predicted_index)
                
                # Determine text color based on the prediction result
                text_color = "red" if predicted_class == 'Wildfire detected' else "green"
                
                # Display the predicted class name and index with text color based on prediction result
                st.write(f'Predicted Class Name: <span style="color:{text_color};">{predicted_class}</span>', 
                         unsafe_allow_html=True)

# Run the Streamlit app
if __name__ == "__main__":
    main()
