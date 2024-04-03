import os
import numpy as np
import tensorflow as tf
import cv2
import time
import streamlit as st
from collections import Counter
 
# Define class names in order
class_names = ['Attentive', 'Non vigilant', 'Drowsy']
 
# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=r"drowsy_model.tflite")
interpreter.allocate_tensors()
 
# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
 
# Create a directory to save the captured images
output_directory = r"streamlit_output"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
 
# Streamlit UI
st.title("Drowsiness Detection")
 
# Start video capture
start_capture = st.button("Start Capture")
 
if start_capture:
    # Start video capture
    cap = cv2.VideoCapture(0)
    count = 0
    start_time = time.time()
    capture_interval = 10
 
    while time.time() - start_time <= 60:  # Capture video for 1 minute
        ret, frame = cap.read()
        count += 1
 
        # Save a frame every 10 seconds
        if count % 300 == 0:
            file_name = f"{output_directory}/frame_{count // 30}.jpg"
            cv2.imwrite(file_name, frame)
            img = cv2.resize(frame, (256, 256))
            img = np.array(img, dtype="float32")
            img = np.reshape(img, (1, 256, 256, 3))
            img /= 255.0  # Normalize the image
            st.image(frame, channels="BGR", use_column_width=True, caption="Image Capture")
 
            # Test the model on the input data.
            interpreter.set_tensor(input_details[0]['index'], img)
            interpreter.invoke()
 
            # Get the predicted label
            output_data = interpreter.get_tensor(output_details[0]['index'])
            predicted_label = np.argmax(output_data, axis=1)
 
            # Get the predicted class name
            predicted_class_name = class_names[predicted_label[0]]
 
            # Display the prediction
            st.write(f"Frame: {count}, Predicted class: {predicted_class_name}")
 
        # Check if "Start Capture" button is unpressed
        if not start_capture:
            break
 
    # Release the video capture
    cap.release()
    cv2.destroyAllWindows()
 
    # List of predicted class names (replace this with your actual list of predictions)
    predicted_class_names = ['Attentive', 'Non vigilant', 'Drowsy']
 
    # Use Counter to count occurrences of each class name
    class_counts = Counter(predicted_class_names)
 
    # Find the class name with the maximum count
    most_frequent_class = class_counts.most_common(1)[0][0]
 
    st.write("Most frequent predicted class:", most_frequent_class)
    st.write("Video Capture and Prediction Complete")