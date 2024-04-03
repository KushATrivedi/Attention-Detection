import os
import numpy as np
import tensorflow as tf
import cv2
import time
 
class_names = ['attentive', 'no vigilant', 'drowsy']
 
interpreter = tf.lite.Interpreter(model_path=r"drowsy_model.tflite")
interpreter.allocate_tensors()
 
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
 
output_directory = r"app_output"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
 
cap = cv2.VideoCapture(0)
count = 0
 
# video for 1 minute
start_time = time.time()
while time.time() - start_time <= 60:
    ret, frame = cap.read()
    count += 1
 
    # Display live video
    cv2.imshow('Video Capture', frame)
 
    # frame saved every 10 seconds
    if count % 300 == 0:  
        file_name = f"{output_directory}/frame_{count // 30}.jpg"
        cv2.imwrite(file_name, frame)
 
        img = cv2.resize(frame, (256, 256))
        img = np.array(img, dtype="float32")
        img = np.reshape(img, (1, 256, 256, 3))
 
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
 
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_label = np.argmax(output_data, axis=1)
 
        predicted_class_name = class_names[predicted_label[0]]
 
        print(f"Frame: {count}, Predicted class: {predicted_class_name}")
 
    # Break the loop 
    if cv2.waitKey(1) == 13:
        break
 
# Release the video capture
cap.release()
cv2.destroyAllWindows()

from collections import Counter
 
# List of predicted class names (replace this with your actual list of predictions)
predicted_class_names = ['attentive', 'no vigilant', 'drowsy']
 
# Use Counter to count occurrences of each class name
class_counts = Counter(predicted_class_names)
 
# Find the class name with the maximum count
most_frequent_class = class_counts.most_common(1)[0][0]
 
print("Most frequent predicted class:", most_frequent_class)

print("Video Capture and Prediction Complete")