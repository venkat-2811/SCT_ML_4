import cv2
import numpy as np
from keras.models import load_model

# Load your trained model
model = load_model('hand_gesture_recognition_model.h5')

# Start capturing video from the webcam (DroidCam Client)
cap = cv2.VideoCapture(1)  # Make sure the correct camera index is used

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to capture image")
        break

    # Resize the frame to match the input shape of the model
    resized = cv2.resize(frame, (224, 224))  # Adjust the size according to your model's input shape
    resized = resized.astype('float32') / 255.0  # Normalize pixel values to [0, 1]
    
    # Expand dimensions to match the expected input shape of the model
    input_data = np.expand_dims(resized, axis=0)  # Shape becomes (1, 224, 224, 3) if input was (224, 224, 3)

    # Make predictions
    prediction = model.predict(input_data)

    # Interpret prediction here, e.g., find the class with the highest probability
    predicted_class = np.argmax(prediction, axis=1)
    print(f"Predicted class: {predicted_class}")

    # Display the frame with prediction
    cv2.putText(frame, f'Predicted: {predicted_class[0]}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('Hand Gesture Recognition', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
