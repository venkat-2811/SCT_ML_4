# Hand Gesture Recognition

## Project Overview

This project is designed to recognize hand gestures using a webcam feed. The system captures real-time video from a webcam, processes each frame, and predicts the gesture shown in the frame using a pre-trained deep learning model. This project can be useful for various applications, including controlling devices, virtual reality interactions, and more.

## Features

- **Real-time Gesture Recognition**: The system captures video from a webcam and recognizes hand gestures in real-time.
- **Pre-trained Model Integration**: The project uses a pre-trained deep learning model for gesture recognition, allowing for accurate predictions.
- **User-Friendly Interface**: The project displays the live webcam feed with the predicted gesture label overlaid on the video.

## Requirements

### Hardware:
- A computer with a webcam (using DroidCam Client or other webcam software).

### Software:
- Python 3.x
- OpenCV (`cv2`)
- NumPy
- TensorFlow/Keras (or any other deep learning framework you used to train your model)
- DroidCam Client (for using your mobile as a webcam)

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/venkat-2811/SCT_ML_4.git
   cd hand-gesture-recognition
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Make sure `requirements.txt` includes:
   ```
   opencv-python
   numpy
   tensorflow
   ```

3. **Set Up DroidCam (if using a smartphone as a webcam)**:
   - Install the DroidCam app on your smartphone.
   - Install the DroidCam Client on your computer.
   - Connect your smartphone to your computer via Wi-Fi or USB and follow the DroidCam instructions to start using it as a webcam.

## Usage

1. **Load the Pre-trained Model**:
   Ensure that you have your trained model saved as `model.h5` or any other name. Update the script with the correct path to the model file.

2. **Run the Script**:
   ```bash
   python hand_gesture_recognition.py
   ```

3. **Start Gesture Recognition**:
   - The script will start capturing video from your webcam.
   - It will process each frame, predict the gesture, and display the prediction on the video feed.
   - Press `q` to quit the application.

## Project Structure

```
hand-gesture-recognition/
│
├── model.py               # Pre-trained gesture recognition model
├── app.py
├── hand_gesture_recognition.py  # Main script for running the project
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies
```

## Customization

- **Model Training**: If you want to train your own model, you can modify the script to include a training pipeline or use another script dedicated to training. Ensure your dataset is organized and preprocessed correctly.
- **Webcam Feed**: If you have multiple cameras or a different setup, you may need to adjust the `cv2.VideoCapture()` index to match your webcam source.
- **Gesture Classes**: Modify the script to reflect the number of gesture classes and their corresponding labels based on your model's output.

## Future Enhancements

- **Add More Gestures**: Expand the number of gestures recognized by the system.
- **Optimize Model**: Improve the accuracy and speed of the model with further training and optimization techniques.
- **Cross-Platform Support**: Ensure compatibility with various operating systems and webcam configurations.

## Troubleshooting

- **Webcam Not Detected**: Ensure that the correct camera index is used in the `cv2.VideoCapture()` method.
- **Incorrect Predictions**: Ensure that the image preprocessing steps match those used during model training (e.g., image size, normalization).
- **Performance Issues**: Consider optimizing the model or reducing the video resolution to improve real-time performance.

## Contributing

Contributions are welcome! If you have ideas for improvements or new features, feel free to submit a pull request.
