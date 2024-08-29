import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def load_data(data_dir, img_size=(64, 64)):
    gestures = ['01_palm', '02_l', '03_fist', '04_fist_moved', '05_thumb', 
                '06_index', '07_ok', '08_palm_moved', '09_c', '10_down']
    
    data = []
    labels = []
    
    for folder_num in range(10):  # There are 10 folders (00, 01, ..., 09)
        folder_path = os.path.join(data_dir, f"{folder_num:02d}")
        for i, gesture in enumerate(gestures):
            gesture_folder = os.path.join(folder_path, gesture)
            for img_name in os.listdir(gesture_folder):
                img_path = os.path.join(gesture_folder, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, img_size)
                data.append(img)
                labels.append(i)
    
    data = np.array(data).reshape(-1, img_size[0], img_size[1], 1) / 255.0
    labels = to_categorical(labels, num_classes=len(gestures))
    
    return train_test_split(data, labels, test_size=0.2, random_state=42)

if __name__ == "__main__":
    data_dir = "leapGestRecog"
    X_train, X_test, y_train, y_test = load_data(data_dir)
    print("Data loaded and split into train and test sets.")
