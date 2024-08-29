import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from preprocess import load_data

# Build the CNN model
def build_model(input_shape=(64, 64, 1), num_classes=10):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    data_dir = "leapGestRecog"
    X_train, X_test, y_train, y_test = load_data(data_dir)
    
    model = build_model()
    model.summary()
    
    history = model.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test), batch_size=32)
    model.save('hand_gesture_recognition_model.h5')
    print("Model training complete and saved as hand_gesture_recognition_model.h5")
