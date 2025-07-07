# Strawberry Runner Identification and Classification Pipeline

# 1. Data Collection & Preprocessing
# Assuming you will gather a dataset of labeled strawberry runners and non-runners images

import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

# Directory structure assumption:
# dataset/
#    runner/
#    non_runner/

DATA_DIR = 'dataset'
IMG_SIZE = 128

def load_images(data_dir):
    images = []
    labels = []
    categories = ['runner', 'non_runner']

    for label, category in enumerate(categories):
        path = os.path.join(data_dir, category)
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            img_array = cv2.imread(img_path)
            if img_array is not None:
                img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                images.append(img_array)
                labels.append(label)

    return np.array(images), np.array(labels)

# Load and split data
images, labels = load_images(DATA_DIR)
images = images / 255.0  # Normalize
labels = to_categorical(labels)

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# 2. Model Building
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 3. Model Training
datagen = ImageDataGenerator(rotation_range=30, zoom_range=0.2, horizontal_flip=True)
datagen.fit(X_train)

history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    epochs=20, validation_data=(X_test, y_test))

# 4. Model Evaluation
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# 5. Prediction Example
def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    class_idx = np.argmax(prediction)
    class_labels = ['Runner', 'Non-Runner']
    return class_labels[class_idx]

# Example Usage
# result = predict_image('/path/to/test/image.jpg')
# print('Predicted class:', result)

# 6. Save the Model
model.save('strawberry_runner_classifier.h5')

# README Suggestion:
# 1. Brief about the project.
# 2. Explain data structure and how to collect/build the dataset.
# 3. Explain the model architecture.
# 4. Instructions for running the code locally.
# 5. Example image predictions and model performance screenshots.
