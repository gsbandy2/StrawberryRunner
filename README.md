# strawberry_runner_classification
This project builds an end-to-end Convolutional Neural Network (CNN) to automatically detect and classify strawberry runners from plant images.

## Structure 

strawberry_runner_classification/
- strawberry_runner_classification/
  - dataset/
    - runner/ (Images of strawberry runners)
    - non_runner/ (Images without runners)
  - strawberry_runner_classifier.h5 (Trained model)
  - strawberry_runner_classifier.py (Full machine learning pipeline)
  - README.md

## Key Features

- Image loading and preprocessing
- Data augmentation to improve model performance
- CNN model for binary classification
- Accuracy evaluation and performance monitoring
- Single image prediction functionality
- Model export for easy deployment

## Dataset Requirements

Organize your dataset as follows:
dataset/
├── runner/ # Images showing strawberry runners
└── non_runner/ # Images without runners

Each image should be in `.jpg` or `.png` format.

You can collect your own images or expand the dataset using web scraping or manual photography.

## How to Run

### 1. Install Required Libraries:
pip install tensorflow opencv-python scikit-learn numpy

### 2. Train the Model:
python strawberry_runner_classifier.py

### 3. Predict New Images:
Open a Python console and run:

from strawberry_runner_classifier import predict_image
result = predict_image('/path/to/image.jpg')
print('Predicted class:', result)


## Model Architecture

- Input: 128x128 RGB images
- Layers:
  - 3 Convolutional layers with ReLU activations
  - MaxPooling layers
  - Fully connected Dense layer
  - Dropout for regularization
- Output: Softmax layer (Runner or Non-Runner)

## Results

The model typically achieves:
- Test Accuracy: Greater than 80% (dependent on dataset size and quality)

To improve results:
- Increase dataset size
- Use transfer learning with pre-trained models (e.g., MobileNet, ResNet)
- Tune hyperparameters

## Example Prediction
Predicted class: Runner

## Future Improvements

- Deploy the model using Streamlit for live predictions
- Build a mobile-friendly classifier
- Add more diverse data (different backgrounds, lighting, growth stages)

