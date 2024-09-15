Image Classification Model
Overview
This project involves training and evaluating a Convolutional Neural Network (CNN) for classifying images into two categories: Cats and Dogs. The model was built using TensorFlow and Keras, and it leverages transfer learning for improved performance. The dataset consists of images collected from the Pexels API.

Project Structure
model_training.ipynb: Jupyter notebook for training the CNN model.
best_model.h5: Saved model with the best performance on validation data.
predict_images.py: Script for predicting image classes from a directory.
/content/pexels_images/: Directory containing images for testing.
Dataset
The dataset used for training the model includes images of cats and dogs. The images were collected using the Pexels API.

Model Details
Architecture: Convolutional Neural Network (CNN)
Epochs: 20
Learning Rate: Initial rate of 0.001, reduced during training
Input Size: 128x128 pixels
Loss Function: Binary Cross-Entropy
Optimizer: Adam
Training Results
The model was trained for 20 epochs with the following results:

Final Training Accuracy: 94.12%
Final Validation Accuracy: 87.05%
Test Accuracy: 88.98%
Test Loss: 0.3217
Prediction Results
The model was evaluated on 80 images, of which it successfully predicted 49 as cats. This indicates a good performance in identifying cats among the images.
