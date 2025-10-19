# Python-Deep-Learning-Project-Synthetic-Image-Classifier


Synthetic Image Classifier (Deep Learning) 🧠✨

Project Tags

💡 Overview

This project is a single-file, self-contained Python script that showcases a complete Deep Learning pipeline using TensorFlow and Keras.

It trains a Convolutional Neural Network (CNN) to perform a binary classification task: distinguishing between two simple, synthetically generated grayscale image patterns.

The Classification Task:

The model learns to identify images belonging to one of two classes:

Class A (Square Pattern)

Class B (Circle Pattern)

🛠️ Requirements

To run this script, you must have the following Python libraries installed. The selected code snippet highlights the necessary installation command:

pip install tensorflow numpy scikit-learn


🚀 Key Features

Custom CNN Architecture: Defines a sequential CNN model with multiple Conv2D and MaxPooling2D layers, optimized for image feature extraction.

Synthetic Dataset Generator: Dynamically creates 2,000 custom 40x40 pixel grayscale images (IMG_SIZE = 40) and corresponding labels, ensuring the script is fully runnable out-of-the-box.

Training & Evaluation: Compiles and trains the model, outputting key metrics like Test Accuracy, Test Loss, and a detailed Classification Report.

Demonstration: Includes a final step to generate a new, unseen sample and demonstrate the model's prediction and confidence score.

▶️ How to Run

Save the code provided in the Canvas (deep_learning_classifier.py).

Ensure all prerequisites are installed (see Requirements section).

Execute the script from your terminal:

python deep_learning_classifier.py


The console output will guide you through the data setup, model training progress, and the final classification results.
