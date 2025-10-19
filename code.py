
"""
Python Deep Learning Project: Synthetic Image Classifier (CNN)

This script uses TensorFlow/Keras to build and train a simple Convolutional Neural Network (CNN)
for a binary classification task on synthetic, image-like data.

To run this script, you must have the following libraries installed:
pip install tensorflow numpy scikit-learn
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

# --- Configuration ---
IMG_SIZE = 40
CHANNELS = 1 # Grayscale
N_SAMPLES = 2000
CLASS_LABELS = ['Class A (Square)', 'Class B (Circle)']
RANDOM_SEED = 42

print(f"--- Deep Learning Setup: CNN for {IMG_SIZE}x{IMG_SIZE} Grayscale Images ---")

# --- 1. DATA GENERATION AND PREPARATION ---

def create_synthetic_data(n_samples, size, seed):
    """Generates synthetic image-like data for binary classification."""
    np.random.seed(seed)
    
    X = np.zeros((n_samples, size, size, 1), dtype=np.float32)
    y = np.zeros(n_samples, dtype=int)
    
    for i in range(n_samples):
        # Generate random noise background
        image = np.random.rand(size, size, 1) * 0.1
        
        # Randomly choose class (0 or 1)
        is_class_b = np.random.randint(0, 2)
        y[i] = is_class_b

        if is_class_b:
            # Class B: Draw a simple "circle" shape (simulated high values in center)
            center_x, center_y = size // 2, size // 2
            radius = size // 4
            for r in range(size):
                for c in range(size):
                    if (r - center_y)**2 + (c - center_x)**2 < radius**2:
                        image[r, c, 0] += 0.9 # Add a bright spot
        else:
            # Class A: Draw a simple "square" shape (simulated high values in a corner)
            start_x, start_y = size // 8, size // 8
            end_x, end_y = size // 2, size // 2
            image[start_x:end_x, start_y:end_y, 0] += 0.9
            
        # Normalize the image (0 to 1)
        X[i] = np.clip(image, 0, 1)
        
    return X, y

# Generate the synthetic dataset
X, y = create_synthetic_data(N_SAMPLES, IMG_SIZE, RANDOM_SEED)

# Binarize labels (0 -> [1, 0], 1 -> [0, 1]) for categorical crossentropy
# This is necessary for two classes when using softmax/categorical_crossentropy
y_one_hot = LabelBinarizer().fit_transform(y)
if y_one_hot.shape[1] == 1:
    y_one_hot = np.hstack([1 - y_one_hot, y_one_hot]) # Ensure 2 columns for binary classification

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_one_hot, test_size=0.2, random_state=RANDOM_SEED
)

print(f"Generated {N_SAMPLES} samples. Training on {len(X_train)}, Testing on {len(X_test)}.")
print(f"Input Shape: {X_train.shape[1:]} (H x W x C)\n")


# --- 2. MODEL DEFINITION (CNN) ---

print("--- Defining Convolutional Neural Network (CNN) ---")

model = Sequential([
    # Convolutional Layer 1: Learn basic features (edges, shapes)
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, CHANNELS)),
    MaxPooling2D((2, 2)), # Reduce spatial size

    # Convolutional Layer 2: Learn more complex features
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Flatten(), # Prepare output for the dense layers

    # Dense Layer: Standard neural network layer
    Dense(128, activation='relu'),

    # Output Layer: 2 units for 2 classes (A and B)
    Dense(2, activation='softmax')
])

# Compile the model
# categorical_crossentropy is used because the labels are one-hot encoded (y_one_hot)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("Model architecture defined.")
# model.summary() # Uncomment to see the layer details


# --- 3. MODEL TRAINING ---

print("\n--- Training Model (5 Epochs) ---")
# Training the model
history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.1, # Use a small portion of training data for validation
    verbose=1 # Show progress
)

print("\nModel training complete.")


# --- 4. MODEL EVALUATION ---

print("\n--- Model Evaluation ---")

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}\n")

# Detailed classification report
y_pred_one_hot = model.predict(X_test)
y_pred = np.argmax(y_pred_one_hot, axis=1) # Convert one-hot back to class index
y_true = np.argmax(y_test, axis=1)

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=CLASS_LABELS))


# --- 5. MAKING A NEW PREDICTION ---

print("\n--- Sample Prediction ---")

# Create one new, random sample data point to test
new_sample_X, new_sample_y = create_synthetic_data(1, IMG_SIZE, 99) # Use a different seed
true_class_index = new_sample_y[0]

# Prediction
prediction_one_hot = model.predict(new_sample_X)
predicted_class_index = np.argmax(prediction_one_hot, axis=1)[0]
confidence = prediction_one_hot[0][predicted_class_index] * 100

# Displaying the result
print(f"Generated True Class: '{CLASS_LABELS[true_class_index]}'")
print(f"The CNN predicts the sample belongs to: '{CLASS_LABELS[predicted_class_index]}'")
print(f"Confidence: {confidence:.2f}%")
