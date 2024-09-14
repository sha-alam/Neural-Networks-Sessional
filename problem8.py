import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
import matplotlib.pyplot as plt

# Generate synthetic data for digits 1 to 5
def generate_synthetic_data(num_samples=1000):
    X = np.zeros((num_samples, 5, 5, 1), dtype=np.float32)  # Black and white images
    y = np.zeros(num_samples, dtype=np.int)  # Labels
    
    for i in range(num_samples):
        digit = np.random.randint(1, 6)  # Random digit from 1 to 5
        y[i] = digit - 1  # Convert to zero-based index
        
        # Create a simple pattern for the digit (this is just for demonstration)
        if digit == 1:
            X[i, :, :, 0] = np.array([[0, 0, 1, 0, 0],
                                      [0, 1, 1, 0, 0],
                                      [0, 1, 1, 0, 0],
                                      [0, 1, 1, 0, 0],
                                      [0, 1, 1, 0, 0]], dtype=np.float32)
        elif digit == 2:
            X[i, :, :, 0] = np.array([[1, 1, 1, 1, 0],
                                      [0, 0, 0, 1, 0],
                                      [0, 1, 1, 0, 0],
                                      [1, 1, 0, 0, 0],
                                      [1, 1, 1, 1, 1]], dtype=np.float32)
        elif digit == 3:
            X[i, :, :, 0] = np.array([[1, 1, 1, 1, 0],
                                      [0, 0, 1, 1, 0],
                                      [0, 1, 1, 1, 0],
                                      [0, 0, 1, 1, 0],
                                      [1, 1, 1, 1, 0]], dtype=np.float32)
        elif digit == 4:
            X[i, :, :, 0] = np.array([[1, 0, 0, 1, 0],
                                      [1, 0, 0, 1, 0],
                                      [1, 1, 1, 1, 0],
                                      [0, 0, 0, 1, 0],
                                      [0, 0, 0, 1, 0]], dtype=np.float32)
        elif digit == 5:
            X[i, :, :, 0] = np.array([[1, 1, 1, 1, 1],
                                      [1, 0, 0, 0, 0],
                                      [1, 1, 1, 1, 0],
                                      [0, 0, 0, 0, 1],
                                      [1, 1, 1, 1, 0]], dtype=np.float32)
    
    y = tf.keras.utils.to_categorical(y, num_classes=5)
    return X, y

# Generate data
X, y = generate_synthetic_data()

# Define CNN architecture
model = models.Sequential([
    layers.InputLayer(input_shape=(5, 5, 1)),

    layers.Conv2D(16, (3, 3), padding='valid', activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(5, activation='softmax')  # 5 classes for digits 1 to 5
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the CNN
history = model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

# Save the trained model
model_save_path = 'digit_recognition_model.h5'
model.save(model_save_path)

# Load and preprocess a random sample image for classification
def predict_and_display(image):
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)
    return predicted_class + 1  # Convert back to digit (1 to 5)

# Create a sample test image
test_image = np.array([[1, 1, 1, 1, 1],
                       [1, 0, 0, 0, 0],
                       [1, 1, 1, 1, 0],
                       [0, 0, 0, 0, 1],
                       [1, 1, 1, 1, 0]], dtype=np.float32)
test_image = np.expand_dims(test_image, axis=-1)  # Add channel dimension

# Predict and display result
predicted_digit = predict_and_display(test_image)
print(f'Predicted digit: {predicted_digit}')

# Display the test image
plt.imshow(test_image.squeeze(), cmap='gray')
plt.title(f'Predicted: {predicted_digit}')
plt.axis('off')
plt.show()
