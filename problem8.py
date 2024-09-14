import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Function to load audio data and extract MFCC features
def load_audio_data(data_dir):
    X = []
    y = []
    labels = os.listdir(data_dir)
    
    for label in labels:
        label_dir = os.path.join(data_dir, label)
        for file_name in os.listdir(label_dir):
            file_path = os.path.join(label_dir, file_name)
            # Load the audio file
            audio, sample_rate = librosa.load(file_path, sr=None)
            
            # Extract MFCC features
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
            mfccs = np.mean(mfccs.T, axis=0)  # Taking the mean of MFCC features across time
            X.append(mfccs)
            y.append(label)
    
    return np.array(X), np.array(y)

# Function to process the recorded audio and extract MFCC features
def extract_mfcc_from_audio(audio, sample_rate, n_mfcc=13):
    mfccs = librosa.feature.mfcc(y=audio.flatten(), sr=sample_rate, n_mfcc=n_mfcc)
    mfccs = np.mean(mfccs.T, axis=0)  # Take the mean across time axis
    return mfccs.reshape(1, -1)  # Return as a 2D array for model input

# Function to predict a digit from a sample audio file
def predict_digit_from_file(model, le, file_path):
    audio, sample_rate = librosa.load(file_path, sr=None)  # Load the sample audio file
    mfccs = extract_mfcc_from_audio(audio, sample_rate)
    prediction = model.predict(mfccs)
    predicted_label = np.argmax(prediction)
    
    # Ensure predicted label is within valid range
    if predicted_label >= len(le.classes_):
        raise ValueError(f"Predicted label {predicted_label} is out of bounds. Expected range: [0, {len(le.classes_)-1}]")
    
    print(f"Predicted label from file: {le.inverse_transform([predicted_label])[0]}")
    return predicted_label

# Set paths to your dataset
data_dir = r'E:\Course\4-2\Neural Networks-MIH\Sessional\My Code\Audio'  # Dataset directory with subfolders for each class (1 to 4)

# Load the audio data and their corresponding labels
X, y = load_audio_data(data_dir)

# Encode the labels to integers
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Check if the encoding is within valid range
if not np.all(np.isin(y_encoded, np.arange(4))):
    raise ValueError("Encoded labels are out of bounds. Expected range: [0, 3]")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Build the ANN model
model = models.Sequential([
    layers.InputLayer(input_shape=(13,)),  # 13 MFCC coefficients
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(4, activation='softmax')  # 4 output classes (numbers 0 to 3)
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the ANN model
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), batch_size=16)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.2f}")

# Save the trained model
model_save_path = r'E:\Course\4-2\Neural Networks-MIH\Sessional\My Code\speech_model.h5'
model.save(model_save_path)

# Plot training and validation accuracy and loss curves
def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    
    # Plot accuracy
    plt.figure()
    plt.plot(epochs, acc, 'bo-', label='Training accuracy')
    plt.plot(epochs, val_acc, 'ro-', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Plot loss
    plt.figure()
    plt.plot(epochs, loss, 'bo-', label='Training loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

plot_history(history)

# Predict a digit from a sample audio file
sample_audio_file = r'E:\Course\4-2\Neural Networks-MIH\Sessional\My Code\Sample\sample2.wav'
predicted_digit = predict_digit_from_file(model, le, sample_audio_file)
