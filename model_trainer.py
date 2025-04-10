import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Constants and Configuration
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
NUM_CLASSES = len(EMOTIONS)
IMAGE_SIZE = 48
BATCH_SIZE = 64
EPOCHS = 50
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'emotion_model.h5')

# Create models directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# Function to download FER2013 dataset from Kaggle
def download_fer2013():
    """
    Downloads the FER2013 dataset from Kaggle.
    You need to have a Kaggle account and API key set up.
    """
    try:
        print("Attempting to download FER2013 dataset...")
        # Try to import kaggle package
        import kaggle
        # Download the dataset
        kaggle.api.dataset_download_files('msambare/fer2013', path='.', unzip=True)
        print("FER2013 dataset downloaded successfully.")
        return True
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please download the FER2013 dataset manually from https://www.kaggle.com/datasets/msambare/fer2013")
        print("After downloading, place the CSV file in the current directory.")
        return False

# Function to load FER2013 dataset
def load_fer2013(dataset_path='fer2013.csv'):
    """
    Loads the FER2013 dataset from CSV
    Returns X (images) and y (labels)
    """
    print(f"Loading dataset from {dataset_path}...")
    
    if not os.path.exists(dataset_path):
        # Try to download
        if not download_fer2013():
            print("Dataset not found and automatic download failed.")
            return None, None
    
    try:
        # Load data
        data = pd.read_csv(dataset_path)
        
        # Extract pixel values and labels
        pixels = data['pixels'].values
        emotions = data['emotion'].values
        
        # Convert pixels to images
        X = []
        for pixel_sequence in pixels:
            face = [int(pixel) for pixel in pixel_sequence.split(' ')]
            face = np.array(face).reshape(IMAGE_SIZE, IMAGE_SIZE)
            X.append(face)
        
        X = np.array(X)
        
        # Normalize images to [0, 1]
        X = X / 255.0
        
        # Reshape to include channel dimension
        X = X.reshape(X.shape[0], IMAGE_SIZE, IMAGE_SIZE, 1)
        
        # Convert labels to one-hot encoding
        y = to_categorical(emotions, NUM_CLASSES)
        
        print(f"Loaded dataset with {X.shape[0]} samples.")
        return X, y
    
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None

# Build CNN model for emotion recognition
def build_model():
    """
    Creates and returns a CNN model for emotion recognition
    """
    model = Sequential()
    
    # First convolution block
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Second convolution block
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Third convolution block
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss=CategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    print(model.summary())
    return model

# Function to train the model
def train_model(X_train, y_train, X_val, y_val):
    """
    Trains the emotion recognition model
    """
    # Initialize the model
    model = build_model()
    
    # Set up callbacks
    checkpoint = ModelCheckpoint(
        MODEL_PATH,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=15,
        verbose=1,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )
    
    callbacks = [checkpoint, early_stopping, reduce_lr]
    
    # Data augmentation for training
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )
    
    # Train the model
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )
    
    return model, history

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model on the test dataset
    """
    # Evaluate the model
    scores = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Loss: {scores[0]}")
    print(f"Test Accuracy: {scores[1]}")
    
    return scores

# Function to plot training history
def plot_training_history(history):
    """
    Plots the training and validation accuracy/loss curves
    """
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    
    print("Training history plot saved as 'training_history.png'")

# Main function
def main():
    print("Facial Emotion Recognition Model Training")
    print("========================================")
    
    # Load dataset
    X, y = load_fer2013()
    if X is None or y is None:
        print("Failed to load dataset. Exiting.")
        return
    
    # Split dataset into training, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train the model
    print("\nTraining model...")
    model, history = train_model(X_train, y_train, X_val, y_val)
    
    # Evaluate the model
    print("\nEvaluating model...")
    evaluate_model(model, X_test, y_test)
    
    # Plot training history
    plot_training_history(history)
    
    print(f"\nModel saved as '{MODEL_PATH}'")
    print("Done! You can now use this model with ChromaMood.")

if __name__ == "__main__":
    main()