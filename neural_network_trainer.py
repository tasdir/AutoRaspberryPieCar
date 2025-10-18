"""
Neural Network Training Module for Autonomous Driving
Train a CNN to predict steering angles from road images
"""

import os
import json
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class CNNTrainer:
    """
    Convolutional Neural Network trainer for autonomous driving
    Uses TensorFlow/Keras for model training
    """
    
    def __init__(self, data_dir='training_data', model_name='lane_model'):
        """
        Initialize trainer
        Args:
            data_dir: Directory containing training data
            model_name: Name for saved model
        """
        self.data_dir = data_dir
        self.model_name = model_name
        self.model = None
        self.history = None
        
        # Try to import TensorFlow
        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers
            self.tf = tf
            self.keras = keras
            self.layers = layers
            print("TensorFlow version:", tf.__version__)
        except ImportError:
            print("Warning: TensorFlow not installed.")
            print("Install with: pip install tensorflow")
            self.tf = None
            
    def load_data(self, test_size=0.2):
        """
        Load training data from directory
        Args:
            test_size: Fraction of data to use for testing
        Returns:
            X_train, X_test, y_train, y_test
        """
        print("Loading dataset...")
        
        # Load labels
        labels_file = os.path.join(self.data_dir, 'labels.json')
        images_dir = os.path.join(self.data_dir, 'images')
        
        if not os.path.exists(labels_file):
            raise Exception(f"Labels file not found: {labels_file}")
            
        with open(labels_file, 'r') as f:
            dataset_info = json.load(f)
            
        data = dataset_info['data']
        print(f"Total samples: {len(data)}")
        
        # Load images and labels
        X = []
        y = []
        
        for i, entry in enumerate(data):
            image_path = os.path.join(images_dir, entry['image'])
            
            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}")
                continue
                
            # Load and preprocess image
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0  # Normalize to [0, 1]
            
            X.append(img)
            y.append(entry['curve'])
            
            if (i + 1) % 100 == 0:
                print(f"Loaded {i + 1}/{len(data)} images")
                
        X = np.array(X)
        y = np.array(y)
        
        print(f"\nDataset loaded successfully!")
        print(f"Images shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        print(f"Label range: [{y.min():.3f}, {y.max():.3f}]")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        print(f"\nTraining samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
        
    def build_model(self, input_shape=(240, 480, 3)):
        """
        Build CNN model architecture
        Based on NVIDIA's end-to-end learning architecture
        Args:
            input_shape: Shape of input images (height, width, channels)
        """
        if self.tf is None:
            raise Exception("TensorFlow not available")
            
        print("\nBuilding model...")
        
        model = self.keras.Sequential([
            # Input layer (normalization is done during preprocessing)
            self.layers.InputLayer(input_shape=input_shape),
            
            # Convolutional layers
            self.layers.Conv2D(24, (5, 5), strides=(2, 2), activation='relu'),
            self.layers.Conv2D(36, (5, 5), strides=(2, 2), activation='relu'),
            self.layers.Conv2D(48, (5, 5), strides=(2, 2), activation='relu'),
            self.layers.Conv2D(64, (3, 3), activation='relu'),
            self.layers.Conv2D(64, (3, 3), activation='relu'),
            
            # Flatten
            self.layers.Flatten(),
            
            # Fully connected layers
            self.layers.Dense(100, activation='relu'),
            self.layers.Dropout(0.5),
            self.layers.Dense(50, activation='relu'),
            self.layers.Dropout(0.5),
            self.layers.Dense(10, activation='relu'),
            
            # Output layer (steering angle)
            self.layers.Dense(1, activation='tanh')
        ])
        
        # Compile model
        model.compile(
            optimizer=self.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        
        print("\nModel architecture:")
        model.summary()
        
        return model
        
    def train(self, X_train, y_train, X_test, y_test, 
              epochs=50, batch_size=32):
        """
        Train the model
        Args:
            X_train, y_train: Training data
            X_test, y_test: Testing data
            epochs: Number of training epochs
            batch_size: Batch size
        """
        if self.model is None:
            raise Exception("Model not built. Call build_model() first.")
            
        print("\n" + "="*60)
        print("Starting training...")
        print("="*60)
        
        # Callbacks
        callbacks = [
            self.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            self.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001
            ),
            self.keras.callbacks.ModelCheckpoint(
                f'{self.model_name}_best.h5',
                monitor='val_loss',
                save_best_only=True
            )
        ]
        
        # Train
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\nTraining completed!")
        
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        Args:
            X_test, y_test: Test data
        """
        if self.model is None:
            raise Exception("Model not available")
            
        print("\n" + "="*60)
        print("Evaluating model...")
        print("="*60)
        
        # Evaluate
        loss, mae = self.model.evaluate(X_test, y_test, verbose=0)
        
        print(f"Test Loss (MSE): {loss:.4f}")
        print(f"Test MAE: {mae:.4f}")
        
        # Predictions
        y_pred = self.model.predict(X_test, verbose=0).flatten()
        
        # Calculate additional metrics
        mse = np.mean((y_test - y_pred) ** 2)
        rmse = np.sqrt(mse)
        r2 = 1 - (np.sum((y_test - y_pred) ** 2) / 
                  np.sum((y_test - np.mean(y_test)) ** 2))
        
        print(f"RMSE: {rmse:.4f}")
        print(f"R² Score: {r2:.4f}")
        
        return y_pred
        
    def plot_training_history(self, save_path='training_history.png'):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(self.history.history['loss'], label='Training Loss')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (MSE)')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # MAE plot
        ax2.plot(self.history.history['mae'], label='Training MAE')
        ax2.plot(self.history.history['val_mae'], label='Validation MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.set_title('Training and Validation MAE')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"\nTraining history plot saved: {save_path}")
        plt.close()
        
    def plot_predictions(self, y_test, y_pred, save_path='predictions.png'):
        """Plot predictions vs actual values"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Scatter plot
        ax1.scatter(y_test, y_pred, alpha=0.5)
        ax1.plot([-1, 1], [-1, 1], 'r--', lw=2)
        ax1.set_xlabel('Actual Steering Angle')
        ax1.set_ylabel('Predicted Steering Angle')
        ax1.set_title('Predictions vs Actual')
        ax1.grid(True)
        ax1.axis('equal')
        
        # Error distribution
        errors = y_pred - y_test
        ax2.hist(errors, bins=50, edgecolor='black')
        ax2.set_xlabel('Prediction Error')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Error Distribution')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Predictions plot saved: {save_path}")
        plt.close()
        
    def save_model(self, filepath=None):
        """Save trained model"""
        if self.model is None:
            raise Exception("No model to save")
            
        if filepath is None:
            filepath = f'{self.model_name}.h5'
            
        self.model.save(filepath)
        print(f"\nModel saved: {filepath}")
        
    def load_model(self, filepath=None):
        """Load saved model"""
        if self.tf is None:
            raise Exception("TensorFlow not available")
            
        if filepath is None:
            filepath = f'{self.model_name}.h5'
            
        self.model = self.keras.models.load_model(filepath)
        print(f"Model loaded: {filepath}")
        
        return self.model


def main():
    """Main training pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train neural network for autonomous driving')
    parser.add_argument('--data-dir', type=str, default='training_data',
                        help='Directory containing training data')
    parser.add_argument('--model-name', type=str, default='lane_model',
                        help='Name for saved model')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Fraction of data for testing')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = CNNTrainer(args.data_dir, args.model_name)
    
    # Load data
    X_train, X_test, y_train, y_test = trainer.load_data(test_size=args.test_size)
    
    # Build model
    trainer.build_model(input_shape=X_train.shape[1:])
    
    # Train model
    trainer.train(X_train, y_train, X_test, y_test,
                  epochs=args.epochs, batch_size=args.batch_size)
    
    # Evaluate
    y_pred = trainer.evaluate(X_test, y_test)
    
    # Plot results
    trainer.plot_training_history()
    trainer.plot_predictions(y_test, y_pred)
    
    # Save model
    trainer.save_model()
    
    print("\n" + "="*60)
    print("Training pipeline completed!")
    print("="*60)


if __name__ == '__main__':
    main()

