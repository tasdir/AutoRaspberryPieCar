"""
Neural Network-based Autonomous Driving
Uses trained CNN model for steering prediction
"""

import cv2
import numpy as np
import motor_control
import time
import argparse


class NeuralAutonomousCar:
    """Autonomous car using neural network for steering prediction"""
    
    def __init__(self, model_path, video_source=0, use_gpio=True, 
                 base_speed=50, display_mode=1):
        """
        Initialize neural autonomous car
        Args:
            model_path: Path to trained model (.h5 file)
            video_source: 0 for Pi camera, or path to video file
            use_gpio: Enable/disable GPIO control
            base_speed: Base driving speed (0-100)
            display_mode: 0=no display, 1=show video with predictions
        """
        self.video_source = video_source
        self.base_speed = base_speed
        self.display_mode = display_mode
        self.is_running = False
        
        # Load neural network model
        print(f"Loading model: {model_path}")
        self.model = self.load_model(model_path)
        
        # Initialize camera/video
        print(f"Initializing camera/video: {video_source}")
        self.cap = cv2.VideoCapture(video_source)
        
        if not self.cap.isOpened():
            raise Exception(f"Could not open video source: {video_source}")
            
        # Initialize motor control
        print(f"Initializing motor control (GPIO: {use_gpio})")
        self.motors = motor_control.MotorControl(use_gpio=use_gpio)
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        
        # Prediction smoothing
        self.prediction_history = []
        self.smoothing_window = 5
        
    def load_model(self, model_path):
        """Load trained Keras model"""
        try:
            import tensorflow as tf
            from tensorflow import keras
            
            # Check TensorFlow version
            tf_version = tf.__version__
            print(f"TensorFlow version: {tf_version}")
            
            # For newer TensorFlow versions with stricter security
            # Load without compilation to avoid Lambda layer issues
            try:
                # Try standard loading first
                model = tf.keras.models.load_model(model_path)
                print("Model loaded successfully!")
            except (ValueError, TypeError) as e:
                error_msg = str(e)
                if "Lambda" in error_msg or "safe_mode" in error_msg or "deserialize" in error_msg:
                    print("Warning: Model compatibility issue detected.")
                    print("Loading without compilation and recompiling...")
                    
                    # Load without compilation
                    try:
                        # For Keras 3.x
                        keras.config.enable_unsafe_deserialization()
                        model = tf.keras.models.load_model(model_path, compile=False)
                    except:
                        # Try alternative approach
                        import h5py
                        print("Alternative loading method...")
                        model = tf.keras.models.load_model(
                            model_path, 
                            compile=False,
                            custom_objects={'tf': tf}
                        )
                    
                    # Recompile the model with correct metrics
                    print("Recompiling model...")
                    model.compile(
                        optimizer=keras.optimizers.Adam(learning_rate=0.001),
                        loss='mse',
                        metrics=['mae']
                    )
                    print("Model loaded and recompiled successfully!")
                else:
                    raise
            
            print(f"Input shape: {model.input_shape}")
            print(f"Output shape: {model.output_shape}")
            return model
        except ImportError:
            raise Exception("TensorFlow not installed. Install with: pip install tensorflow")
        except Exception as e:
            print("\n" + "="*60)
            print("MODEL LOADING FAILED")
            print("="*60)
            print(f"Error: {e}")
            print("\nThe model was created with an older version and has compatibility issues.")
            print("\nRECOMMENDED SOLUTION: Retrain the model with the updated architecture.")
            print("\nSteps:")
            print("1. python neural_network_trainer.py --data-dir training_data --epochs 50")
            print("2. python neural_drive.py --model lane_model.h5 --video vid1.mp4 --no-gpio")
            print("\nThe new model will not have Lambda layer issues.")
            print("="*60)
            raise Exception(f"Model loading failed. Please retrain with updated architecture.")
            
    def preprocess_frame(self, frame):
        """
        Preprocess frame for model input
        Args:
            frame: Input BGR image
        Returns:
            Preprocessed image ready for model
        """
        # Resize to model input size
        frame = cv2.resize(frame, (480, 240))
        
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        frame_normalized = frame_rgb / 255.0
        
        # Add batch dimension
        frame_batched = np.expand_dims(frame_normalized, axis=0)
        
        return frame_batched, frame
        
    def predict_steering(self, frame):
        """
        Predict steering angle from frame
        Args:
            frame: Input image
        Returns:
            steering_angle: Predicted steering angle (-1 to 1)
        """
        # Preprocess
        frame_preprocessed, _ = self.preprocess_frame(frame)
        
        # Predict
        prediction = self.model.predict(frame_preprocessed, verbose=0)
        steering_angle = float(prediction[0][0])
        
        # Apply smoothing
        self.prediction_history.append(steering_angle)
        if len(self.prediction_history) > self.smoothing_window:
            self.prediction_history.pop(0)
            
        smoothed_angle = np.mean(self.prediction_history)
        
        return smoothed_angle
        
    def run(self):
        """Main driving loop"""
        print("\n" + "="*50)
        print("Starting Neural Network Autonomous Driving")
        print("="*50)
        print(f"Model: Loaded")
        print(f"Base Speed: {self.base_speed}%")
        print("Press 'q' to quit, SPACE for emergency stop")
        print("="*50 + "\n")
        
        self.is_running = True
        frame_counter = 0
        
        try:
            while self.is_running:
                # Read frame
                success, frame = self.cap.read()
                
                if not success:
                    # If it's a video file, loop it
                    if isinstance(self.video_source, str):
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        frame_counter = 0
                        continue
                    else:
                        print("Error: Could not read frame from camera")
                        break
                
                frame_counter += 1
                self.frame_count += 1
                
                # Predict steering
                steering_angle = self.predict_steering(frame)
                
                # Control motors
                self.motors.set_curve(steering_angle, base_speed=self.base_speed)
                
                # Display
                if self.display_mode > 0:
                    display_frame = cv2.resize(frame, (480, 240))
                    
                    # Draw steering indicator
                    h, w = display_frame.shape[:2]
                    midY = h // 2
                    center_x = w // 2
                    
                    # Steering line
                    end_x = int(center_x + (steering_angle * 150))
                    cv2.line(display_frame, (center_x, midY), 
                            (end_x, midY), (255, 0, 255), 5)
                    cv2.circle(display_frame, (end_x, midY), 10, (0, 255, 0), -1)
                    
                    # Text
                    cv2.putText(display_frame, f"Steering: {steering_angle:.3f}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(display_frame, f"Speed: {self.base_speed}%", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    cv2.imshow('Neural Autonomous Drive', display_frame)
                
                # FPS calculation
                if self.frame_count % 30 == 0:
                    elapsed = time.time() - self.start_time
                    fps = self.frame_count / elapsed
                    print(f"FPS: {fps:.2f} | Steering: {steering_angle:.3f} | Speed: {self.base_speed}%")
                
                # Keyboard controls
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nQuitting...")
                    break
                elif key == ord('+') or key == ord('='):
                    self.base_speed = min(100, self.base_speed + 5)
                    print(f"Speed increased to {self.base_speed}%")
                elif key == ord('-'):
                    self.base_speed = max(0, self.base_speed - 5)
                    print(f"Speed decreased to {self.base_speed}%")
                elif key == ord(' '):
                    print("EMERGENCY STOP")
                    self.motors.stop()
                    time.sleep(1)
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            
        finally:
            self.cleanup()
            
    def cleanup(self):
        """Cleanup resources"""
        print("\nCleaning up...")
        self.is_running = False
        self.motors.cleanup()
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Print statistics
        if self.frame_count > 0:
            elapsed = time.time() - self.start_time
            avg_fps = self.frame_count / elapsed
            print(f"\nSession Statistics:")
            print(f"  Total Frames: {self.frame_count}")
            print(f"  Duration: {elapsed:.2f} seconds")
            print(f"  Average FPS: {avg_fps:.2f}")
        
        print("Cleanup complete")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Neural Network Autonomous Driving')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model (.h5 file)')
    parser.add_argument('--video', type=str, default='vid1.mp4',
                        help='Video source (0 for camera, or path to video file)')
    parser.add_argument('--speed', type=int, default=50,
                        help='Base driving speed (0-100)')
    parser.add_argument('--no-gpio', action='store_true',
                        help='Disable GPIO (simulation mode)')
    parser.add_argument('--display', type=int, default=1, choices=[0, 1],
                        help='Display mode: 0=none, 1=show predictions')
    parser.add_argument('--camera', action='store_true',
                        help='Use Pi camera instead of video file')
    
    args = parser.parse_args()
    
    # Determine video source
    if args.camera:
        video_source = 0
    else:
        video_source = args.video
    
    # Create and run neural autonomous car
    try:
        car = NeuralAutonomousCar(
            model_path=args.model,
            video_source=video_source,
            use_gpio=not args.no_gpio,
            base_speed=args.speed,
            display_mode=args.display
        )
        car.run()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

