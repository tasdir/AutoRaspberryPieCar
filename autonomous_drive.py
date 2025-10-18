"""
Main Autonomous Driving Script for Raspberry Pi Car
Integrates lane detection with motor control for autonomous navigation
"""

import cv2
import numpy as np
import lane_detection_module as lane_module
import motor_control
import utlis
import time
import argparse


class AutonomousCar:
    """Main autonomous car controller"""
    
    def __init__(self, video_source=0, use_gpio=True, base_speed=50, display_mode=1):
        """
        Initialize autonomous car
        Args:
            video_source: 0 for Pi camera, or path to video file
            use_gpio: Enable/disable GPIO control
            base_speed: Base driving speed (0-100)
            display_mode: 0=no display, 1=result only, 2=all debug images
        """
        self.video_source = video_source
        self.base_speed = base_speed
        self.display_mode = display_mode
        self.is_running = False
        
        # Initialize camera/video
        print(f"Initializing camera/video: {video_source}")
        self.cap = cv2.VideoCapture(video_source)
        
        if not self.cap.isOpened():
            raise Exception(f"Could not open video source: {video_source}")
            
        # Initialize motor control
        print(f"Initializing motor control (GPIO: {use_gpio})")
        self.motors = motor_control.MotorControl(use_gpio=use_gpio)
        
        # Initialize trackbars for perspective transform adjustment
        initial_trackbar_vals = [102, 80, 20, 214]
        utlis.initializeTrackbars(initial_trackbar_vals)
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        
        # Safety parameters
        self.max_no_lane_frames = 30  # Stop if no lane detected for this many frames
        self.no_lane_count = 0
        
        # Curve smoothing
        self.curve_history = []
        self.curve_avg_length = 10
        
    def process_frame(self, frame):
        """
        Process a single frame for lane detection
        Args:
            frame: Input image frame
        Returns:
            curve: Detected curve value (-1 to 1)
        """
        # Resize frame for consistent processing
        frame = cv2.resize(frame, (480, 240))
        
        # Get lane curve
        curve = lane_module.getLaneCurve(frame, display=self.display_mode)
        
        return curve
        
    def control_motors(self, curve):
        """
        Control motors based on detected curve
        Args:
            curve: Normalized curve value (-1 to 1)
        """
        # Check if curve is valid
        if curve is None or np.isnan(curve):
            self.no_lane_count += 1
            print(f"Warning: Invalid curve detected ({self.no_lane_count}/{self.max_no_lane_frames})")
            
            if self.no_lane_count >= self.max_no_lane_frames:
                print("ERROR: No lane detected for too long. Stopping for safety.")
                self.motors.stop()
                return False
        else:
            self.no_lane_count = 0
            
        # Apply curve smoothing
        self.curve_history.append(curve)
        if len(self.curve_history) > self.curve_avg_length:
            self.curve_history.pop(0)
        
        smoothed_curve = np.mean(self.curve_history)
        
        # Control motors based on curve
        self.motors.set_curve(smoothed_curve, base_speed=self.base_speed)
        
        return True
        
    def run(self):
        """Main driving loop"""
        print("\n" + "="*50)
        print("Starting Autonomous Driving")
        print("="*50)
        print(f"Base Speed: {self.base_speed}%")
        print(f"Display Mode: {self.display_mode}")
        print("Press 'q' to quit")
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
                
                # Process frame and get curve
                curve = self.process_frame(frame)
                
                # Control motors
                if not self.control_motors(curve):
                    break
                
                # Calculate and display FPS every 30 frames
                if self.frame_count % 30 == 0:
                    elapsed_time = time.time() - self.start_time
                    fps = self.frame_count / elapsed_time
                    print(f"FPS: {fps:.2f} | Curve: {curve:.3f} | Speed: {self.base_speed}%")
                
                # Check for quit command
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
                    # Emergency stop
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
            elapsed_time = time.time() - self.start_time
            avg_fps = self.frame_count / elapsed_time
            print(f"\nSession Statistics:")
            print(f"  Total Frames: {self.frame_count}")
            print(f"  Duration: {elapsed_time:.2f} seconds")
            print(f"  Average FPS: {avg_fps:.2f}")
        
        print("Cleanup complete")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Autonomous Raspberry Pi Car')
    parser.add_argument('--video', type=str, default='vid1.mp4',
                        help='Video source (0 for camera, or path to video file)')
    parser.add_argument('--speed', type=int, default=50,
                        help='Base driving speed (0-100)')
    parser.add_argument('--no-gpio', action='store_true',
                        help='Disable GPIO (simulation mode)')
    parser.add_argument('--display', type=int, default=2, choices=[0, 1, 2],
                        help='Display mode: 0=none, 1=result, 2=debug')
    parser.add_argument('--camera', action='store_true',
                        help='Use Pi camera instead of video file')
    
    args = parser.parse_args()
    
    # Determine video source
    if args.camera:
        video_source = 0
    else:
        video_source = args.video
    
    # Create and run autonomous car
    try:
        car = AutonomousCar(
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

