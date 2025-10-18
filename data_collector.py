"""
Data Collection Module for Training Dataset
Extracts frames and labels from video for machine learning
"""

import cv2
import numpy as np
import os
import json
from datetime import datetime
import lane_detection_module as lane_module
import utlis


class DataCollector:
    """Collect training data from video files"""
    
    def __init__(self, video_path, output_dir='training_data'):
        """
        Initialize data collector
        Args:
            video_path: Path to input video
            output_dir: Directory to save training data
        """
        self.video_path = video_path
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, 'images')
        self.labels_file = os.path.join(output_dir, 'labels.json')
        
        # Create directories
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise Exception(f"Could not open video: {video_path}")
            
        # Data storage
        self.dataset = []
        
        # Initialize trackbars
        initial_trackbar_vals = [102, 80, 20, 214]
        utlis.initializeTrackbars(initial_trackbar_vals)
        
    def collect_data(self, frame_skip=5, max_frames=None):
        """
        Collect training data from video
        Args:
            frame_skip: Save every Nth frame
            max_frames: Maximum number of frames to process (None = all)
        """
        print(f"Collecting data from: {self.video_path}")
        print(f"Output directory: {self.output_dir}")
        print(f"Frame skip: {frame_skip}")
        print("="*60)
        
        frame_count = 0
        saved_count = 0
        
        try:
            while True:
                success, frame = self.cap.read()
                
                if not success:
                    print("\nEnd of video reached")
                    break
                    
                frame_count += 1
                
                # Skip frames
                if frame_count % frame_skip != 0:
                    continue
                
                # Check max frames limit
                if max_frames and saved_count >= max_frames:
                    print(f"\nReached maximum frames limit: {max_frames}")
                    break
                
                # Process frame
                frame_resized = cv2.resize(frame, (480, 240))
                
                # Get lane curve (label)
                curve = lane_module.getLaneCurve(frame_resized, display=2)
                
                # Save image
                image_filename = f"frame_{saved_count:05d}.jpg"
                image_path = os.path.join(self.images_dir, image_filename)
                cv2.imwrite(image_path, frame_resized)
                
                # Store data
                data_entry = {
                    'image': image_filename,
                    'curve': float(curve),
                    'frame_number': frame_count,
                    'timestamp': saved_count
                }
                self.dataset.append(data_entry)
                
                saved_count += 1
                
                # Progress update
                if saved_count % 10 == 0:
                    print(f"Processed: {saved_count} frames | Current curve: {curve:.3f}")
                
                # Allow user to quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nCollection stopped by user")
                    break
                    
        except KeyboardInterrupt:
            print("\nCollection interrupted by user")
            
        finally:
            self.save_dataset()
            self.cleanup()
            
        print(f"\nCollection complete!")
        print(f"Total frames saved: {saved_count}")
        print(f"Dataset saved to: {self.labels_file}")
        
    def save_dataset(self):
        """Save dataset labels to JSON file"""
        dataset_info = {
            'video_source': self.video_path,
            'total_samples': len(self.dataset),
            'collection_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'image_size': [480, 240],
            'data': self.dataset
        }
        
        with open(self.labels_file, 'w') as f:
            json.dump(dataset_info, f, indent=2)
            
        print(f"\nDataset metadata saved: {self.labels_file}")
        
    def cleanup(self):
        """Cleanup resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        
    def analyze_dataset(self):
        """Analyze collected dataset statistics"""
        if not self.dataset:
            print("No data collected yet")
            return
            
        curves = [d['curve'] for d in self.dataset]
        
        print("\n" + "="*60)
        print("Dataset Analysis")
        print("="*60)
        print(f"Total samples: {len(curves)}")
        print(f"Curve range: [{min(curves):.3f}, {max(curves):.3f}]")
        print(f"Mean curve: {np.mean(curves):.3f}")
        print(f"Std deviation: {np.std(curves):.3f}")
        print(f"Median curve: {np.median(curves):.3f}")
        
        # Distribution analysis
        straight = sum(1 for c in curves if abs(c) < 0.1)
        left = sum(1 for c in curves if c < -0.1)
        right = sum(1 for c in curves if c > 0.1)
        
        print(f"\nDirection distribution:")
        print(f"  Straight: {straight} ({100*straight/len(curves):.1f}%)")
        print(f"  Left turns: {left} ({100*left/len(curves):.1f}%)")
        print(f"  Right turns: {right} ({100*right/len(curves):.1f}%)")
        print("="*60)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect training data from video')
    parser.add_argument('--video', type=str, default='vid1.mp4',
                        help='Input video file')
    parser.add_argument('--output', type=str, default='training_data',
                        help='Output directory for training data')
    parser.add_argument('--skip', type=int, default=5,
                        help='Save every Nth frame')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Maximum number of frames to collect')
    
    args = parser.parse_args()
    
    # Create collector
    collector = DataCollector(args.video, args.output)
    
    # Collect data
    collector.collect_data(frame_skip=args.skip, max_frames=args.max_frames)
    
    # Analyze dataset
    collector.analyze_dataset()


if __name__ == '__main__':
    main()

