"""
Data Augmentation Module
Expand training dataset through image transformations
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path
import random


class DataAugmentor:
    """Augment training data for neural network"""
    
    def __init__(self, input_dir='training_data', output_dir='training_data_augmented'):
        """
        Initialize augmentor
        Args:
            input_dir: Original training data directory
            output_dir: Directory for augmented data
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, 'images')
        
        # Create output directory
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Load original dataset
        labels_file = os.path.join(input_dir, 'labels.json')
        with open(labels_file, 'r') as f:
            self.original_data = json.load(f)
            
        self.augmented_data = []
        
    def horizontal_flip(self, image, steering):
        """
        Flip image horizontally and negate steering angle
        This doubles your dataset instantly!
        """
        flipped_image = cv2.flip(image, 1)  # 1 = horizontal flip
        flipped_steering = -steering  # Negate steering angle
        return flipped_image, flipped_steering
    
    def adjust_brightness(self, image, factor=None):
        """
        Adjust image brightness randomly
        Args:
            image: Input image
            factor: Brightness factor (0.5-1.5), None for random
        """
        if factor is None:
            factor = 0.5 + np.random.random()  # 0.5 to 1.5
            
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    def add_shadow(self, image):
        """
        Add random shadow to image
        Simulates varying lighting conditions
        """
        height, width = image.shape[:2]
        
        # Random shadow region
        x1, y1 = np.random.randint(0, width), 0
        x2, y2 = np.random.randint(0, width), height
        
        # Create mask
        mask = np.zeros((height, width), dtype=np.uint8)
        vertices = np.array([[x1, y1], [x2, y2], [width, height], [width, 0]], dtype=np.int32)
        cv2.fillPoly(mask, [vertices], 255)
        
        # Apply shadow
        shadow_factor = 0.3 + 0.4 * np.random.random()  # 0.3 to 0.7
        image_shadow = image.copy()
        image_shadow[mask == 255] = (image_shadow[mask == 255] * shadow_factor).astype(np.uint8)
        
        return image_shadow
    
    def shift_image(self, image, steering, shift_range=20):
        """
        Shift image horizontally and adjust steering
        Args:
            shift_range: Maximum horizontal shift in pixels
        """
        height, width = image.shape[:2]
        shift = np.random.randint(-shift_range, shift_range)
        
        # Affine transformation matrix
        M = np.float32([[1, 0, shift], [0, 1, 0]])
        shifted_image = cv2.warpAffine(image, M, (width, height))
        
        # Adjust steering (shift left = steer right to compensate)
        steering_adjustment = -shift / (width / 2) * 0.2  # Scale factor
        adjusted_steering = np.clip(steering + steering_adjustment, -1, 1)
        
        return shifted_image, adjusted_steering
    
    def add_noise(self, image, noise_factor=10):
        """
        Add Gaussian noise to image
        """
        noise = np.random.randn(*image.shape) * noise_factor
        noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
        return noisy_image
    
    def adjust_contrast(self, image, factor=None):
        """
        Adjust image contrast
        """
        if factor is None:
            factor = 0.7 + 0.6 * np.random.random()  # 0.7 to 1.3
            
        mean = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean()
        contrasted = np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)
        return contrasted
    
    def augment_single_image(self, image_path, steering, index, augment_types):
        """
        Apply multiple augmentation techniques to a single image
        """
        input_image_path = os.path.join(self.input_dir, 'images', image_path)
        image = cv2.imread(input_image_path)
        
        if image is None:
            print(f"Warning: Could not read {input_image_path}")
            return []
        
        augmented_samples = []
        
        # Original image
        original_filename = f"aug_{index:05d}_original.jpg"
        cv2.imwrite(os.path.join(self.images_dir, original_filename), image)
        augmented_samples.append({
            'image': original_filename,
            'curve': steering,
            'augmentation': 'original'
        })
        
        # 1. Horizontal Flip (essential for balance)
        if 'flip' in augment_types:
            flipped_img, flipped_steer = self.horizontal_flip(image, steering)
            filename = f"aug_{index:05d}_flip.jpg"
            cv2.imwrite(os.path.join(self.images_dir, filename), flipped_img)
            augmented_samples.append({
                'image': filename,
                'curve': flipped_steer,
                'augmentation': 'horizontal_flip'
            })
        
        # 2. Brightness variations
        if 'brightness' in augment_types:
            for i, factor in enumerate([0.6, 0.8, 1.2, 1.4]):
                bright_img = self.adjust_brightness(image, factor)
                filename = f"aug_{index:05d}_bright_{i}.jpg"
                cv2.imwrite(os.path.join(self.images_dir, filename), bright_img)
                augmented_samples.append({
                    'image': filename,
                    'curve': steering,
                    'augmentation': f'brightness_{factor}'
                })
        
        # 3. Shadow
        if 'shadow' in augment_types:
            for i in range(2):  # 2 random shadows
                shadow_img = self.add_shadow(image)
                filename = f"aug_{index:05d}_shadow_{i}.jpg"
                cv2.imwrite(os.path.join(self.images_dir, filename), shadow_img)
                augmented_samples.append({
                    'image': filename,
                    'curve': steering,
                    'augmentation': f'shadow_{i}'
                })
        
        # 4. Horizontal shift
        if 'shift' in augment_types:
            for i in range(2):  # Left and right shifts
                shift_img, shift_steer = self.shift_image(image, steering)
                filename = f"aug_{index:05d}_shift_{i}.jpg"
                cv2.imwrite(os.path.join(self.images_dir, filename), shift_img)
                augmented_samples.append({
                    'image': filename,
                    'curve': shift_steer,
                    'augmentation': f'shift_{i}'
                })
        
        # 5. Noise
        if 'noise' in augment_types:
            noisy_img = self.add_noise(image)
            filename = f"aug_{index:05d}_noise.jpg"
            cv2.imwrite(os.path.join(self.images_dir, filename), noisy_img)
            augmented_samples.append({
                'image': filename,
                'curve': steering,
                'augmentation': 'noise'
            })
        
        # 6. Contrast
        if 'contrast' in augment_types:
            for i, factor in enumerate([0.8, 1.2]):
                contrast_img = self.adjust_contrast(image, factor)
                filename = f"aug_{index:05d}_contrast_{i}.jpg"
                cv2.imwrite(os.path.join(self.images_dir, filename), contrast_img)
                augmented_samples.append({
                    'image': filename,
                    'curve': steering,
                    'augmentation': f'contrast_{factor}'
                })
        
        # 7. Combined augmentations
        if 'combined' in augment_types:
            # Brightness + Shadow
            combined_img = self.adjust_brightness(image, 0.7 + 0.6 * np.random.random())
            combined_img = self.add_shadow(combined_img)
            filename = f"aug_{index:05d}_combined.jpg"
            cv2.imwrite(os.path.join(self.images_dir, filename), combined_img)
            augmented_samples.append({
                'image': filename,
                'curve': steering,
                'augmentation': 'combined'
            })
        
        return augmented_samples
    
    def augment_dataset(self, augment_types=['flip', 'brightness', 'shadow', 'shift']):
        """
        Augment entire dataset
        Args:
            augment_types: List of augmentation types to apply
        """
        print(f"Augmenting dataset with: {augment_types}")
        print(f"Original samples: {len(self.original_data['data'])}")
        
        for idx, sample in enumerate(self.original_data['data']):
            augmented = self.augment_single_image(
                sample['image'],
                sample['curve'],
                idx,
                augment_types
            )
            self.augmented_data.extend(augmented)
            
            if (idx + 1) % 50 == 0:
                print(f"Processed {idx + 1}/{len(self.original_data['data'])} images")
        
        print(f"\nAugmentation complete!")
        print(f"Original samples: {len(self.original_data['data'])}")
        print(f"Augmented samples: {len(self.augmented_data)}")
        print(f"Expansion factor: {len(self.augmented_data) / len(self.original_data['data']):.2f}x")
        
        # Save augmented dataset
        self.save_dataset()
        
    def save_dataset(self):
        """Save augmented dataset metadata"""
        output_metadata = {
            'original_dataset': self.input_dir,
            'total_samples': len(self.augmented_data),
            'original_samples': len(self.original_data['data']),
            'augmentation_date': str(np.datetime64('now')),
            'image_size': [480, 240],
            'data': self.augmented_data
        }
        
        labels_file = os.path.join(self.output_dir, 'labels.json')
        with open(labels_file, 'w') as f:
            json.dump(output_metadata, f, indent=2)
        
        print(f"\nAugmented dataset saved to: {self.output_dir}")
        print(f"Labels file: {labels_file}")
        
    def analyze_distribution(self):
        """Analyze steering angle distribution"""
        if not self.augmented_data:
            print("No augmented data available. Run augment_dataset() first.")
            return
        
        curves = [d['curve'] for d in self.augmented_data]
        
        print("\n" + "="*60)
        print("AUGMENTED DATASET ANALYSIS")
        print("="*60)
        print(f"Total samples: {len(curves)}")
        print(f"Mean steering: {np.mean(curves):.3f}")
        print(f"Std deviation: {np.std(curves):.3f}")
        print(f"Min: {np.min(curves):.3f}, Max: {np.max(curves):.3f}")
        
        # Distribution
        straight = sum(1 for c in curves if abs(c) < 0.1)
        left = sum(1 for c in curves if c < -0.1)
        right = sum(1 for c in curves if c > 0.1)
        
        print(f"\nDirection distribution:")
        print(f"  Straight: {straight} ({100*straight/len(curves):.1f}%)")
        print(f"  Left: {left} ({100*left/len(curves):.1f}%)")
        print(f"  Right: {right} ({100*right/len(curves):.1f}%)")
        print("="*60)


def main():
    """Main function with different augmentation strategies"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Augment training dataset')
    parser.add_argument('--input', type=str, default='training_data',
                        help='Input directory with original data')
    parser.add_argument('--output', type=str, default='training_data_augmented',
                        help='Output directory for augmented data')
    parser.add_argument('--strategy', type=str, default='aggressive',
                        choices=['minimal', 'moderate', 'aggressive', 'custom'],
                        help='Augmentation strategy')
    
    args = parser.parse_args()
    
    # Define augmentation strategies
    strategies = {
        'minimal': ['flip'],  # 2x increase (244 → 488)
        'moderate': ['flip', 'brightness', 'shadow'],  # ~8x increase (244 → ~1,952)
        'aggressive': ['flip', 'brightness', 'shadow', 'shift', 'noise', 'contrast', 'combined'],  # ~15x increase (244 → ~3,660)
    }
    
    # Create augmentor
    augmentor = DataAugmentor(args.input, args.output)
    
    # Select strategy
    if args.strategy in strategies:
        augment_types = strategies[args.strategy]
        print(f"\nUsing {args.strategy} strategy: {augment_types}")
    else:
        augment_types = ['flip', 'brightness', 'shadow']  # Default
    
    # Augment dataset
    augmentor.augment_dataset(augment_types)
    
    # Analyze results
    augmentor.analyze_distribution()
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print(f"1. Your augmented data is in: {args.output}/")
    print(f"2. Train with augmented data:")
    print(f"   python neural_network_trainer.py --data-dir {args.output} --epochs 50")
    print("="*60)


if __name__ == '__main__':
    main()

