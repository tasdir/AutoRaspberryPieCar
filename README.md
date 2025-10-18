# Autonomous Raspberry Pi Car

A complete autonomous driving system for Raspberry Pi using computer vision and optional deep learning for lane detection and navigation.

##  Features

- **Lane Detection**: Computer vision-based lane detection using OpenCV
- **Motor Control**: Full motor control system for Raspberry Pi GPIO
- **Autonomous Driving**: Integrated system for autonomous navigation
- **Data Collection**: Tools to create training datasets from video
- **Neural Network**: Optional CNN-based approach using TensorFlow
- **Flexible Configuration**: Easy-to-modify configuration file
- **Simulation Mode**: Test without Raspberry Pi hardware

## Project Structure

```
AutoRaspberryCar/
├── lane_detection_module.py    # Core lane detection using CV
├── utlis.py                     # Utility functions for image processing
├── motor_control.py             # Motor control for Raspberry Pi GPIO
├── autonomous_drive.py          # Main autonomous driving script (CV-based)
├── data_collector.py            # Collect training data from videos
├── neural_network_trainer.py   # Train CNN model
├── neural_drive.py              # Autonomous driving using neural network
├── config.py                    # Configuration settings
├── requirements.txt             # Python dependencies
├── vid1.mp4                     # Sample video for testing/training
└── README.md                    # This file
```

## Hardware Requirements

### Raspberry Pi Setup
- Raspberry Pi 3/4 (recommended)
- Pi Camera Module or USB Camera
- L298N Motor Driver (or similar H-bridge)
- DC Motors (2x for differential drive)
- Power supply for motors (7.4V - 12V recommended)
- Chassis and wheels

### Pin Connections (Default)
```
Motor Driver → Raspberry Pi (BOARD numbering)
ENA (Left PWM)     → Pin 33
IN1 (Left Dir 1)   → Pin 35
IN2 (Left Dir 2)   → Pin 37
ENB (Right PWM)    → Pin 32
IN3 (Right Dir 1)  → Pin 36
IN4 (Right Dir 2)  → Pin 38
GND                → GND
```

**Note**: Pin configuration can be changed in `config.py`

## Installation

### 1. Clone the Repository
```bash
cd ~
git clone <your-repo-url>
cd AutoRaspberryCar
```

### 2. Install Dependencies

**On Raspberry Pi:**
```bash
# Update system
sudo apt-get update
sudo apt-get upgrade

# Install system dependencies
sudo apt-get install python3-opencv python3-pip

# Install Python packages
pip3 install -r requirements.txt

# Install RPi.GPIO (on Raspberry Pi only)
pip3 install RPi.GPIO
```

**On Development Machine (for testing/training):**
```bash
pip install -r requirements.txt
```

### 3. Verify Installation
```bash
python3 motor_control.py  # Test motor control (simulation mode)
python3 config.py          # View configuration
```

## Usage

### Method 1: Computer Vision-Based (No Training Required)

#### Test with Video File
```bash
# Run with video file and no GPIO (simulation)
python3 autonomous_drive.py --video vid1.mp4 --no-gpio --display 2

# Adjust speed
python3 autonomous_drive.py --video vid1.mp4 --no-gpio --speed 60
```

#### Run on Raspberry Pi
```bash
# With Pi Camera
python3 autonomous_drive.py --camera --speed 50 --display 1

# With USB Camera
python3 autonomous_drive.py --video 0 --speed 50
```

**Keyboard Controls:**
- `q` - Quit
- `+/=` - Increase speed
- `-` - Decrease speed
- `SPACE` - Emergency stop

### Method 2: Neural Network-Based (Requires Training)

#### Step 1: Collect Training Data
```bash
# Collect data from your video
python3 data_collector.py --video vid1.mp4 --output training_data --skip 5

# This will create:
# - training_data/images/ (extracted frames)
# - training_data/labels.json (steering labels)
```

#### Step 2: Train Neural Network
```bash
# Train the model
python3 neural_network_trainer.py \
    --data-dir training_data \
    --model-name lane_model \
    --epochs 50 \
    --batch-size 32

# This will create:
# - lane_model.h5 (trained model)
# - lane_model_best.h5 (best model checkpoint)
# - training_history.png (training plots)
# - predictions.png (prediction analysis)
```

#### Step 3: Run Autonomous Driving with Neural Network
```bash
# Test with video
python3 neural_drive.py --model lane_model.h5 --video vid1.mp4 --no-gpio

# Run on Raspberry Pi with camera
python3 neural_drive.py --model lane_model.h5 --camera --speed 50
```

## Configguration

Edit `config.py` to customize:

- **Camera settings**: Resolution, frame rate
- **Lane detection**: HSV thresholds, perspective transform points
- **Motor control**: Pin assignments, PWM frequency, speed limits
- **Neural network**: Model architecture, training parameters
- **Safety**: Maximum no-lane detection frames

```python
# Example: Change base speed
BASE_SPEED = 60  # Change from 50 to 60

# Example: Adjust GPIO pins
MOTOR_PINS = {
    'ENA': 33,  # Change according to your wiring
    'IN1': 35,
    # ... etc
}
```

## Calibration

### 1. Adjust Perspective Transform

Run the lane detection with trackbars to adjust the perspective:

```bash
python3 lane_detection_module.py
```

Use the trackbars to set the four points for perspective transform:
- Width Top: Inner width at top
- Height Top: Top position
- Width Bottom: Inner width at bottom
- Height Bottom: Bottom position

Update these values in `config.py` under `PERSPECTIVE_POINTS`.

### 2. Tune HSV Color Thresholds

For different lighting conditions, adjust HSV thresholds in `config.py`:
```python
HSV_LOWER_WHITE = [80, 0, 0]    # Adjust for your conditions
HSV_UPPER_WHITE = [255, 160, 255]
```

### 3. Motor Calibration

If your car drifts to one side, you may need to:
1. Check motor connections
2. Adjust individual motor speeds in `motor_control.py`
3. Calibrate wheel alignment

## Performance Tricks

### For Better Lane Detection:
- Ensure good lighting conditions
- Use high-contrast lane markings
- Calibrate perspective transform for your camera angle
- Adjust HSV thresholds for your environment

### For Neural Network Training:
- Collect diverse training data (various lighting, curves) - Here I only used small dataset, which might not be good for in real world implications.
- Use data augmentation (flip images for left/right balance)
- Train for more epochs if underfitting
- Reduce learning rate if training is unstable

### For Raspberry Pi Performance:
- Use Raspberry Pi 4 for better performance
- Reduce frame resolution if FPS is too low
- Close unnecessary applications
- Consider using TensorFlow Lite for faster inference

## Troubleshooting

### Camera Issues
```bash
# Test camera
python3 -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera Error')"

# For Pi Camera, enable in raspi-config
sudo raspi-config
# Interface Options → Camera → Enable
```

### GPIO Permissions
```bash
# Add user to gpio group
sudo usermod -a -G gpio $USER
# Logout and login again
```

### Import Errors
```bash
# Reinstall OpenCV
pip3 install --upgrade opencv-python

# For TensorFlow on Raspberry Pi
pip3 install tensorflow-aarch64  # For 64-bit Pi OS
```

### Motor Not Moving
1. Check power supply to motor driver
2. Verify GPIO connections
3. Test with `motor_control.py` in simulation mode first
4. Check motor driver enable pins

## Example Workflow

### Complete Pipeline:

1. **Record Road Video**
   ```bash
   # Use your phone or Pi camera to record road footage
   # Save as vid1.mp4
   ```

2. **Test Lane Detection**
   ```bash
   python3 lane_detection_module.py
   # Adjust trackbars until lanes are detected well
   ```

3. **Collect Training Data**
   ```bash
   python3 data_collector.py --video vid1.mp4 --skip 3
   ```

4. **Train Neural Network** (Optional)
   ```bash
   python3 neural_network_trainer.py --epochs 100
   ```

5. **Test Autonomous Driving**
   ```bash
   # CV-based approach
   python3 autonomous_drive.py --video vid1.mp4 --no-gpio
   
   # Or NN-based approach
   python3 neural_drive.py --model lane_model.h5 --video vid1.mp4 --no-gpio
   ```

6. **Deploy on Raspberry Pi**
   ```bash
   # Transfer code to Raspberry Pi
   # Install dependencies
   # Run with real camera and motors
   python3 autonomous_drive.py --camera
   ```

##  Safety Considerations

 **IMPORTANT SAFETY NOTES:**

1. **Always test in a safe, controlled environment**
2. **Start with low speeds and gradually increase**
3. **Have an emergency stop mechanism ready**
4. **Never test near people, animals, or obstacles**
5. **Monitor the car at all times during autonomous operation**
6. **Ensure proper power supply to prevent brownouts**
7. **Use fuses to protect electronics**
8. **Be ready to manually take control**

##  Advanced Features

### Data Augmentation
Modify `neural_network_trainer.py` to add:
- Image flipping (horizontal)
- Brightness adjustment
- Shadow simulation
- Perspective shifts

### Multi-Camera Setup
Extend `autonomous_drive.py` to use multiple cameras for better depth perception.

### Sensor Fusion
Add ultrasonic sensors or LIDAR for obstacle avoidance.

### PID Control
Implement PID controller for smoother steering in `motor_control.py`.




