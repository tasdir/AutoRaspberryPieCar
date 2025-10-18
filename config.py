"""
Configuration File for Autonomous Raspberry Pi Car
Centralized settings for easy tuning
"""

# ============================================================================
# CAMERA SETTINGS
# ============================================================================

# Video source (0 for Pi Camera, or path to video file)
VIDEO_SOURCE = 'vid1.mp4'

# Frame dimensions
FRAME_WIDTH = 480
FRAME_HEIGHT = 240

# ============================================================================
# LANE DETECTION SETTINGS
# ============================================================================

# Perspective transform points (for warp)
# Format: [Width Top, Height Top, Width Bottom, Height Bottom]
PERSPECTIVE_POINTS = [102, 80, 20, 214]

# HSV color thresholds for white lane detection
HSV_LOWER_WHITE = [80, 0, 0]
HSV_UPPER_WHITE = [255, 160, 255]

# Histogram settings
HISTOGRAM_MIN_PERCENTAGE = 0.5  # For middle point
HISTOGRAM_AVG_PERCENTAGE = 0.9  # For curve calculation
HISTOGRAM_REGION = 4  # Region of interest (1 = full height, 4 = bottom quarter)

# Curve smoothing
CURVE_AVERAGE_LENGTH = 10  # Number of frames to average

# ============================================================================
# MOTOR CONTROL SETTINGS
# ============================================================================

# GPIO Pin Configuration (BCM numbering if GPIO.setmode(GPIO.BCM))
# Using BOARD numbering by default
MOTOR_PINS = {
    'ENA': 33,  # Left motor PWM
    'IN1': 35,  # Left motor direction 1
    'IN2': 37,  # Left motor direction 2
    'ENB': 32,  # Right motor PWM
    'IN3': 36,  # Right motor direction 1
    'IN4': 38   # Right motor direction 2
}

# PWM frequency (Hz)
PWM_FREQUENCY = 1000

# Base driving speed (0-100%)
BASE_SPEED = 50

# Speed limits
MIN_SPEED = 30
MAX_SPEED = 80

# ============================================================================
# AUTONOMOUS DRIVING SETTINGS
# ============================================================================

# Curve dead zone (don't turn if abs(curve) < this value)
CURVE_DEAD_ZONE = 0.05

# Maximum turn ratio (0-1, how much to slow down inner wheel)
MAX_TURN_RATIO = 0.7

# Safety: Maximum frames with no lane detection before stopping
MAX_NO_LANE_FRAMES = 30

# Display mode: 0=none, 1=result only, 2=debug (all images)
DISPLAY_MODE = 2

# ============================================================================
# DATA COLLECTION SETTINGS
# ============================================================================

# Output directory for training data
TRAINING_DATA_DIR = 'training_data'

# Frame skip (save every Nth frame)
DATA_COLLECTION_FRAME_SKIP = 5

# Maximum frames to collect (None = unlimited)
MAX_TRAINING_FRAMES = None

# ============================================================================
# NEURAL NETWORK SETTINGS
# ============================================================================

# Model architecture
MODEL_NAME = 'lane_model'

# Training parameters
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
TEST_SIZE = 0.2  # Fraction of data for testing

# Dropout rate
DROPOUT_RATE = 0.5

# Early stopping patience
EARLY_STOPPING_PATIENCE = 10

# Learning rate reduction
LR_REDUCTION_FACTOR = 0.5
LR_REDUCTION_PATIENCE = 5

# Prediction smoothing window
PREDICTION_SMOOTHING_WINDOW = 5

# ============================================================================
# RASPBERRY PI SPECIFIC SETTINGS
# ============================================================================

# Enable GPIO (set False for testing on non-Pi systems)
USE_GPIO = True

# Camera settings for Pi Camera
PI_CAMERA_RESOLUTION = (640, 480)
PI_CAMERA_FRAMERATE = 30

# ============================================================================
# DEBUG SETTINGS
# ============================================================================

# Enable verbose output
VERBOSE = True

# Save debug images
SAVE_DEBUG_IMAGES = False
DEBUG_IMAGES_DIR = 'debug_images'

# FPS display update interval (frames)
FPS_UPDATE_INTERVAL = 30

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_config_dict():
    """Return all configuration as a dictionary"""
    return {
        'camera': {
            'video_source': VIDEO_SOURCE,
            'frame_width': FRAME_WIDTH,
            'frame_height': FRAME_HEIGHT,
            'pi_camera_resolution': PI_CAMERA_RESOLUTION,
            'pi_camera_framerate': PI_CAMERA_FRAMERATE
        },
        'lane_detection': {
            'perspective_points': PERSPECTIVE_POINTS,
            'hsv_lower_white': HSV_LOWER_WHITE,
            'hsv_upper_white': HSV_UPPER_WHITE,
            'histogram_min_percentage': HISTOGRAM_MIN_PERCENTAGE,
            'histogram_avg_percentage': HISTOGRAM_AVG_PERCENTAGE,
            'histogram_region': HISTOGRAM_REGION,
            'curve_average_length': CURVE_AVERAGE_LENGTH
        },
        'motor_control': {
            'pins': MOTOR_PINS,
            'pwm_frequency': PWM_FREQUENCY,
            'base_speed': BASE_SPEED,
            'min_speed': MIN_SPEED,
            'max_speed': MAX_SPEED
        },
        'autonomous_driving': {
            'curve_dead_zone': CURVE_DEAD_ZONE,
            'max_turn_ratio': MAX_TURN_RATIO,
            'max_no_lane_frames': MAX_NO_LANE_FRAMES,
            'display_mode': DISPLAY_MODE
        },
        'data_collection': {
            'output_dir': TRAINING_DATA_DIR,
            'frame_skip': DATA_COLLECTION_FRAME_SKIP,
            'max_frames': MAX_TRAINING_FRAMES
        },
        'neural_network': {
            'model_name': MODEL_NAME,
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'test_size': TEST_SIZE,
            'dropout_rate': DROPOUT_RATE,
            'early_stopping_patience': EARLY_STOPPING_PATIENCE,
            'lr_reduction_factor': LR_REDUCTION_FACTOR,
            'lr_reduction_patience': LR_REDUCTION_PATIENCE,
            'prediction_smoothing_window': PREDICTION_SMOOTHING_WINDOW
        },
        'raspberry_pi': {
            'use_gpio': USE_GPIO
        },
        'debug': {
            'verbose': VERBOSE,
            'save_debug_images': SAVE_DEBUG_IMAGES,
            'debug_images_dir': DEBUG_IMAGES_DIR,
            'fps_update_interval': FPS_UPDATE_INTERVAL
        }
    }


def print_config():
    """Print current configuration"""
    import json
    config = get_config_dict()
    print("="*60)
    print("CURRENT CONFIGURATION")
    print("="*60)
    print(json.dumps(config, indent=2))
    print("="*60)


if __name__ == '__main__':
    print_config()

