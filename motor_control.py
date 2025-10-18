"""
Motor Control Module for Raspberry Pi Autonomous Car
Handles motor control using L298N motor driver or similar H-bridge
"""

class MotorControl:
    """
    Motor control class for Raspberry Pi GPIO
    Compatible with L298N motor driver
    """
    
    def __init__(self, use_gpio=True):
        """
        Initialize motor control
        Args:
            use_gpio: Set to False for testing without GPIO (simulation mode)
        """
        self.use_gpio = use_gpio
        self.current_speed = 0
        self.current_direction = 0
        
        if self.use_gpio:
            try:
                import RPi.GPIO as GPIO
                self.GPIO = GPIO
                self.setup_gpio()
            except ImportError:
                print("Warning: RPi.GPIO not found. Running in simulation mode.")
                self.use_gpio = False
        
        # Motor driver pins configuration
        # Left Motor
        self.ENA = 33  # Left motor speed control (PWM)
        self.IN1 = 35  # Left motor direction 1
        self.IN2 = 37  # Left motor direction 2
        
        # Right Motor
        self.ENB = 32  # Right motor speed control (PWM)
        self.IN3 = 36  # Right motor direction 1
        self.IN4 = 38  # Right motor direction 2
        
        # PWM frequency
        self.pwm_frequency = 1000
        
    def setup_gpio(self):
        """Setup GPIO pins for motor control"""
        self.GPIO.setmode(self.GPIO.BOARD)
        self.GPIO.setwarnings(False)
        
        # Setup motor pins
        self.GPIO.setup(self.ENA, self.GPIO.OUT)
        self.GPIO.setup(self.IN1, self.GPIO.OUT)
        self.GPIO.setup(self.IN2, self.GPIO.OUT)
        self.GPIO.setup(self.ENB, self.GPIO.OUT)
        self.GPIO.setup(self.IN3, self.GPIO.OUT)
        self.GPIO.setup(self.IN4, self.GPIO.OUT)
        
        # Setup PWM for speed control
        self.pwm_left = self.GPIO.PWM(self.ENA, self.pwm_frequency)
        self.pwm_right = self.GPIO.PWM(self.ENB, self.pwm_frequency)
        
        # Start PWM with 0% duty cycle
        self.pwm_left.start(0)
        self.pwm_right.start(0)
        
    def move_forward(self, speed=50):
        """
        Move car forward
        Args:
            speed: Speed percentage (0-100)
        """
        speed = max(0, min(100, speed))  # Clamp between 0-100
        
        if self.use_gpio:
            # Left motor forward
            self.GPIO.output(self.IN1, self.GPIO.HIGH)
            self.GPIO.output(self.IN2, self.GPIO.LOW)
            self.pwm_left.ChangeDutyCycle(speed)
            
            # Right motor forward
            self.GPIO.output(self.IN3, self.GPIO.HIGH)
            self.GPIO.output(self.IN4, self.GPIO.LOW)
            self.pwm_right.ChangeDutyCycle(speed)
        else:
            print(f"[SIM] Moving forward at speed: {speed}%")
            
        self.current_speed = speed
        self.current_direction = 0
        
    def move_backward(self, speed=50):
        """
        Move car backward
        Args:
            speed: Speed percentage (0-100)
        """
        speed = max(0, min(100, speed))
        
        if self.use_gpio:
            # Left motor backward
            self.GPIO.output(self.IN1, self.GPIO.LOW)
            self.GPIO.output(self.IN2, self.GPIO.HIGH)
            self.pwm_left.ChangeDutyCycle(speed)
            
            # Right motor backward
            self.GPIO.output(self.IN3, self.GPIO.LOW)
            self.GPIO.output(self.IN4, self.GPIO.HIGH)
            self.pwm_right.ChangeDutyCycle(speed)
        else:
            print(f"[SIM] Moving backward at speed: {speed}%")
            
        self.current_speed = speed
        
    def turn_left(self, speed=50, turn_ratio=0.5):
        """
        Turn left while moving forward
        Args:
            speed: Base speed percentage (0-100)
            turn_ratio: How much to reduce left motor speed (0-1)
        """
        speed = max(0, min(100, speed))
        left_speed = speed * (1 - turn_ratio)
        
        if self.use_gpio:
            # Left motor slower
            self.GPIO.output(self.IN1, self.GPIO.HIGH)
            self.GPIO.output(self.IN2, self.GPIO.LOW)
            self.pwm_left.ChangeDutyCycle(left_speed)
            
            # Right motor normal
            self.GPIO.output(self.IN3, self.GPIO.HIGH)
            self.GPIO.output(self.IN4, self.GPIO.LOW)
            self.pwm_right.ChangeDutyCycle(speed)
        else:
            print(f"[SIM] Turning left - Left: {left_speed}%, Right: {speed}%")
            
        self.current_speed = speed
        self.current_direction = -turn_ratio
        
    def turn_right(self, speed=50, turn_ratio=0.5):
        """
        Turn right while moving forward
        Args:
            speed: Base speed percentage (0-100)
            turn_ratio: How much to reduce right motor speed (0-1)
        """
        speed = max(0, min(100, speed))
        right_speed = speed * (1 - turn_ratio)
        
        if self.use_gpio:
            # Left motor normal
            self.GPIO.output(self.IN1, self.GPIO.HIGH)
            self.GPIO.output(self.IN2, self.GPIO.LOW)
            self.pwm_left.ChangeDutyCycle(speed)
            
            # Right motor slower
            self.GPIO.output(self.IN3, self.GPIO.HIGH)
            self.GPIO.output(self.IN4, self.GPIO.LOW)
            self.pwm_right.ChangeDutyCycle(right_speed)
        else:
            print(f"[SIM] Turning right - Left: {speed}%, Right: {right_speed}%")
            
        self.current_speed = speed
        self.current_direction = turn_ratio
        
    def stop(self):
        """Stop all motors"""
        if self.use_gpio:
            self.pwm_left.ChangeDutyCycle(0)
            self.pwm_right.ChangeDutyCycle(0)
            self.GPIO.output(self.IN1, self.GPIO.LOW)
            self.GPIO.output(self.IN2, self.GPIO.LOW)
            self.GPIO.output(self.IN3, self.GPIO.LOW)
            self.GPIO.output(self.IN4, self.GPIO.LOW)
        else:
            print("[SIM] Stopping motors")
            
        self.current_speed = 0
        self.current_direction = 0
        
    def set_curve(self, curve_value, base_speed=50):
        """
        Set motor speeds based on curve value from lane detection
        Args:
            curve_value: Normalized curve value (-1 to 1)
                        -1 = sharp left, 0 = straight, 1 = sharp right
            base_speed: Base speed for the car (0-100)
        """
        curve_value = max(-1, min(1, curve_value))  # Clamp to [-1, 1]
        
        if abs(curve_value) < 0.05:  # Dead zone for small curves
            self.move_forward(base_speed)
        elif curve_value < 0:  # Turn left
            turn_ratio = abs(curve_value)
            self.turn_left(base_speed, turn_ratio)
        else:  # Turn right
            turn_ratio = curve_value
            self.turn_right(base_speed, turn_ratio)
            
    def cleanup(self):
        """Cleanup GPIO on exit"""
        if self.use_gpio:
            self.stop()
            self.pwm_left.stop()
            self.pwm_right.stop()
            self.GPIO.cleanup()
            print("GPIO cleaned up")
        else:
            print("[SIM] Cleanup completed")


# Test code
if __name__ == "__main__":
    import time
    
    print("Testing Motor Control Module")
    print("Running in simulation mode (no GPIO)")
    
    motors = MotorControl(use_gpio=False)
    
    try:
        print("\n1. Moving forward...")
        motors.move_forward(speed=60)
        time.sleep(2)
        
        print("\n2. Turning left...")
        motors.turn_left(speed=60, turn_ratio=0.5)
        time.sleep(2)
        
        print("\n3. Turning right...")
        motors.turn_right(speed=60, turn_ratio=0.5)
        time.sleep(2)
        
        print("\n4. Testing curve-based control...")
        print("   Slight left curve:")
        motors.set_curve(-0.3, base_speed=60)
        time.sleep(2)
        
        print("   Sharp right curve:")
        motors.set_curve(0.8, base_speed=60)
        time.sleep(2)
        
        print("\n5. Stopping...")
        motors.stop()
        
    except KeyboardInterrupt:
        print("\nTest interrupted")
    finally:
        motors.cleanup()
        print("Test completed")

