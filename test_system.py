"""
System Test Script
Quick diagnostic for all components
"""

import sys
import os

def test_imports():
    """Test if all required packages are installed"""
    print("\n" + "="*60)
    print("TESTING PACKAGE IMPORTS")
    print("="*60)
    
    packages = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
    }
    
    optional_packages = {
        'tensorflow': 'tensorflow',
        'sklearn': 'scikit-learn',
        'matplotlib': 'matplotlib',
        'RPi.GPIO': 'RPi.GPIO (Raspberry Pi only)'
    }
    
    all_good = True
    
    # Test required packages
    for package, install_name in packages.items():
        try:
            __import__(package)
            print(f"[OK]  {package:20s} - OK")
        except ImportError:
            print(f"[FAIL] {package:20s} - MISSING (install: {install_name})")
            all_good = False
    
    # Test optional packages
    print("\nOptional packages:")
    for package, install_name in optional_packages.items():
        try:
            __import__(package)
            print(f"[OK]  {package:20s} - OK")
        except ImportError:
            print(f"[SKIP] {package:20s} - Not installed ({install_name})")
    
    return all_good


def test_files():
    """Test if all required files exist"""
    print("\n" + "="*60)
    print("TESTING PROJECT FILES")
    print("="*60)
    
    required_files = [
        'lane_detection_module.py',
        'utlis.py',
        'motor_control.py',
        'autonomous_drive.py',
        'data_collector.py',
        'neural_network_trainer.py',
        'neural_drive.py',
        'config.py',
        'requirements.txt',
        'README.md'
    ]
    
    all_good = True
    for file in required_files:
        if os.path.exists(file):
            print(f"[OK]   {file}")
        else:
            print(f"[FAIL] {file} - MISSING")
            all_good = False
    
    return all_good


def test_video():
    """Test if video file exists and can be opened"""
    print("\n" + "="*60)
    print("TESTING VIDEO FILE")
    print("="*60)
    
    try:
        import cv2
        
        if os.path.exists('vid1.mp4'):
            cap = cv2.VideoCapture('vid1.mp4')
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    height, width = frame.shape[:2]
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    print(f"[OK]   vid1.mp4 exists and is readable")
                    print(f"  Resolution: {width}x{height}")
                    print(f"  FPS: {fps}")
                    print(f"  Frames: {int(frames)}")
                    print(f"  Duration: {frames/fps:.1f} seconds")
                    cap.release()
                    return True
                else:
                    print("[FAIL] vid1.mp4 exists but cannot read frames")
                    return False
            else:
                print("[FAIL] vid1.mp4 exists but cannot be opened")
                return False
        else:
            print("[SKIP] vid1.mp4 not found (optional for testing)")
            return True
    except Exception as e:
        print(f"[FAIL] Error testing video: {e}")
        return False


def test_camera():
    """Test camera availability"""
    print("\n" + "="*60)
    print("TESTING CAMERA")
    print("="*60)
    
    try:
        import cv2
        
        # Try camera index 0
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                height, width = frame.shape[:2]
                print(f"[OK]   Camera detected (index 0)")
                print(f"  Resolution: {width}x{height}")
                cap.release()
                return True
            else:
                print("[SKIP] Camera opened but cannot read frames")
                print("  (This is OK if testing without camera)")
                cap.release()
                return True
        else:
            print("[SKIP] No camera detected at index 0")
            print("  (This is OK if testing without camera)")
            return True
    except Exception as e:
        print(f"[SKIP] Camera test skipped: {e}")
        return True


def test_gpio():
    """Test GPIO availability (Raspberry Pi only)"""
    print("\n" + "="*60)
    print("TESTING GPIO")
    print("="*60)
    
    try:
        import RPi.GPIO as GPIO
        GPIO.setmode(GPIO.BOARD)
        print("[OK]   RPi.GPIO available")
        print("  GPIO mode: BOARD")
        GPIO.cleanup()
        return True
    except ImportError:
        print("[SKIP] RPi.GPIO not available")
        print("  (This is OK if not running on Raspberry Pi)")
        return True
    except Exception as e:
        print(f"[SKIP] GPIO test skipped: {e}")
        return True


def test_modules():
    """Test if custom modules can be imported"""
    print("\n" + "="*60)
    print("TESTING CUSTOM MODULES")
    print("="*60)
    
    modules = [
        'lane_detection_module',
        'utlis',
        'motor_control',
        'config'
    ]
    
    all_good = True
    for module in modules:
        try:
            __import__(module)
            print(f"[OK]   {module}")
        except Exception as e:
            print(f"[FAIL] {module} - Error: {e}")
            all_good = False
    
    return all_good


def test_config():
    """Test configuration"""
    print("\n" + "="*60)
    print("TESTING CONFIGURATION")
    print("="*60)
    
    try:
        import config
        
        print(f"[OK]   Configuration loaded")
        print(f"  Base Speed: {config.BASE_SPEED}%")
        print(f"  Frame Size: {config.FRAME_WIDTH}x{config.FRAME_HEIGHT}")
        print(f"  Use GPIO: {config.USE_GPIO}")
        print(f"  Display Mode: {config.DISPLAY_MODE}")
        
        return True
    except Exception as e:
        print(f"[FAIL] Error loading config: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("\n")
    print("="*60)
    print("AUTONOMOUS RASPBERRY PI CAR - SYSTEM TEST")
    print("="*60)
    
    tests = [
        ("Package Imports", test_imports),
        ("Project Files", test_files),
        ("Video File", test_video),
        ("Camera", test_camera),
        ("GPIO", test_gpio),
        ("Custom Modules", test_modules),
        ("Configuration", test_config)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\nError in {name}: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status:8s} - {name}")
    
    print("="*60)
    print(f"Results: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("\n*** All tests passed! System is ready. ***")
        print("\nNext steps:")
        print("1. Test lane detection: python lane_detection_module.py")
        print("2. Test autonomous drive: python autonomous_drive.py --video vid1.mp4 --no-gpio")
        print("3. See QUICK_START.md for more information")
    else:
        print("\n*** WARNING: Some tests failed. Please resolve issues before proceeding. ***")
        print("See README.md for installation instructions.")
    
    return passed == total


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)

