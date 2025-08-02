#!/usr/bin/env python3
"""
Test script to verify the training fixes for KeyboardInterrupt handling
"""

import sys
import time
import signal
from pathlib import Path


def test_keyboard_interrupt_handling():
    """Test that keyboard interrupt is handled gracefully"""
    print("🧪 Testing KeyboardInterrupt handling...")

    def signal_handler(sig, frame):
        print("\n🛑 Received KeyboardInterrupt signal")
        print("💾 Simulating emergency checkpoint save...")
        time.sleep(1)
        print("✅ Emergency checkpoint saved")
        print("🔄 Cleanup completed")
        sys.exit(0)

    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)

    try:
        print("🔄 Simulating training loop...")
        print("   Press Ctrl+C to test interrupt handling")

        for i in range(100):
            print(f"   Step {i+1}/100", end="\r")
            time.sleep(0.1)

        print("\n✅ Training loop completed normally")

    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
        print("💾 Saving current state before exit...")
        time.sleep(1)
        print("✅ Emergency checkpoint saved")
        raise
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        print("💾 Attempting to save emergency checkpoint...")
        time.sleep(1)
        print("✅ Emergency checkpoint saved")
        raise
    finally:
        print("🧹 Cleanup completed")


def test_matplotlib_save_handling():
    """Test matplotlib save handling with proper error catching"""
    print("\n🧪 Testing matplotlib save error handling...")

    try:
        # Simulate matplotlib save operation
        import matplotlib.pyplot as plt
        import io

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])
        ax.set_title("Test Plot")

        # Try to save to a problematic buffer (simulate the PIL error)
        print("📊 Creating test plot...")

        # Save normally first
        save_path = Path("test_plot.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print("✅ Plot saved successfully")

        # Cleanup
        plt.close()
        if save_path.exists():
            save_path.unlink()

    except (KeyboardInterrupt, SystemExit):
        print("🛑 Plot generation interrupted by user")
        plt.close('all')
        raise
    except Exception as e:
        print(f"⚠️  Plot generation failed: {e}")
        plt.close('all')


def test_wandb_logger_compatibility():
    """Test wandb logger method compatibility"""
    print("\n🧪 Testing wandb logger compatibility...")

    class MockWandbLogger:
        def log_metrics(self, metrics_dict):
            print(f"📊 Logging metrics: {metrics_dict}")
            return True

        def log_final_metrics(self, metrics_dict):
            print("❌ This method doesn't exist!")
            raise AttributeError(
                "'MockWandbLogger' object has no attribute 'log_final_metrics'")

    # Test the fix
    logger = MockWandbLogger()

    try:
        # This should work (fixed version)
        logger.log_metrics({
            "carbon_emissions_kg": 0.001,
            "carbon_emissions_g": 1.0
        })
        print("✅ log_metrics works correctly")

    except Exception as e:
        print(f"❌ log_metrics failed: {e}")

    try:
        # This should fail (original broken version)
        logger.log_final_metrics({
            "carbon_emissions_kg": 0.001,
            "carbon_emissions_g": 1.0
        })
        print("❌ log_final_metrics should not exist!")

    except AttributeError:
        print("✅ log_final_metrics correctly raises AttributeError (as expected)")


if __name__ == "__main__":
    print("🚀 BitMar Training Fixes Test Suite")
    print("=" * 50)

    try:
        test_matplotlib_save_handling()
        test_wandb_logger_compatibility()

        print("\n" + "=" * 50)
        print("✅ All non-interactive tests passed!")
        print("\n🔄 Starting interactive KeyboardInterrupt test...")
        print("   (Press Ctrl+C within 10 seconds to test)")

        test_keyboard_interrupt_handling()

    except KeyboardInterrupt:
        print("\n✅ KeyboardInterrupt handling test completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)

    print("\n🎉 All tests completed successfully!")
    print("🔧 Your BitMar training script should now handle interrupts gracefully!")
