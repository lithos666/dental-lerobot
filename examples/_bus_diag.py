"""Quick bus diagnostic: repeatedly ping all motors to check communication stability."""

import time

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

motors = {f"m{i}": Motor(i, "sts3215", MotorNormMode.RANGE_M100_100) for i in range(1, 7)}
bus = FeetechMotorsBus(port="COM24", motors=motors)
bus.connect(handshake=False)

for trial in range(5):
    found = bus.broadcast_ping()
    print(f"broadcast_ping trial {trial}: found ids = {sorted(found) if found else None}", flush=True)
    time.sleep(0.2)

for trial in range(5):
    try:
        pos = bus.sync_read("Present_Position", normalize=False, num_retry=2)
        print(f"sync_read trial {trial}: OK {pos}", flush=True)
    except Exception as e:
        print(f"sync_read trial {trial}: FAILED - {e}", flush=True)
    time.sleep(0.2)

bus.disconnect(disable_torque=False)
print("done", flush=True)
