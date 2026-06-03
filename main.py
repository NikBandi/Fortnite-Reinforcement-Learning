import cv2
import numpy as np
import time
import threading
import pygetwindow as gw
import mss
import pydirectinput


# Tune these after running the calibration helper below
REWARD_REGION = {"top": 680, "left": 750, "width": 400, "height": 60}   # "Eliminated" red text region
REWARD_COLOR_HSV_LOW  = np.array([0,   150, 150])   # red hue range low
REWARD_COLOR_HSV_HIGH = np.array([10,  255, 255])   # red hue range high

CAPTURE_FPS = 144
WINDOW_TITLE = "Fortnite"  # partial match

# TODO: Detect Fornite Window and only capture that window

class FortniteCapture():

    def __init__(self):
        self.sct = mss.mss()
        self.frame = None
        self.running = False
        self._lock = threading.Lock()

    def find_window(self):
        wins = gw.getWindowsWithTitle(WINDOW_TITLE)
        if not wins:
            raise RuntimeError(f"No window found matching '{WINDOW_TITLE}'")
        w = wins[0]
        return {"top": w.top, "left": w.left, "width": w.width, "height": w.height}
    
    # start recording the screen
    def start(self):
        self.monitor = self.find_window()
        self.running = True
        self._thread = threading.Thread(target=self.capture_screen, daemon=True)
        self._thread.start()
    
    def capture_screen(self):
        interval = 1.0 / CAPTURE_FPS

        while self.running:  # <-- add this loop
            raw = self.sct.grab(self.monitor)
            frame = np.array(raw)[:, :, :3]
            with self._lock:
                self.frame = frame
            time.sleep(interval)

    def get_frame(self):
        with self._lock:

            if self.frame is not None:
                return self.frame.copy()
            else:
                return None

    def stop(self):
        self.running = False

# TODO: Add a method to detect the "Eliminated" text and return a reward signal

class RewardDetector:

    def __init__(self, monitor_offset):

        self.reward_region = {
            "top":    monitor_offset["top"]  + REWARD_REGION["top"],
            "left":   monitor_offset["left"] + REWARD_REGION["left"],
            "width":  REWARD_REGION["width"],
            "height": REWARD_REGION["height"],
        }


        self.reward_color_hsv_low = REWARD_COLOR_HSV_LOW
        self.reward_color_hsv_high = REWARD_COLOR_HSV_HIGH

        self.last_reward_time = 0
        self.cooldown = 4  # seconds to wait after detecting a reward before looking for another (to avoid duplicates)

    def detect_reward(self, frame):

        now = time.time()
        if now - self.last_reward_time < self.cooldown:
            return False

        y1 = REWARD_REGION["top"]
        y2 = y1 + REWARD_REGION["height"]
        x1 = REWARD_REGION["left"]
        x2 = x1 + REWARD_REGION["width"]
        roi = frame[y1:y2, x1:x2]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, REWARD_COLOR_HSV_LOW, REWARD_COLOR_HSV_HIGH)
        pixel_count = cv2.countNonZero(mask)

        if pixel_count > 200:  # tune this threshold
            self.last_reward_time = now
            return True
        
        return False
    

# TODO: Add a method to send keyboard/mouse inputs to the game window

class AgentInput:

    def focus_window(self):
        wins = gw.getWindowsWithTitle("Fortnite")
        if wins:
            win = wins[0]
            if win.isMinimized:
                win.restore()
            win.activate()
            time.sleep(0.01)
            
    # Non-blocking input sender. Uses pydirectinput for direct hardware input

    def move_forward(self, duration=0.3):
        threading.Thread(target=self._hold_key, args=("w", duration), daemon=True).start()

    def _hold_key(self, key, duration):
        pydirectinput.keyDown(key)
        time.sleep(duration)
        pydirectinput.keyUp(key)

    def turn_left(self, amount=5):
        pydirectinput.moveRel(-amount, 0, relative=True)

    def turn_right(self, amount=5):
        pydirectinput.moveRel(amount, 0, relative=True)

    def aim_at(self, dx, dy):
        pydirectinput.moveRel(int(dx), int(dy), relative=True)

    def shoot(self):
        pydirectinput.click()

""" 
# TODO: Calibration helper to visualize the reward region and tune coordinates before running the main loop

def calibrate():

    # Kill a guard, then run this to see the captured frame and drawf
    # the reward region box so you can tune REWARD_REGION coordinates.
    
    cap = FortniteCapture()
    cap.start()
    time.sleep(1)
    frame = cap.get_frame()
    cap.stop()

    if frame is None:
        print("No frame captured — is Fortnite running?")
        return

    # Draw reward region box
    y1 = REWARD_REGION["top"]
    y2 = y1 + REWARD_REGION["height"]
    x1 = REWARD_REGION["left"]
    x2 = x1 + REWARD_REGION["width"]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow("Calibration — adjust REWARD_REGION to cover red text", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
"""

# Main Loop

def main():
    cap = FortniteCapture()
    cap.start()
    detector = RewardDetector(cap.monitor)
    agent = AgentInput()

    print("Pipeline running. Press Ctrl+C to stop.")
    try:    
        while True:
            frame = cap.get_frame()
            if frame is None:
                continue

            reward = detector.detect_reward(frame)
            if reward:
                print(f"[{time.time():.2f}] REWARD DETECTED — guard eliminated")
                # TODO: pass reward signal to your RL environment step

            # TODO: replace with actual RL policy actions
            agent.move_forward(0.3)
            time.sleep(1.0 / CAPTURE_FPS)

    except KeyboardInterrupt:
        print("Stopping.")
        cap.stop()

if __name__ == "__main__":

    # Run calibrate() first to verify your REWARD_REGION coords (The current are customize for my monitor and window size- there is no guarentee that this will work for you without adjustment),
    # then switch to main() once it's aligned.
    #calibrate()

    main()