import gymnasium
import numpy as np
import time
from ultralytics import YOLO

from main import FortniteCapture, RewardDetector, AgentInput

BEST_PT_PATH = r"C:\Users\Sri Nikhil\source\repos\Projects\Fortnite Reinforcement Learning\runs\detect\guard_detector_v1-2\weights\best.pt"

class FortniteEnv(gymnasium.Env):

    def __init__(self):
        super().__init__()
        
        self.action_space = gymnasium.spaces.Discrete(5)

        self.observation_space = gymnasium.spaces.Box(
            low=0, high=1, shape=(5,), dtype=np.float32
        )

        self.capture = FortniteCapture()
        self.capture.start()

        self.just_shot = False
        self.wasted_shot = False

        self.sweep_direction = 1   # 1 = right, -1 = left
        self.steps_without_guard = 0

        self.detector = RewardDetector(self.capture.monitor)
        self.agent = AgentInput()
        self.model = YOLO(BEST_PT_PATH)

        self.last_obs = np.zeros(5, dtype=np.float32)
        self.last_bbox_area = 0.0
        self.step_count = 0
        self.MAX_STEPS = 300
        
        pass


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.just_shot = False
        self.wasted_shot = False

        self.step_count = 0
        self.last_bbox_area = 0.0
        self.steps_without_guard = 0
        self.sweep_direction = 1 

        self.episode_start_time = time.time()

        deadline = time.time() + 10.0

        while time.time() < deadline:
            obs = self._get_obs()
            if obs[4] > 0.5:   # confidence threshold — guard is visible
                break
            time.sleep(0.1)
        else:
            obs = np.zeros(5, dtype=np.float32)

        self.last_obs = obs
        return obs, {}

    def step(self, action):

        self.step_count += 1

        self.agent.focus_window()
        self.just_shot = False
        self.wasted_shot = False  # add this

        # Action mapping
        if action == 0:
            self.agent.turn_left()
        elif action == 1:
            self.agent.turn_right()
        elif action == 2:
            if self.last_obs[4] > 0:  # guard detected — aim at it
                screen_width = self.capture.monitor["width"]
                screen_height = self.capture.monitor["height"]
                x_error = self.last_obs[0] - 0.5
                y_error = self.last_obs[1] - 0.5
                aim_error = abs(x_error) + abs(y_error)
                sensitivity = max(0.5, min(3.0, aim_error * 10.0))
                dx = x_error * screen_width * sensitivity
                dy = y_error * screen_height * sensitivity
                self.agent.aim_at(dx, dy)
            # no else here anymore
        elif action == 3:
            if self.last_obs[4] > 0:
                aim_error = abs(self.last_obs[0] - 0.5) + abs(self.last_obs[1] - 0.5)
                if aim_error < 0.20:
                    self.agent.shoot()
                    self.just_shot = True
                else:
                    # shot selected but not on target — penalize wasted shot
                    self.wasted_shot = True
            else:
                # shot selected but no guard detected
                self.wasted_shot = True
        elif action == 4:
            time.sleep(0.03)

        # Search sweep — runs regardless of action if guard not visible
        if self.last_obs[4] == 0:
            self.steps_without_guard += 1
            
            # gradually increase sweep speed the longer guard is lost
            sweep_amount = min(30 + self.steps_without_guard * 3, 300)
            
            # reverse direction every 30 steps
            if self.steps_without_guard % 30 == 0:
                self.sweep_direction *= -1
            
            self.agent.turn_left(amount=sweep_amount) if self.sweep_direction == -1 else self.agent.turn_right(amount=sweep_amount)
        else:
            self.steps_without_guard = 0  # reset when guard found

        time.sleep(0.03)  # action duration — 20Hz control frequency

        obs = self._get_obs()
        frame = self.capture.get_frame()

        if frame is not None:
            done = self.detector.detect_reward(frame)
        else:
            done = False

        time_elapsed = time.time() - self.episode_start_time
        timed_out = time_elapsed >= 9.0
        truncated = self.step_count >= self.MAX_STEPS or timed_out

        reward = self._compute_reward(obs)

        if done:
            reward += 5.0
        elif timed_out:
            reward -= 2.0

        self.last_obs = obs
        return obs, reward, done, truncated, {}
    
    def _get_obs(self):
        frame = self.capture.get_frame()

        if frame is None:
            return np.zeros(5, dtype=np.float32)

        results = self.model(frame, verbose=False)

        if results and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            best_idx = boxes.conf.argmax().item()
            x_center = boxes.xywhn[best_idx][0].item()
            y_center = boxes.xywhn[best_idx][1].item()
            width    = boxes.xywhn[best_idx][2].item()
            height   = boxes.xywhn[best_idx][3].item()
            conf     = boxes.conf[best_idx].item()
            return np.array([x_center, y_center, width, height, conf], dtype=np.float32)

        return np.zeros(5, dtype=np.float32)
    
    def _compute_reward(self, obs):

        reward = -0.005  # timestep penalty

        width    = obs[2]
        height   = obs[3]
        confidence = obs[4]

        if confidence > 0:
            reward += 0.01  # guard is visible

            # bonus for finding guard after it was lost
            if self.last_obs[4] == 0:
                reward += 0.1  # just reacquired the target

            # penalize sitting on target without shooting
            aim_error_now = abs(self.last_obs[0] - 0.5) + abs(self.last_obs[1] - 0.5)
            if aim_error_now < 0.20 and not self.just_shot:
                reward -= 0.05

            if self.wasted_shot:
                reward -= 0.10

            if self.just_shot:
                aim_error_for_shot = abs(obs[0] - 0.5) + abs(obs[1] - 0.5)
                if aim_error_for_shot < 0.20:
                    reward += 0.25

            current_area = width * height

            if current_area > self.last_bbox_area:
                reward += 0.02
            else:
                reward -= 0.01
            self.last_bbox_area = current_area  # always update

        else:
            reward -= 0.01  # lost sight of guard
            self.last_bbox_area = 0.0  # reset so redetection starts fresh

        return reward
    
    def close(self):
        self.capture.stop()