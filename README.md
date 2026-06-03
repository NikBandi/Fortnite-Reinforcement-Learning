# Fortnite Reinforcement Learning Agent

A reinforcement learning agent that learns to eliminate guards in Fortnite using **Proximal Policy Optimization (PPO)**. The agent sees the world through a custom-trained YOLOv8 model, computes rewards from real-time screen analysis, and controls the game entirely through hardware-level mouse and keyboard input — no game memory reading, no APIs, just vision and input.

---

## How It Works

The agent operates on a remarkably small observation space: a single 5-value vector from YOLO's bounding box output (`x_center`, `y_center`, `width`, `height`, `confidence`). From that alone, it learns to search for guards, track them, close distance, aim, and shoot.

Training runs entirely in real-time against a live Fortnite session. Each episode ends either when the agent eliminates a guard (+5.0 reward) or when the 9-second timeout fires (-2.0 reward). Everything in between is shaped through a dense reward function that rewards visibility, approach, and accurate shots while penalizing wasted inputs and lost targets.

---

## Architecture

| File | Purpose |
|------|---------|
| `env.py` | Gymnasium environment — observation, action, reward |
| `model.py` | ActorCritic network + PPO trainer |
| `main.py` | Screen capture, reward detection, input sending |
| `train.py` | Entry point — instantiates and runs training |

### Network

A shared-trunk ActorCritic with two heads:

- **Input:** 5 values (YOLO bounding box)
- **Shared trunk:** `Linear(5→64) → Tanh → Linear(64→64) → Tanh`
- **Policy head:** `Linear(64→5)` — logits over 5 discrete actions
- **Value head:** `Linear(64→1)` — scalar state value V(s)

### Action Space

| Action | Behavior |
|--------|---------|
| 0 | Turn left |
| 1 | Turn right |
| 2 | Aim at guard (proportional mouse move toward bbox center) |
| 3 | Shoot (only fires if guard detected and aim error < 0.20) |
| 4 | No-op |

---

## Reward Function

| Event | Reward |
|-------|--------|
| Per timestep | -0.005 |
| Guard visible | +0.01 |
| Guard reacquired after loss | +0.10 |
| Aimed on target without shooting | -0.05 |
| Shot while on target | +0.25 |
| Wasted shot (off target or no guard) | -0.10 |
| Bounding box area growing | +0.02 |
| Bounding box area shrinking | -0.01 |
| Guard lost from view | -0.01 |
| Guard eliminated | +5.0 |
| Episode timeout (9 seconds) | -2.0 |

---

## PPO Hyperparameters

| Parameter | Value |
|-----------|-------|
| n_steps | 1024 |
| n_epochs | 10 |
| batch_size | 64 |
| gamma | 0.99 |
| gae_lambda | 0.95 |
| clip_epsilon | 0.2 |
| value loss coeff (c1) | 0.5 |
| entropy coeff (c2) | 0.03 |
| learning rate | 3e-4 |
| episode timeout | 9 seconds |
| total training steps | 20,000 |

---

## Requirements

**Python 3.10+** is recommended. Install dependencies with:

```bash
pip install torch ultralytics gymnasium opencv-python mss pygetwindow pydirectinput tensorboard numpy
```

You'll also need:
- A trained YOLOv8 guard detector (`best.pt`) — not included in this repo
- Fortnite running in a Creative mode map with guard NPCs
- Windows OS (pydirectinput is Windows-only)

---

## Setup

1. Clone the repo:
```bash
git clone https://github.com/NikBandi/Fortnite-Reinforcement-Learning.git
cd Fortnite-Reinforcement-Learning
```

2. Update the YOLO weights path in `env.py`:
```python
BEST_PT_PATH = r"path\to\your\weights\best.pt"
```

3. Calibrate the reward detection region in `main.py` by running:
```python
# In main.py, switch the bottom block to:
calibrate()
```
This opens a window showing the capture frame with the reward region drawn on it. Adjust `REWARD_REGION` coordinates until the box covers the red "Eliminated" text on your screen.

4. Once calibrated, switch back to `main()` and you're ready to train.

---

## Training

Make sure Fortnite is fully loaded into your map with guards present before starting. Then run:

```bash
python train.py
```

The agent will start collecting rollouts immediately. Checkpoints are saved to `model/` every 5 rollouts.

To monitor training in real-time:
```bash
tensorboard --logdir runs/ppo_fortnite
# Open http://localhost:6006
```

**What to watch:** `train/mean_reward` trending upward is the signal that matters. Loss going negative is normal — the entropy term pulls it there.

### Typical Progression

| Rollouts | Expected Behavior |
|----------|------------------|
| 1–5 | Random exploration, reward near 0 |
| 6–10 | Guard tracking starts to emerge |
| 15–20 | Aim → shoot sequence appearing |
| 30+ | Consistent eliminations |

### Resuming from a Checkpoint

```python
# In train.py, before trainer.train():
model.load('checkpoint_XXXXX_XXXXXXXXXX.pth')
```

> **Note:** Only load checkpoints saved with the same reward function. Loading an old checkpoint after changing rewards will cause the policy to behave incorrectly.

---

## Project Notes

A few design decisions worth knowing before you dig in:

**The search sweep is hardcoded, not learned.** When the guard leaves frame, the camera automatically sweeps left and right with increasing speed. This runs independently of the policy — it's a practical shortcut that keeps episodes productive without requiring the agent to learn search from scratch.

**The shoot gate is intentional.** Action 3 only fires a click if YOLO detects the guard and aim error is below 0.20. The policy can select "shoot" all it wants, but the gate prevents reward hacking through click spamming.

**Observation space is intentionally minimal.** Five numbers is all the agent gets. No pixels, no health bars, no minimap. This keeps the network tiny and training fast, at the cost of the agent being blind to everything except the guard's bounding box.

---

## File Paths

```
YOLO weights:  runs\detect\guard_detector_v1-2\weights\best.pt
Checkpoints:   model\
TensorBoard:   runs\ppo_fortnite\
```
