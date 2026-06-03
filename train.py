from env import FortniteEnv
from model import ActorCritic, ModelTrainer

env = FortniteEnv()
model = ActorCritic()
trainer = ModelTrainer(model, env)

try:
    trainer.train(total_steps=20_000)
finally:
    env.close()