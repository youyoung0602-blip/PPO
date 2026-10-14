from __future__ import annotations
import argparse
from stable_baselines3 import PPO
from algorithm.environment.waam_env import WAAMBaselineEnv
from algorithm.preprocessing.task_generator import build_scenario


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stl", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--model", required=True)
    args = p.parse_args()

    scenario = build_scenario(args.stl, args.config)
    env = WAAMBaselineEnv(scenario)
    model = PPO.load(args.model)
    obs, _ = env.reset()
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    print("final makespan:", info.get("makespan"))


if __name__ == "__main__":
    main()
