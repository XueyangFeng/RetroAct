import argparse
import sys
sys.path.append('/data/fengxueyang/RetroAct/hotpotqa/reflexion')
from agents_zs import ReactAgent, ReflectionAgent
from model import LocalModel
from environment import QAEnv
import os

def main(lora_dir, base_model_dir, save_file):
    model = LocalModel(model_dir=base_model_dir, lora_dir=lora_dir)
    env = QAEnv(split="test")
    res = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    em_res = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for i in range(100):
        agent = ReflectionAgent(question_id=i, action_model=model, reflect_model=model, environment=env, save_dir=save_file)
        for j in range(10):
            agent.run()
            res[j] += agent.f1()
            if agent.f1() == 1.0:
                em_res[j] += 1.0
        res_str = " ".join([str(r) for r in res])
        em_res_str = " ".join([str(r) for r in em_res])
        print(f"res: {res_str}")
        print(f"em_res: {em_res_str}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a QA model with specified directories.")
    parser.add_argument("--lora_dir", required=True, help="Path to the LoRA directory")
    parser.add_argument("--base_model_dir", required=True, help="Path to the base model directory")
    parser.add_argument("--save_file", required=True, help="Directory to save the results")

    args = parser.parse_args()

    main(args.lora_dir, args.base_model_dir, args.save_file)
