"""veRL preprocessing and training runner script.

This script formats the local GSM8K JSON dataset into Parquet,
and runs the veRL PPO/GRPO trainer.
"""

from __future__ import annotations

import os
import sys
import json
import subprocess
import pandas as pd

def preprocess():
    print("=== [veRL Runner] Preprocessing GSM8K data ===")
    
    # Locate the local gsm8k dataset
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, "gsm8k_train.json")
    
    if not os.path.exists(json_path):
        # Fallback if run elsewhere
        json_path = "/app/gsm8k_train.json"
        
    with open(json_path, "r") as f:
        dataset = json.load(f)
        
    data = []
    for item in dataset:
        q = item["question"]
        a = item["answer"]
        # Extract numeric answer from Weng earns... format (#### 60)
        ref_ans = a.split("####")[-1].strip() if "####" in a else a
        
        data.append({
            "prompt": [{"role": "user", "content": q}],
            "extra_info": {
                "reference": ref_ans,
                "answer": a
            }
        })
        
    df = pd.DataFrame(data)
    
    # Write to temp files
    train_path = "/tmp/train.parquet"
    test_path = "/tmp/test.parquet"
    
    df.to_parquet(train_path)
    df.to_parquet(test_path)
    print(f"=== [veRL Runner] Saved {len(df)} lines to {train_path} ===")

def main():
    preprocess()
    
    # Launch verl training
    cmd = [
        sys.executable, "-m", "verl.trainer.main_ppo",
        "data.train_files=/tmp/train.parquet",
        "data.val_files=/tmp/test.parquet",
        "algorithm.adv_estimator=grpo",
        "trainer.n_gpus_per_node=1",
        "trainer.nnodes=1",
        "trainer.total_epochs=1",
        "actor_rollout_ref.actor.strategy=fsdp",
        "actor_rollout_ref.ref.strategy=fsdp",
        "actor_rollout_ref.actor.ppo_mini_batch_size=1",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.2",
        "trainer.max_steps=10",
        "trainer.logger=['console']",
    ]
    
    print("=== [veRL Runner] Launching veRL PPO/GRPO ===")
    print("Command:", " ".join(cmd))
    
    # We flush stdout to make sure logs stream immediately
    sys.stdout.flush()
    sys.stderr.flush()
    
    res = subprocess.run(cmd, capture_output=False)
    sys.exit(res.returncode)

if __name__ == "__main__":
    main()
