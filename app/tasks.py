"""Ray remote tasks and actors for the distributed RL loop.

This replaces the Mandelbrot calculations with a simple RL (GRPO) loop
using Qwen2.5-Instruct and GSM8K to demonstrate convergence on L4 GPUs on GKE.
"""

from __future__ import annotations

import os
import re
import json
import time
import socket
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
import ray


@ray.remote(num_gpus=1)
class RLTrainer:
    """Ray Actor running on a GPU node, performing rollouts and GRPO policy updates."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", lr: float = 5e-5):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing RLTrainer on {self.device} (hostname: {socket.gethostname()})")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model in bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map={"": self.device}
        )

        # Configure LoRA to save memory and optimize weights
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        )
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

        # Optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        # Load local GSM8K train dataset
        dataset_path = os.path.join(os.path.dirname(__file__), "gsm8k_train.json")
        with open(dataset_path, "r") as f:
            self.dataset = json.load(f)
        print(f"Loaded {len(self.dataset)} training examples.")

        self.step_count = 0

    def train_step(self, batch_size: int = 2, group_size: int = 4) -> dict:
        """Perform a single GRPO-like training step.

        Selects random prompts, generates completions, computes rewards,
        computes relative advantages, and runs backprop.
        """
        import random
        samples = random.sample(self.dataset, min(batch_size, len(self.dataset)))

        step_logs = []
        all_loss = 0.0
        all_rewards = []

        for sample in samples:
            question = sample["question"]
            target_answer = sample["answer"]

            # Chat formatting for Qwen with explicit formatting instructions
            messages = [
                {"role": "system", "content": "You are a math tutor. Solve the following grade-school math problem step-by-step. Write out your reasoning clearly, and end your response by stating the final numerical answer on a new line with the prefix '#### ', like this: #### <number>"},
                {"role": "user", "content": question}
            ]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            # Tokenize prompt
            prompt_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            prompt_input_ids = prompt_inputs["input_ids"]
            prompt_len = prompt_input_ids.shape[1]

            # Replicate input ids for the group size to sample multiple rollouts
            repeated_input_ids = prompt_input_ids.repeat(group_size, 1)

            # 1. Rollout (Generate completions)
            self.model.eval()
            with torch.no_grad():
                outputs = self.model.generate(
                    repeated_input_ids,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            completions = []
            rewards = []
            completion_input_ids_list = []

            for out in outputs:
                gen_tokens = out[prompt_len:]
                completion_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
                completions.append(completion_text)

                # Compute rule-based reward (accurate or not)
                reward = self.reward_fn(completion_text, target_answer)
                rewards.append(reward)
                all_rewards.append(reward)

                completion_input_ids_list.append(out)

            # 2. Advantage Estimation (within group)
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
            mean_r = rewards_tensor.mean()
            std_r = rewards_tensor.std()
            if std_r < 1e-4:
                advantages = rewards_tensor - mean_r
            else:
                advantages = (rewards_tensor - mean_r) / (std_r + 1e-6)

            # 3. Policy Update (GRPO optimization)
            self.model.train()

            # Align lengths by padding
            batch_inputs = []
            batch_labels = []
            max_len = max(out.shape[0] for out in completion_input_ids_list)

            for out in completion_input_ids_list:
                pad_len = max_len - out.shape[0]
                if pad_len > 0:
                    padded_out = torch.cat([
                        out, 
                        torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long, device=self.device)
                    ])
                else:
                    padded_out = out

                # Target labels: mask out prompt tokens and pad tokens with -100
                labels = padded_out.clone()
                labels[:prompt_len] = -100
                labels[out.shape[0]:] = -100

                batch_inputs.append(padded_out)
                batch_labels.append(labels)

            input_ids_tensor = torch.stack(batch_inputs)
            labels_tensor = torch.stack(batch_labels)

            # Forward
            logits = self.model(input_ids_tensor).logits

            # Log probabilities
            log_probs = torch.log_softmax(logits, dim=-1)
            shift_logits = log_probs[:, :-1, :].contiguous()
            shift_labels = labels_tensor[:, 1:].contiguous()

            # Filter generated labels
            loss_mask = shift_labels != -100
            gather_labels = shift_labels.clone()
            gather_labels[gather_labels == -100] = 0

            token_log_probs = torch.gather(shift_logits, dim=-1, index=gather_labels.unsqueeze(-1)).squeeze(-1)
            token_log_probs = token_log_probs * loss_mask

            # Logprob of complete generated sequence
            seq_log_probs = token_log_probs.sum(dim=-1) / loss_mask.sum(dim=-1).clamp(min=1)

            # GRPO surrogate loss (without KL penalty for simple demo stability/memory)
            advantages_device = advantages.to(self.device)
            loss = - (seq_log_probs * advantages_device).mean()

            # Optimization Step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            all_loss += loss.item()

            step_logs.append({
                "question": question,
                "target": target_answer,
                "completions": [
                    {"text": c, "reward": r, "advantage": adv.item()}
                    for c, r, adv in zip(completions, rewards, advantages)
                ]
            })

        self.step_count += 1
        avg_loss = all_loss / len(samples)
        avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0

        return {
            "step": self.step_count,
            "loss": avg_loss,
            "avg_reward": avg_reward,
            "logs": step_logs,
            "pod_name": socket.gethostname()
        }

    def reward_fn(self, completion: str, target: str) -> float:
        pred = self.extract_answer(completion)
        tgt = self.extract_answer(target)
        if pred is not None and tgt is not None:
            try:
                if float(pred) == float(tgt):
                    return 1.0
            except ValueError:
                if pred == tgt:
                    return 1.0
        return 0.0

    def extract_answer(self, text: str) -> str | None:
        # Try finding #### <number>
        match = re.search(r'####\s*(-?\d+(?:\.\d+)?)', text)
        if match:
            return match.group(1).strip()

        # Try finding \boxed{<number>}
        match = re.search(r'\\boxed\{([^{}]+)\}', text)
        if match:
            val = match.group(1).strip()
            num_match = re.search(r'(-?\d+(?:\.\d+)?)', val)
            if num_match:
                return num_match.group(1).strip()
            return val

        # Fallback: last number
        numbers = re.findall(r'(-?\d+(?:\.\d+)?)', text)
        if numbers:
            return numbers[-1].strip()
        return None
