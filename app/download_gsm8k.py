import json
import os
from datasets import load_dataset

def download():
    print("Downloading GSM8K dataset...")
    
    # Calculate output path relative to this script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(script_dir, "gsm8k_train.json")
    
    try:
        dataset = load_dataset("gsm8k", "main")
        train_data = []
        for item in dataset["train"]:
            train_data.append({
                "question": item["question"],
                "answer": item["answer"]
            })
        
        # Save a subset of 1000 examples to keep the file size reasonable
        subset = train_data[:1000]
        with open(out_path, "w") as f:
            json.dump(subset, f, indent=2)
        print(f"Successfully saved {len(subset)} examples to {out_path}")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        # Write a dummy dataset if it fails
        dummy = [
            {
                "question": "Weng earns $12 an hour for babysitting. Yesterday, she babysat for 5 hours. How much money did she earn?",
                "answer": "Weng earns $12 an hour. She babysat for 5 hours. So she earned 12 * 5 = $60.\n#### 60"
            },
            {
                "question": "Albert is 4 years older than his sister. His sister is 12 years old. How old is Albert?",
                "answer": "His sister is 12 years old. Albert is 4 years older, so Albert is 12 + 4 = 16 years old.\n#### 16"
            }
        ]
        with open(out_path, "w") as f:
            json.dump(dummy, f, indent=2)
        print(f"Wrote dummy dataset to {out_path} due to error.")

if __name__ == "__main__":
    download()
