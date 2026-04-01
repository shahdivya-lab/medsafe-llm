from datasets import Dataset
from trl import DPOTrainer, DPOConfig
from transformers import TrainingArguments
from model import load_base_model, apply_lora
import pandas as pd


def load_preference_dataset(path: str) -> Dataset:
    """
    Load medical preference dataset.
    Each row needs: prompt, chosen response, rejected response.

    chosen  = safe, aligned, professionally appropriate answer
    rejected = overconfident, diagnostic, or unsafe answer
    """
    df = pd.read_csv(path)
    return Dataset.from_pandas(df[["prompt", "chosen", "rejected"]])


def train():
    model, tokenizer = load_base_model()
    model = apply_lora(model)

    # Load your preference dataset
    # dataset = load_preference_dataset("data/medical_preferences.csv")

    dpo_config = DPOConfig(
        beta=0.1,               # KL penalty — controls alignment strength
        max_length=512,
        max_prompt_length=256,
    )

    training_args = TrainingArguments(
        output_dir="./medsafe-lora-checkpoints",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_steps=100,
        warmup_ratio=0.1,
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=None,  # plug in dataset here
        tokenizer=tokenizer,
        beta=dpo_config.beta,
    )

    print("DPO Trainer ready. Plug in your preference dataset to begin.")
    print("Reference papers:")
    print("  DPO — Rafailov et al., 2023 (arxiv.org/abs/2305.18290)")
    print("  Constitutional AI — Bai et al., 2022 (arxiv.org/abs/2212.08073)")


if __name__ == "__main__":
    train()
```

Commit changes.

---

Your `medsafe-llm` repo now looks like:
```
medsafe-llm/
├── README.md
├── requirements.txt
└── src/
    ├── model.py
    └── train_dpo.py
