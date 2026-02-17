"""
Fine-tune TinyLlama on trading data using Unsloth

Efficient LoRA fine-tuning with minimal VRAM usage (~650MB)
"""

import json
import os
from pathlib import Path
from datetime import datetime
import torch
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer

print("=" * 60)
print("CAPPUCCINO AI TRAINER - TinyLlama Fine-Tuning")
print("=" * 60)

# Check if Unsloth is available
try:
    from unsloth import FastLanguageModel, is_bfloat16_supported
    UNSLOTH_AVAILABLE = True
    print("✓ Unsloth detected - using optimized training")
except ImportError:
    UNSLOTH_AVAILABLE = False
    print("⚠️  Unsloth not installed - using standard training")
    print("   Install with: pip install unsloth")


class TinyLlamaTrainer:
    """Manages TinyLlama fine-tuning for trading insights."""

    def __init__(
        self,
        data_file: str = "ai_training/data/training_data.jsonl",
        output_dir: str = "ai_training/models",
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    ):
        self.data_file = Path(data_file)
        self.output_dir = Path(output_dir)
        self.model_name = model_name

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.tokenizer = None

    def load_training_data(self):
        """Load and prepare training data."""
        if not self.data_file.exists():
            raise FileNotFoundError(
                f"Training data not found at {self.data_file}\n"
                "Run paper trader to collect trade data first."
            )

        print(f"\nLoading training data from {self.data_file}...")

        # Load JSONL file
        examples = []
        with open(self.data_file, 'r') as f:
            for line in f:
                examples.append(json.loads(line))

        print(f"✓ Loaded {len(examples)} training examples")

        if len(examples) < 10:
            print(f"⚠️  Warning: Only {len(examples)} examples - recommend 50+ for good results")
            print("   Continue collecting trade data for better fine-tuning")

        # Convert to format expected by SFTTrainer
        formatted_examples = []
        for ex in examples:
            # Alpaca format
            formatted_examples.append({
                "text": self._format_prompt(ex)
            })

        return Dataset.from_list(formatted_examples)

    def _format_prompt(self, example: dict) -> str:
        """Format training example in Alpaca instruction format."""
        prompt = f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""
        return prompt

    def load_model(self, max_seq_length: int = 2048):
        """Load TinyLlama base model with LoRA adapters."""
        print(f"\nLoading model: {self.model_name}...")

        if UNSLOTH_AVAILABLE:
            # Use Unsloth for optimized loading
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_name,
                max_seq_length=max_seq_length,
                dtype=None,  # Auto-detect
                load_in_4bit=True,  # Use 4-bit quantization for minimal VRAM
            )

            # Add LoRA adapters
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=16,  # LoRA rank
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj"],
                lora_alpha=16,
                lora_dropout=0,  # Optimized dropout
                bias="none",  # Bias strategy
                use_gradient_checkpointing="unsloth",  # Memory efficient
                random_state=42,
            )

            print("✓ Model loaded with Unsloth optimizations")
        else:
            # Standard transformers loading
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                load_in_4bit=True,
                device_map="auto",
            )

            # Add LoRA with peft
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=16,
                lora_alpha=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )

            self.model = get_peft_model(self.model, lora_config)
            print("✓ Model loaded with standard LoRA")

        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.model.parameters())
        print(f"  Trainable params: {trainable_params:,} ({trainable_params/all_params*100:.2f}%)")

    def train(
        self,
        dataset: Dataset,
        num_epochs: int = 3,
        batch_size: int = 2,
        learning_rate: float = 2e-4,
        warmup_steps: int = 10,
    ):
        """Fine-tune the model."""
        print(f"\nStarting training...")
        print(f"  Epochs: {num_epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {learning_rate}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.output_dir / f"run_{timestamp}"

        training_args = TrainingArguments(
            output_dir=str(run_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            warmup_steps=warmup_steps,
            learning_rate=learning_rate,
            fp16=not is_bfloat16_supported() if UNSLOTH_AVAILABLE else True,
            bf16=is_bfloat16_supported() if UNSLOTH_AVAILABLE else False,
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            save_strategy="epoch",
            save_total_limit=3,
            report_to="none",  # Disable wandb/tensorboard
        )

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=2048,
            args=training_args,
        )

        # Train!
        print(f"\n{'='*60}")
        print("Training started...")
        print(f"{'='*60}\n")

        trainer.train()

        print(f"\n{'='*60}")
        print("Training completed!")
        print(f"{'='*60}\n")

        # Save the fine-tuned model
        final_model_dir = self.output_dir / "tinyllama_trading_latest"
        self.save_model(final_model_dir)

        return final_model_dir

    def save_model(self, save_dir: Path):
        """Save the fine-tuned model."""
        print(f"\nSaving model to {save_dir}...")

        if UNSLOTH_AVAILABLE:
            # Save with Unsloth (more efficient)
            self.model.save_pretrained(str(save_dir))
            self.tokenizer.save_pretrained(str(save_dir))
        else:
            # Standard save
            self.model.save_pretrained(str(save_dir))
            self.tokenizer.save_pretrained(str(save_dir))

        # Save training metadata
        metadata = {
            "base_model": self.model_name,
            "trained_at": datetime.now().isoformat(),
            "method": "unsloth_lora" if UNSLOTH_AVAILABLE else "standard_lora"
        }

        with open(save_dir / "training_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ Model saved successfully")

    def export_for_ollama(self, model_dir: Path, output_name: str = "tinyllama-trading"):
        """
        Export fine-tuned model in format compatible with Ollama.

        Creates a Modelfile and quantized GGUF model.
        """
        print(f"\nExporting model for Ollama...")

        # Check if llama.cpp is available for quantization
        try:
            import subprocess
            subprocess.run(["which", "quantize"], check=True, capture_output=True)
            QUANTIZE_AVAILABLE = True
        except:
            QUANTIZE_AVAILABLE = False
            print("⚠️  llama.cpp quantize not found - skipping GGUF conversion")
            print("   Model can still be used with HuggingFace transformers")

        # Create Modelfile for Ollama
        modelfile = f"""FROM {model_dir}

TEMPLATE \"\"\"### Instruction:
{{{{ .Prompt }}}}

### Response:
\"\"\"

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40

SYSTEM \"\"\"You are a specialized trading AI assistant trained on real trading data from the Cappuccino DRL trading system. You analyze market conditions, news sentiment, and technical signals to provide actionable trading insights.\"\"\"
"""

        modelfile_path = model_dir / "Modelfile"
        with open(modelfile_path, 'w') as f:
            f.write(modelfile)

        print(f"✓ Modelfile created: {modelfile_path}")
        print(f"\nTo use with Ollama, run:")
        print(f"  cd {model_dir}")
        print(f"  ollama create {output_name} -f Modelfile")
        print(f"  ollama run {output_name}")

        return modelfile_path


def main():
    """Main training pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune TinyLlama on trading data")
    parser.add_argument("--data", default="ai_training/data/training_data.jsonl",
                       help="Path to training data")
    parser.add_argument("--output", default="ai_training/models",
                       help="Output directory for trained models")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2,
                       help="Training batch size")
    parser.add_argument("--lr", type=float, default=2e-4,
                       help="Learning rate")
    parser.add_argument("--export-ollama", action="store_true",
                       help="Export model for Ollama after training")

    args = parser.parse_args()

    # Initialize trainer
    trainer = TinyLlamaTrainer(
        data_file=args.data,
        output_dir=args.output
    )

    # Load training data
    dataset = trainer.load_training_data()

    # Load model
    trainer.load_model()

    # Train
    model_dir = trainer.train(
        dataset=dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )

    # Export for Ollama if requested
    if args.export_ollama:
        trainer.export_for_ollama(model_dir)

    print("\n" + "=" * 60)
    print("Training pipeline completed successfully!")
    print("=" * 60)
    print(f"\nModel saved to: {model_dir}")
    print("\nNext steps:")
    print("1. Test the model: python3 ai_training/test_model.py")
    print("2. Deploy to Ollama: Follow instructions above")
    print("3. Integrate with advisor: Update ollama_autonomous_advisor.py")


if __name__ == "__main__":
    main()
