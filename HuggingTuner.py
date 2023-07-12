from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
)
import torch
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import load_dataset


class Tuner:
    def __init__(self, model_id) -> None:
        self.model_id = self.model_id

    def _setup_bnb_config(self):
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        return None

    def _setup_model_and_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=self._setup_bnb_config(),
            device_map={"": 0},
        ).gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)
