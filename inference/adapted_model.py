"""Reusable wrapper around the frozen Qwen2.5-1.5B-Instruct base model plus
the Stage 4 LoRA adapter.

Loading (tokenizer, base model, adapter attachment) happens once, in
`AdaptedModel.__init__`; call `.ask(context, question)` as many times as
needed afterward without reloading anything. Used by
`inference/run_inference.py` (Stage 7's CLI/interactive tool).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_preparation.prepare_dataset import RESPONSE_FORMAT_INSTRUCTION

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_DIR = REPO_ROOT / "models" / "qwen2.5-1.5b-instruct-lora-behavior-adapter"
MAX_NEW_TOKENS = 80


def _pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_prompt(context: str, question: str) -> str:
    """The exact task format the adapter was trained on (Stage 2's dataset
    prompt shape) — context, question, and the format/grounding
    instruction, with no worked examples."""
    return f"Context: {context}\nQuestion: {question}\nInstruction: {RESPONSE_FORMAT_INSTRUCTION}"


class AdaptedModel:
    """Frozen base model + Stage 4 LoRA adapter, loaded once.

        model = AdaptedModel()
        result = model.ask(context, question)
    """

    def __init__(self, adapter_dir: Path = ADAPTER_DIR, device: Optional[str] = None):
        if not adapter_dir.exists():
            raise FileNotFoundError(f"{adapter_dir} not found — run training/train.py (Step 4) first.")

        self.device = device or _pick_device()

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
        base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32, local_files_only=True)
        base_model.to(self.device)

        self.model = PeftModel.from_pretrained(base_model, adapter_dir)
        self.model.to(self.device)
        self.model.eval()

        self.active_adapters = list(self.model.active_adapters)
        assert self.active_adapters == ["default"], (
            f"expected adapter 'default' active, got {self.active_adapters}"
        )

        self.pad_token_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )

    def ask(self, context: str, question: str) -> str:
        """Run one deterministic (greedy) generation and return the raw
        decoded completion — expected to be 'ANSWER: ...\\nCONFIDENCE: ...'."""
        prompt = build_prompt(context, question)
        messages = [{"role": "user", "content": prompt}]
        chat_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(chat_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=1,
                pad_token_id=self.pad_token_id,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
