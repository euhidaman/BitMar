"""
DevBench integration for BitMar model
Compatible with the 2024 BabyLM evaluation pipeline
"""

import torch
from src.hf_compatibility import BitMarProcessor, BitMarEvalModel
from transformers import AutoModel, AutoTokenizer
import sys
from pathlib import Path

# Add BitMar src to path
bitmar_src = Path(__file__).parent.parent / "BitMar" / "src"
sys.path.append(str(bitmar_src))


def load_bitmar_for_devbench(model_path, device="cuda"):
    """
    Load BitMar model for DevBench evaluation
    """
    # Load HuggingFace format model
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True)

    # Create processor
    processor = BitMarProcessor(tokenizer=tokenizer)

    # Create DevBench-compatible wrapper
    eval_model = BitMarEvalModel(
        model=model, processor=processor, device=device)

    return eval_model

# For direct DevBench integration, you can add this to:
# evaluation-pipeline-2024/devbench/model_classes/bitmar.py


class BitMarDevBenchModel:
    """BitMar model wrapper for DevBench evaluation"""

    def __init__(self, model_path, device="cuda"):
        from transformers import AutoModel, AutoTokenizer
        import sys
        from pathlib import Path

        # Add BitMar to path
        bitmar_path = Path(__file__).parent.parent.parent / "BitMar" / "src"
        if str(bitmar_path) not in sys.path:
            sys.path.append(str(bitmar_path))

        from hf_compatibility import BitMarProcessor, BitMarEvalModel

        # Load model and tokenizer
        self.model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True)

        # Create processor
        self.processor = BitMarProcessor(tokenizer=self.tokenizer)

        # Create evaluation wrapper
        self.eval_model = BitMarEvalModel(
            model=self.model,
            processor=self.processor,
            device=device
        )

        # DevBench required attributes
        self.get_image_features = self.eval_model.get_image_features
        self.get_text_features = self.eval_model.get_text_features
        self.get_similarity_scores = self.eval_model.get_similarity_scores
