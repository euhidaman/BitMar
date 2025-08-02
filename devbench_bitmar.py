"""
BitMar model class for DevBench evaluation
Add this file to: evaluation-pipeline-2024/devbench/model_classes/bitmar.py
"""

from devbench.eval_model import EvalModel
from tqdm import tqdm
import torch
import numpy as np
from PIL import Image
import sys
from pathlib import Path


class BitMarDevBenchModel(EvalModel):
    """BitMar model wrapper for DevBench evaluation"""

    def __init__(self, model, processor=None, device="cpu"):
        self.device = device
        self.model = model.to(device)
        self.processor = processor

        # Required methods for DevBench
        self.get_image_features = self.get_all_image_feats
        self.get_text_features = self.get_all_text_feats
        self.get_similarity_scores = self.get_all_sim_scores

    def get_all_sim_scores(self, dataloader):
        """
        Gets image-text similarity scores from a dataloader using BitMar model
        Returns: numpy array of shape [num_trials, num_images_per_trial, num_texts_per_trial]
        """
        all_sims = []
        with torch.no_grad():
            for d in tqdm(dataloader, desc="Processing data"):
                num_images = len(d["images"])
                num_texts = len(d["text"])
                sims = np.zeros((num_images, num_texts))

                for i, image in enumerate(d["images"]):
                    if image.mode != 'RGB':
                        image = image.convert('RGB')

                    for j, text in enumerate(d["text"]):
                        # Prepare inputs for each image-text pair
                        prompt = f"The caption for this image is: {text}."
                        encoding = self.processor(
                            images=image, text=prompt, return_tensors="pt")

                        # Move to device
                        for key in encoding:
                            if torch.is_tensor(encoding[key]):
                                encoding[key] = encoding[key].to(self.device)

                        # Set labels for loss computation
                        encoding['labels'] = encoding['input_ids']

                        # Forward pass
                        outputs = self.model(**encoding)

                        # Use negative loss as similarity score
                        sims[i, j] = -outputs['loss'].detach().cpu().numpy()

                all_sims.append(sims)

        return np.stack(all_sims, axis=0)

    def get_all_image_feats(self, dataloader):
        """
        Gets image features from a dataloader
        Returns: numpy array of shape [num_images, embed_dim]
        """
        all_feats = []
        with torch.no_grad():
            for d in tqdm(dataloader, desc="Processing data"):
                images_rgb = [image.convert("RGB") for image in d["images"]]
                dummy_texts = [" "] * len(images_rgb)

                encoding = self.processor(
                    images=images_rgb, text=dummy_texts, return_tensors="pt")

                # Move to device
                for key in encoding:
                    if torch.is_tensor(encoding[key]):
                        encoding[key] = encoding[key].to(self.device)

                encoding["labels"] = encoding["input_ids"]

                # Get hidden states from the model
                outputs = self.model(**encoding, output_hidden_states=True)

                # Extract features from last hidden state
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                    hidden_states = outputs.hidden_states[-1]
                else:
                    # Fallback: use the model's vision features
                    if hasattr(self.model, 'bitmar') and hasattr(self.model.bitmar, 'last_vision_features'):
                        hidden_states = self.model.bitmar.last_vision_features
                    else:
                        # Use input vision features as fallback
                        hidden_states = encoding.get(
                            'vision_features', torch.zeros(len(images_rgb), 64))

                mean_feats = hidden_states.mean(dim=1).detach().cpu().numpy()
                all_feats.append(mean_feats)

        return np.concatenate(all_feats, axis=0)

    def get_all_text_feats(self, dataloader):
        """
        Gets text features from a dataloader
        Returns: numpy array of shape [num_texts, embed_dim]
        """
        all_feats = []
        with torch.no_grad():
            for d in tqdm(dataloader, desc="Processing data"):
                texts = d["text"] if isinstance(
                    d["text"], list) else [d["text"]]

                # Create dummy vision features
                dummy_vision = torch.zeros(len(texts), 3, 224, 224)

                encoding = self.processor(
                    images=dummy_vision, text=texts, return_tensors="pt")

                # Move to device
                for key in encoding:
                    if torch.is_tensor(encoding[key]):
                        encoding[key] = encoding[key].to(self.device)

                encoding["labels"] = encoding["input_ids"]

                # Get hidden states from the model
                outputs = self.model(**encoding, output_hidden_states=True)

                # Extract text features
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                    hidden_states = outputs.hidden_states[-1]
                else:
                    # Fallback: use text embeddings
                    if hasattr(self.model, 'bitmar') and hasattr(self.model.bitmar, 'last_text_features'):
                        hidden_states = self.model.bitmar.last_text_features
                    else:
                        # Use input embeddings as fallback
                        hidden_states = self.model.get_input_embeddings()(
                            encoding['input_ids'])

                mean_feats = hidden_states.mean(dim=1).detach().cpu().numpy()
                all_feats.append(mean_feats)

        return np.concatenate(all_feats, axis=0)


def load_bitmar_model(model_path, device="cuda"):
    """
    Load BitMar model for DevBench evaluation
    This function should be called from eval_devbench.sh with model_type="bitmar"
    """
    from transformers import AutoModel, AutoTokenizer

    # Load model and tokenizer
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True)

    # Import BitMar processor (assumes BitMar is in the path)
    try:
        from src.hf_compatibility import BitMarProcessor
        processor = BitMarProcessor(tokenizer=tokenizer)
    except ImportError:
        # Fallback processor
        class SimpleProcessor:
            def __init__(self, tokenizer):
                self.tokenizer = tokenizer

            def __call__(self, text=None, images=None, return_tensors="pt", **kwargs):
                result = {}
                if text is not None:
                    if isinstance(text, str):
                        text = [text]
                    tokenized = self.tokenizer(
                        text, return_tensors=return_tensors, padding=True, truncation=True, **kwargs)
                    result.update(tokenized)
                if images is not None:
                    # Simple image processing
                    if isinstance(images, Image.Image):
                        images = [images]
                    # Create dummy vision features
                    result['vision_features'] = torch.zeros(
                        len(images), 3, 224, 224)
                return result

        processor = SimpleProcessor(tokenizer)

    # Create DevBench wrapper
    eval_model = BitMarDevBenchModel(
        model=model, processor=processor, device=device)

    return eval_model
