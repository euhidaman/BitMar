"""
HuggingFace compatibility layer for BitMar model
Makes BitMar compatible with AutoModel classes for evaluation pipeline
Enhanced for both 2024 and 2025 BabyLM evaluation pipelines
"""

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel, AutoConfig, AutoModel, AutoTokenizer, AutoProcessor
from transformers.modeling_outputs import CausalLMOutput
from typing import Dict, Optional, Union, Tuple, List
import json
import logging
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

from .model import BitMarModel

logger = logging.getLogger(__name__)


class BitMarConfig(PretrainedConfig):
    """
    Configuration class for BitMar model
    Compatible with HuggingFace transformers
    """
    model_type = "bitmar"

    def __init__(
        self,
        vocab_size: int = 50257,
        text_encoder_dim: int = 192,
        text_encoder_layers: int = 3,
        text_encoder_heads: int = 3,
        text_decoder_dim: int = 192,
        text_decoder_layers: int = 3,
        text_decoder_heads: int = 3,
        vision_encoder_dim: int = 768,
        vision_latent_size: int = 64,
        vision_hidden_size: int = 32,
        fusion_hidden_size: int = 96,
        fusion_num_heads: int = 3,
        fusion_num_layers: int = 1,
        memory_size: int = 16,
        episode_dim: int = 96,
        memory_alpha: float = 0.3,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        text_encoder_name: str = "gpt2",
        direct_writing: bool = True,
        memory_compression: bool = True,
        vision_compression_method: str = "learnable_projection",
        vision_spatial_pooling: bool = True,
        vision_pool_size: int = 4,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.text_encoder_dim = text_encoder_dim
        self.text_encoder_layers = text_encoder_layers
        self.text_encoder_heads = text_encoder_heads
        self.text_decoder_dim = text_decoder_dim
        self.text_decoder_layers = text_decoder_layers
        self.text_decoder_heads = text_decoder_heads
        self.vision_encoder_dim = vision_encoder_dim
        self.vision_latent_size = vision_latent_size
        self.vision_hidden_size = vision_hidden_size
        self.fusion_hidden_size = fusion_hidden_size
        self.fusion_num_heads = fusion_num_heads
        self.fusion_num_layers = fusion_num_layers
        self.memory_size = memory_size
        self.episode_dim = episode_dim
        self.memory_alpha = memory_alpha
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.text_encoder_name = text_encoder_name
        self.direct_writing = direct_writing
        self.memory_compression = memory_compression
        self.vision_compression_method = vision_compression_method
        self.vision_spatial_pooling = vision_spatial_pooling
        self.vision_pool_size = vision_pool_size

        super().__init__(**kwargs)


class BitMarForCausalLM(PreTrainedModel):
    """
    BitMar model wrapped for HuggingFace compatibility
    Compatible with AutoModelForCausalLM
    """
    config_class = BitMarConfig
    base_model_prefix = "bitmar"
    supports_gradient_checkpointing = True

    def __init__(self, config: BitMarConfig):
        super().__init__(config)

        # Convert HF config to BitMar config format
        bitmar_config = {
            'vocab_size': config.vocab_size,
            'text_encoder_dim': config.text_encoder_dim,
            'text_encoder_layers': config.text_encoder_layers,
            'text_encoder_heads': config.text_encoder_heads,
            'text_decoder_dim': config.text_decoder_dim,
            'text_decoder_layers': config.text_decoder_layers,
            'text_decoder_heads': config.text_decoder_heads,
            'vision_encoder_dim': config.vision_encoder_dim,
            'vision_latent_size': config.vision_latent_size,
            'vision_hidden_size': config.vision_hidden_size,
            'fusion_hidden_size': config.fusion_hidden_size,
            'fusion_num_heads': config.fusion_num_heads,
            'fusion_num_layers': config.fusion_num_layers,
            'memory_size': config.memory_size,
            'episode_dim': config.episode_dim,
            'memory_alpha': config.memory_alpha,
            'max_seq_len': config.max_seq_len,
            'dropout': config.dropout,
            'text_encoder_name': config.text_encoder_name,
            'direct_writing': config.direct_writing,
            'memory_compression': config.memory_compression,
            'vision_compression_method': config.vision_compression_method,
            'vision_spatial_pooling': config.vision_spatial_pooling,
            'vision_pool_size': config.vision_pool_size,
        }

        # Create the actual BitMar model
        self.bitmar = BitMarModel(bitmar_config)

        # Store config for easy access
        self.bitmar_config = bitmar_config

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        vision_features: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> CausalLMOutput:
        """
        Forward pass compatible with HuggingFace CausalLM interface
        """
        if input_ids is None:
            raise ValueError("input_ids is required")

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # For text-only evaluation, create dummy vision features
        if vision_features is None:
            batch_size = input_ids.shape[0]
            device = input_ids.device
            # Create dummy vision features with expected dimension
            vision_features = torch.zeros(
                batch_size,
                self.bitmar_config['vision_latent_size'],
                device=device,
                dtype=torch.float32
            )

        # Call BitMar forward
        outputs = self.bitmar(
            input_ids=input_ids,
            attention_mask=attention_mask,
            vision_features=vision_features,
            labels=labels,
            mode="train" if labels is not None else "inference"
        )

        return CausalLMOutput(
            loss=outputs.get('loss'),
            logits=outputs['logits'],
            hidden_states=None,  # BitMar doesn't return hidden states in expected format
            attentions=None,     # Could be adapted if needed
        )

    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        vision_features: Optional[torch.FloatTensor] = None,
        max_length: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        **kwargs
    ):
        """
        Generate text using BitMar's generation method
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        if vision_features is None:
            batch_size = input_ids.shape[0]
            device = input_ids.device
            vision_features = torch.zeros(
                batch_size,
                self.bitmar_config['vision_latent_size'],
                device=device,
                dtype=torch.float32
            )

        # Use BitMar's generate method
        results = self.bitmar.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            vision_features=vision_features,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p
        )

        return results['generated_ids']

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """
        Prepare inputs for generation
        """
        return {
            "input_ids": input_ids,
            "attention_mask": kwargs.get("attention_mask"),
            "vision_features": kwargs.get("vision_features"),
        }

    def get_input_embeddings(self):
        """Return input embeddings"""
        return self.bitmar.text_decoder.token_embedding

    def set_input_embeddings(self, value):
        """Set input embeddings"""
        self.bitmar.text_decoder.token_embedding = value

    def get_output_embeddings(self):
        """Return output embeddings"""
        return self.bitmar.text_decoder.lm_head

    def set_output_embeddings(self, new_embeddings):
        """Set output embeddings"""
        self.bitmar.text_decoder.lm_head = new_embeddings


# Register the config and model classes
AutoConfig.register("bitmar", BitMarConfig)
AutoModel.register(BitMarConfig, BitMarForCausalLM)

# Also register for AutoModelForCausalLM compatibility
try:
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification
    AutoModelForCausalLM.register(BitMarConfig, BitMarForCausalLM)
    logger.info("✅ BitMarForCausalLM registered successfully")
except ImportError:
    logger.warning("AutoModelForCausalLM not available")


class BitMarForSequenceClassification(PreTrainedModel):
    """
    BitMar model for sequence classification tasks (GLUE, etc.)
    Compatible with AutoModelForSequenceClassification
    """
    config_class = BitMarConfig

    def __init__(self, config):
        super().__init__(config)

        # Convert config to BitMar format
        bitmar_config = {
            'vocab_size': config.vocab_size,
            'text_encoder_dim': config.text_encoder_dim,
            'text_encoder_layers': config.text_encoder_layers,
            'text_encoder_heads': config.text_encoder_heads,
            'text_decoder_dim': config.text_decoder_dim,
            'text_decoder_layers': config.text_decoder_layers,
            'text_decoder_heads': config.text_decoder_heads,
            'vision_encoder_dim': config.vision_encoder_dim,
            'vision_latent_size': config.vision_latent_size,
            'vision_hidden_size': config.vision_hidden_size,
            'fusion_hidden_size': config.fusion_hidden_size,
            'fusion_num_heads': config.fusion_num_heads,
            'fusion_num_layers': config.fusion_num_layers,
            'memory_size': config.memory_size,
            'episode_dim': config.episode_dim,
            'memory_alpha': config.memory_alpha,
            'max_seq_len': config.max_seq_len,
            'dropout': config.dropout,
            'text_encoder_name': config.text_encoder_name,
            'direct_writing': config.direct_writing,
            'memory_compression': config.memory_compression,
            'vision_compression_method': config.vision_compression_method,
            'vision_spatial_pooling': config.vision_spatial_pooling,
            'vision_pool_size': config.vision_pool_size,
        }

        # Create the actual BitMar model
        self.bitmar = BitMarModel(bitmar_config)

        # Add classification head
        self.num_labels = getattr(config, 'num_labels', 2)
        self.classifier = nn.Linear(config.text_decoder_dim, self.num_labels)

        # Store config for easy access
        self.bitmar_config = bitmar_config

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        vision_features: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ):
        """
        Forward pass for sequence classification
        """
        if input_ids is None:
            raise ValueError("input_ids is required")

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # For text-only evaluation, create dummy vision features
        if vision_features is None:
            batch_size = input_ids.shape[0]
            device = input_ids.device
            vision_features = torch.zeros(
                batch_size,
                self.bitmar_config['vision_latent_size'],
                device=device,
                dtype=torch.float32
            )

        # Call BitMar forward to get text representations
        outputs = self.bitmar(
            input_ids=input_ids,
            attention_mask=attention_mask,
            vision_features=vision_features,
            labels=None,  # Don't compute LM loss
            mode="inference"
        )

        # Get the text features for classification
        # Use the last hidden state or pooled output
        if 'text_features' in outputs:
            pooled_output = outputs['text_features'].mean(
                dim=1)  # Mean pooling
        elif 'text_latent' in outputs:
            pooled_output = outputs['text_latent']
        else:
            # Fallback: use the logits and pool them
            pooled_output = outputs['logits'].mean(dim=1)

        # Apply classification head
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression task
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            else:
                # Classification task
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))

        from transformers.modeling_outputs import SequenceClassifierOutput
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )


# Register sequence classification model
try:
    AutoModelForSequenceClassification.register(
        BitMarConfig, BitMarForSequenceClassification)
    logger.info("✅ BitMarForSequenceClassification registered successfully")
except ImportError:
    logger.warning("AutoModelForSequenceClassification not available")


class BitMarProcessor:
    """
    Custom processor for BitMar model compatible with 2024 evaluation pipeline
    Handles both text and vision inputs
    """

    def __init__(self, tokenizer, image_processor=None):
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def __call__(self, text=None, images=None, return_tensors="pt", **kwargs):
        """Process inputs for BitMar model"""
        result = {}

        # Process text
        if text is not None:
            if isinstance(text, str):
                text = [text]

            tokenized = self.tokenizer(
                text,
                return_tensors=return_tensors,
                padding=True,
                truncation=True,
                **kwargs
            )
            result.update(tokenized)

        # Process images
        if images is not None:
            if isinstance(images, Image.Image):
                images = [images]

            # Simple image processing - convert to tensors
            if self.image_processor is not None:
                vision_features = self.image_processor(
                    images, return_tensors=return_tensors)
                result['vision_features'] = vision_features.get(
                    'pixel_values', vision_features)
            else:
                # Default processing: resize and normalize
                processed_images = []
                for img in images:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    # Resize to 224x224 and normalize
                    img = img.resize((224, 224))
                    img_array = np.array(img) / 255.0
                    img_tensor = torch.tensor(
                        img_array).permute(2, 0, 1).float()
                    processed_images.append(img_tensor)

                result['vision_features'] = torch.stack(processed_images)

        return result


class BitMarEvalModel:
    """
    BitMar model wrapper for DevBench evaluation (2024 pipeline)
    Compatible with the EvalModel interface expected by DevBench
    """

    def __init__(self, model, processor=None, device="cpu"):
        self.device = device
        self.model = model.to(device)
        self.processor = processor
        self.tokenizer = processor.tokenizer if processor else None

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
                    hidden_states = self.model.bitmar.last_vision_features if hasattr(
                        self.model.bitmar, 'last_vision_features') else encoding['vision_features']

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
                    hidden_states = self.model.bitmar.last_text_features if hasattr(
                        self.model.bitmar, 'last_text_features') else encoding['input_ids']

                mean_feats = hidden_states.mean(dim=1).detach().cpu().numpy()
                all_feats.append(mean_feats)

        return np.concatenate(all_feats, axis=0)


def save_bitmar_as_hf_model(
    bitmar_model: BitMarModel,
    config_dict: Dict,
    save_directory: Union[str, Path],
    tokenizer=None
):
    """
    Save BitMar model in HuggingFace format for evaluation pipeline compatibility
    Enhanced for both 2024 and 2025 evaluation pipelines

    Args:
        bitmar_model: Trained BitMar model
        config_dict: Model configuration dictionary
        save_directory: Directory to save the model
        tokenizer: Tokenizer to save alongside the model
    """
    save_dir = Path(save_directory)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create HuggingFace config
    hf_config = BitMarConfig(**config_dict)

    # Create wrapper model
    hf_model = BitMarForCausalLM(hf_config)

    # Copy trained weights
    hf_model.bitmar.load_state_dict(bitmar_model.state_dict())

    # Save model and config
    hf_model.save_pretrained(save_dir)

    # Save tokenizer if provided
    if tokenizer is not None:
        tokenizer.save_pretrained(save_dir)

        # Create and save processor for 2024 evaluation pipeline
        processor = BitMarProcessor(tokenizer=tokenizer)

        # Save processor configuration
        processor_config = {
            "processor_class": "BitMarProcessor",
            "tokenizer_class": tokenizer.__class__.__name__,
            "feature_extractor_class": None,
            "image_processor_class": None
        }

        with open(save_dir / "preprocessor_config.json", "w") as f:
            json.dump(processor_config, f, indent=2)

    # Create a model card with evaluation instructions
    model_card_content = f"""---
license: mit
language: en
tags:
- babylm
- multimodal
- episodic-memory
- bitnet
---

# BitMar Model

This is a BitMar (Multimodal BitNet with Episodic Memory) model trained for the BabyLM Challenge.

## Model Architecture
- Text Encoder Layers: {config_dict.get('text_encoder_layers', 3)}
- Text Decoder Layers: {config_dict.get('text_decoder_layers', 3)}
- Vision Latent Size: {config_dict.get('vision_latent_size', 64)}
- Memory Size: {config_dict.get('memory_size', 16)}
- Fusion Layers: {config_dict.get('fusion_num_layers', 1)}

## Evaluation

### Text-only Evaluation (2025 Pipeline)
```bash
# For text-only tasks (BLiMP, EWoK, GLUE)
./eval_zero_shot_fast.sh /path/to/model checkpoint_1M causal

# For fine-tuning tasks
./eval_finetuning.sh /path/to/model
```

### Multimodal Evaluation (2024 Pipeline)
```bash
# For Winoground and VQA
./eval_multimodal.sh /path/to/model

# For DevBench (requires model_type="bitmar")
./eval_devbench.sh /path/to/model bitmar
```

## Usage

```python
from transformers import AutoModel, AutoTokenizer
from src.hf_compatibility import BitMarProcessor, BitMarEvalModel

# Load model
model = AutoModel.from_pretrained("/path/to/model")
tokenizer = AutoTokenizer.from_pretrained("/path/to/model")

# For 2024 evaluation pipeline (DevBench)
processor = BitMarProcessor(tokenizer=tokenizer)
eval_model = BitMarEvalModel(model=model, processor=processor, device="cuda")
```
"""

    with open(save_dir / "README.md", "w") as f:
        f.write(model_card_content)

    logger.info(f"✅ BitMar model saved in HuggingFace format to: {save_dir}")
    logger.info(f"📋 Model card created with evaluation instructions")
    logger.info(
        f"🔧 Processor configuration saved for 2024 pipeline compatibility")

    return hf_model


def load_bitmar_from_hf_format(model_path: Union[str, Path]) -> BitMarForCausalLM:
    """
    Load BitMar model from HuggingFace format

    Args:
        model_path: Path to the saved model directory

    Returns:
        BitMarForCausalLM: Loaded model
    """
    model = BitMarForCausalLM.from_pretrained(model_path)
    return model
