"""
Human-Inspired Visual-First Learning for BitMar
===============================================

This implements a 3-stage training approach that mimics how humans learn:
1. Visual Understanding First (like babies recognizing objects)
2. Visual-Language Grounding (connecting words to what they see)
3. Abstract Language Learning (pure text understanding)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import random

logger = logging.getLogger(__name__)


class HumanInspiredTrainingPipeline:
    """
    3-Stage Human-Inspired Learning Pipeline:
    Stage 1: Visual Understanding (like babies learning to see)
    Stage 2: Visual-Language Grounding (connecting words to images)
    Stage 3: Abstract Language Learning (pure text reasoning)
    """
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.current_stage = 1
        
        # Stage-specific learning configurations
        self.stage_configs = {
            1: {  # Visual Understanding Stage
                'name': 'Visual Understanding',
                'description': 'Learn visual representations without language pressure',
                'epochs': config.get('visual_stage_epochs', 5),
                'learning_rate': config.get('visual_stage_lr', 0.001),
                'focus': 'vision_encoder_only',
                'loss_weights': {'vision_reconstruction': 1.0, 'vision_clustering': 0.5}
            },
            2: {  # Visual-Language Grounding Stage  
                'name': 'Visual-Language Grounding',
                'description': 'Connect words to visual concepts (like children learning "apple")',
                'epochs': config.get('grounding_stage_epochs', 10),
                'learning_rate': config.get('grounding_stage_lr', 0.0005),
                'focus': 'multimodal_alignment',
                'loss_weights': {
                    'cross_modal_contrastive': 1.0,
                    'visual_language_alignment': 0.8,
                    'caption_generation': 0.3
                }
            },
            3: {  # Abstract Language Learning Stage
                'name': 'Abstract Language Learning', 
                'description': 'Learn pure language patterns (like reading books)',
                'epochs': config.get('language_stage_epochs', 15),
                'learning_rate': config.get('language_stage_lr', 0.0003),
                'focus': 'language_modeling',
                'loss_weights': {'language_modeling': 1.0, 'text_understanding': 0.5}
            }
        }
    
    def train_stage_1_visual_understanding(self, visual_dataloader):
        """
        Stage 1: Visual Understanding (Human-like: babies learn to see first)
        
        What happens in human development:
        - Babies spend months just looking at objects
        - They learn to distinguish shapes, colors, textures
        - No pressure to understand language yet
        
        What we do:
        - Train only the vision encoder
        - Learn good visual representations
        - Use unsupervised objectives like reconstruction, clustering
        """
        logger.info("🍼 STAGE 1: Visual Understanding (like babies learning to see)")
        
        # Freeze everything except vision encoder
        self._freeze_all_except_vision()
        
        # Visual learning objectives
        vision_reconstruction_loss = nn.MSELoss()
        
        for epoch in range(self.stage_configs[1]['epochs']):
            epoch_loss = 0
            
            for batch_idx, batch in enumerate(visual_dataloader):
                vision_features = batch['vision_features']  # [B, 768] DiNOv2 features
                
                # Forward through vision encoder
                encoded_vision = self.model.encode_vision(vision_features)  # [B, 32]
                
                # Reconstruction objective (learn to reconstruct original features)
                reconstructed = self.model.vision_reconstruction_head(encoded_vision)
                recon_loss = vision_reconstruction_loss(reconstructed, vision_features)
                
                # Visual clustering objective (group similar concepts)
                cluster_loss = self._compute_visual_clustering_loss(encoded_vision)
                
                # Combined loss
                total_loss = recon_loss + 0.5 * cluster_loss
                
                # Backward pass
                total_loss.backward()
                
                epoch_loss += total_loss.item()
                
                if batch_idx % 100 == 0:
                    logger.info(f"Stage 1 - Epoch {epoch}, Batch {batch_idx}: "
                              f"Recon Loss: {recon_loss.item():.4f}, "
                              f"Cluster Loss: {cluster_loss.item():.4f}")
            
            logger.info(f"Stage 1 - Epoch {epoch} completed. Avg Loss: {epoch_loss/len(visual_dataloader):.4f}")
            
        logger.info("✅ Stage 1 Complete: Vision encoder learned to see!")
    
    def train_stage_2_visual_language_grounding(self, multimodal_dataloader):
        """
        Stage 2: Visual-Language Grounding (Human-like: connecting words to images)
        
        What happens in human development:
        - Child sees an apple and hears "apple"
        - They learn that the sound "apple" refers to the red round object
        - This creates strong visual-language associations
        
        What we do:
        - Use the pre-trained vision encoder from Stage 1
        - Train QFormer to connect text and visual concepts
        - Focus on contrastive learning (align "red car" text with red car image)
        """
        logger.info("👶 STAGE 2: Visual-Language Grounding (like children learning 'apple')")
        
        # Unfreeze QFormer and text components, keep vision partially frozen
        self._setup_for_multimodal_learning()
        
        for epoch in range(self.stage_configs[2]['epochs']):
            epoch_loss = 0
            
            for batch_idx, batch in enumerate(multimodal_dataloader):
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                vision_features = batch['vision_features']
                
                # Forward pass through the model
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask, 
                    vision_features=vision_features,
                    mode="grounding"  # Special mode for stage 2
                )
                
                # Visual-Language Alignment Losses
                
                # 1. Contrastive Loss (like CLIP): "red car" text should match red car image
                text_features = outputs['text_features'].mean(dim=1)  # Pool text
                vision_features_encoded = outputs['vision_latent']
                contrastive_loss = self._compute_contrastive_loss(text_features, vision_features_encoded)
                
                # 2. QFormer Alignment Loss: Query tokens should capture both modalities
                query_features = outputs.get('query_tokens', None)
                if query_features is not None:
                    alignment_loss = self._compute_qformer_alignment_loss(
                        query_features, text_features, vision_features_encoded
                    )
                else:
                    alignment_loss = torch.tensor(0.0)
                
                # 3. Simple Caption Generation (to encourage understanding)
                if 'labels' in batch:
                    caption_loss = outputs.get('loss', torch.tensor(0.0))
                else:
                    caption_loss = torch.tensor(0.0)
                
                # Weighted combination
                weights = self.stage_configs[2]['loss_weights']
                total_loss = (
                    weights['cross_modal_contrastive'] * contrastive_loss +
                    weights['visual_language_alignment'] * alignment_loss +
                    weights['caption_generation'] * caption_loss
                )
                
                total_loss.backward()
                epoch_loss += total_loss.item()
                
                if batch_idx % 100 == 0:
                    logger.info(f"Stage 2 - Epoch {epoch}, Batch {batch_idx}: "
                              f"Contrastive: {contrastive_loss.item():.4f}, "
                              f"Alignment: {alignment_loss.item():.4f}, "
                              f"Caption: {caption_loss.item():.4f}")
            
            logger.info(f"Stage 2 - Epoch {epoch} completed. Avg Loss: {epoch_loss/len(multimodal_dataloader):.4f}")
        
        logger.info("✅ Stage 2 Complete: Model learned to connect words with images!")
    
    def train_stage_3_abstract_language(self, text_only_dataloader):
        """
        Stage 3: Abstract Language Learning (Human-like: reading and abstract thinking)
        
        What happens in human development:
        - After grounding language in visual experience
        - Humans can learn abstract concepts through pure text
        - They can read books and understand concepts they've never seen
        
        What we do:
        - Use the train_50M dataset (pure text)
        - Train advanced language understanding
        - Model can now handle abstract concepts beyond visual experience
        """
        logger.info("🎓 STAGE 3: Abstract Language Learning (like reading books)")
        
        # Setup for language-only learning
        self._setup_for_language_learning()
        
        for epoch in range(self.stage_configs[3]['epochs']):
            epoch_loss = 0
            
            for batch_idx, batch in enumerate(text_only_dataloader):
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch.get('labels', input_ids)
                
                # Forward pass - text only
                outputs = self.model.text_decoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                # Language modeling loss
                lm_loss = outputs['loss']
                
                # Additional text understanding objectives
                text_understanding_loss = self._compute_text_understanding_loss(outputs)
                
                # Combined loss
                weights = self.stage_configs[3]['loss_weights']
                total_loss = (
                    weights['language_modeling'] * lm_loss +
                    weights['text_understanding'] * text_understanding_loss
                )
                
                total_loss.backward()
                epoch_loss += total_loss.item()
                
                if batch_idx % 100 == 0:
                    logger.info(f"Stage 3 - Epoch {epoch}, Batch {batch_idx}: "
                              f"LM Loss: {lm_loss.item():.4f}, "
                              f"Understanding Loss: {text_understanding_loss.item():.4f}")
            
            logger.info(f"Stage 3 - Epoch {epoch} completed. Avg Loss: {epoch_loss/len(text_only_dataloader):.4f}")
        
        logger.info("✅ Stage 3 Complete: Model learned abstract language reasoning!")
    
    def _freeze_all_except_vision(self):
        """Freeze everything except vision encoder for Stage 1"""
        for name, param in self.model.named_parameters():
            if 'vision_encoder' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        # Add reconstruction head if not exists
        if not hasattr(self.model, 'vision_reconstruction_head'):
            self.model.vision_reconstruction_head = nn.Linear(
                self.model.config['vision_latent_size'], 
                self.model.config['vision_encoder_dim']
            )
    
    def _setup_for_multimodal_learning(self):
        """Setup parameters for Stage 2 multimodal learning"""
        for name, param in self.model.named_parameters():
            if any(component in name for component in ['fusion', 'qformer', 'text_encoder']):
                param.requires_grad = True
            elif 'vision_encoder' in name:
                param.requires_grad = False  # Keep vision frozen from Stage 1
            else:
                param.requires_grad = True
    
    def _setup_for_language_learning(self):
        """Setup parameters for Stage 3 language learning"""
        for name, param in self.model.named_parameters():
            if any(component in name for component in ['text_encoder', 'text_decoder']):
                param.requires_grad = True
            else:
                param.requires_grad = False  # Freeze multimodal components
    
    def _compute_visual_clustering_loss(self, vision_features):
        """Encourage similar visual concepts to cluster together"""
        # Simple clustering objective using cosine similarity
        normalized_features = F.normalize(vision_features, dim=1)
        similarity_matrix = torch.matmul(normalized_features, normalized_features.T)
        
        # Encourage high similarity between similar samples
        # This is a simplified version - could use more sophisticated clustering
        diagonal_mask = torch.eye(similarity_matrix.size(0), device=similarity_matrix.device)
        off_diagonal = similarity_matrix * (1 - diagonal_mask)
        
        # Minimize variance in similarities (encourage clustering)
        cluster_loss = torch.var(off_diagonal)
        return cluster_loss
    
    def _compute_contrastive_loss(self, text_features, vision_features, temperature=0.07):
        """CLIP-style contrastive loss for text-image alignment"""
        # Normalize features
        text_features = F.normalize(text_features, dim=-1)
        vision_features = F.normalize(vision_features, dim=-1)
        
        # Compute similarity matrix
        logits = torch.matmul(text_features, vision_features.T) / temperature
        
        # Create labels (positive pairs on diagonal)
        batch_size = text_features.shape[0]
        labels = torch.arange(batch_size, device=logits.device)
        
        # Symmetric loss
        text_to_vision_loss = F.cross_entropy(logits, labels)
        vision_to_text_loss = F.cross_entropy(logits.T, labels)
        
        return (text_to_vision_loss + vision_to_text_loss) / 2
    
    def _compute_qformer_alignment_loss(self, query_features, text_features, vision_features):
        """Ensure QFormer query tokens align with both text and vision"""
        # Query tokens should be similar to both text and vision representations
        query_pooled = query_features.mean(dim=1)  # Pool query tokens
        
        # Alignment with text
        text_alignment = F.mse_loss(query_pooled, text_features)
        
        # Alignment with vision  
        vision_alignment = F.mse_loss(query_pooled, vision_features)
        
        return (text_alignment + vision_alignment) / 2
    
    def _compute_text_understanding_loss(self, outputs):
        """Additional objectives for better text understanding in Stage 3"""
        # Could include objectives like:
        # - Sentence coherence
        # - Grammatical correctness
        # - Semantic consistency
        # For now, return zero (can be extended)
        return torch.tensor(0.0, device=next(self.model.parameters()).device)


class VisualFirstDataset(Dataset):
    """Dataset for Stage 1: Visual-only learning"""
    
    def __init__(self, vision_features_path):
        self.vision_features = np.load(vision_features_path, mmap_mode='r')
    
    def __len__(self):
        return len(self.vision_features)
    
    def __getitem__(self, idx):
        return {
            'vision_features': torch.FloatTensor(self.vision_features[idx])
        }


class MultimodalGroundingDataset(Dataset):
    """Dataset for Stage 2: Visual-Language grounding"""
    
    def __init__(self, captions_path, vision_features_path, tokenizer, max_length=256):
        with open(captions_path, 'r') as f:
            self.captions = json.load(f)
        self.vision_features = np.load(vision_features_path, mmap_mode='r')
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return min(len(self.captions), len(self.vision_features))
    
    def __getitem__(self, idx):
        caption = self.captions[idx]['caption'] if isinstance(self.captions[idx], dict) else self.captions[idx]
        
        # Tokenize caption
        encoded = self.tokenizer(
            caption,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'vision_features': torch.FloatTensor(self.vision_features[idx]),
            'labels': encoded['input_ids'].squeeze()  # For caption generation
        }


class TextOnlyDataset(Dataset):
    """Dataset for Stage 3: Pure text learning from train_50M"""
    
    def __init__(self, text_files, tokenizer, max_length=256):
        self.texts = []
        for text_file in text_files:
            with open(text_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                self.texts.extend([line.strip() for line in lines if line.strip()])
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        logger.info(f"Loaded {len(self.texts)} text samples for Stage 3")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length', 
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'labels': encoded['input_ids'].squeeze()
        }


def create_human_inspired_dataloaders(config):
    """Create dataloaders for all 3 stages of human-inspired learning"""
    
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    dataset_dir = Path(config.get('dataset_dir', '../babylm_dataset'))

    # Stage 1: Visual-only datasets
    cc_visual_dataset = VisualFirstDataset(dataset_dir / 'cc_3M_dino_v2_states_1of2.npy')
    ln_visual_dataset = VisualFirstDataset(dataset_dir / 'local_narr_dino_v2_states.npy')

    visual_dataloader = DataLoader(
        torch.utils.data.ConcatDataset([cc_visual_dataset, ln_visual_dataset]),
        batch_size=config.get('batch_size', 32),
        shuffle=True,
        num_workers=4
    )
    
    # Stage 2: Multimodal datasets
    cc_multimodal = MultimodalGroundingDataset(
        dataset_dir / 'cc_3M_captions.json',
        dataset_dir / 'cc_3M_dino_v2_states_1of2.npy',  # Use first file for now
        tokenizer
    )

    ln_multimodal = MultimodalGroundingDataset(
        dataset_dir / 'local_narr_captions.json',
        dataset_dir / 'local_narr_dino_v2_states.npy',
        tokenizer
    )

    multimodal_dataloader = DataLoader(
        torch.utils.data.ConcatDataset([cc_multimodal, ln_multimodal]),
        batch_size=config.get('batch_size', 32),
        shuffle=True,
        num_workers=4
    )
    
    # Stage 3: Text-only dataset from train_50M (use extracted files)
    train_50m_dir = dataset_dir / 'train_50M'
    if not train_50m_dir.exists():
        raise FileNotFoundError(f"train_50M directory not found at {train_50m_dir}. Please run with --extract_only first.")

    text_files = [
        train_50m_dir / 'childes.train',
        train_50m_dir / 'gutenberg.train',
        train_50m_dir / 'open_subtitles.train',
        train_50m_dir / 'simple_wiki.train',
        train_50m_dir / 'bnc_spoken.train',
        train_50m_dir / 'switchboard.train'
    ]

    # Verify all text files exist
    missing_files = [f for f in text_files if not f.exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing train_50M files: {missing_files}")

    text_dataset = TextOnlyDataset(text_files, tokenizer)
    text_dataloader = DataLoader(
        text_dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=True,
        num_workers=4
    )

    logger.info(f"📊 Dataset Statistics:")
    logger.info(f"   Stage 1 (Visual): {len(visual_dataloader.dataset):,} samples")
    logger.info(f"   Stage 2 (Multimodal): {len(multimodal_dataloader.dataset):,} samples")
    logger.info(f"   Stage 3 (Text-only): {len(text_dataloader.dataset):,} samples")

    return visual_dataloader, multimodal_dataloader, text_dataloader


if __name__ == "__main__":
    # Example usage
    config = {
        'visual_stage_epochs': 5,
        'grounding_stage_epochs': 10,
        'language_stage_epochs': 15,
        'batch_size': 32
    }

    # This would be integrated into your main training script
    print("🧠 Human-Inspired Learning Pipeline Created!")
    print("Stage 1: Visual Understanding (like babies learning to see)")
    print("Stage 2: Visual-Language Grounding (connecting words to images)")
    print("Stage 3: Abstract Language Learning (reading and reasoning)")
