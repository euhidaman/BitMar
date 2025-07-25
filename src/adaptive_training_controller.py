"""
Adaptive Training Controller for BitMar
Monitors cross-modal similarity and automatically applies interventions
when performance drops suddenly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Deque
from collections import deque
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class AdaptiveTrainingController:
    """
    Monitors cross-modal similarity and automatically applies freezing/rebalancing
    when sudden drops are detected.
    """

    def __init__(
        self,
        similarity_window_size: int = 100,  # Steps to track for trend analysis
        drop_threshold: float = 0.15,       # Relative drop to trigger intervention
        min_steps_between_interventions: int = 1000,  # Cooldown period
        freeze_duration_steps: int = 2000,  # How long to freeze when triggered
        loss_rebalance_factor: float = 2.0, # How much to boost cross-modal loss
        similarity_smoothing_alpha: float = 0.1,  # EMA smoothing factor
        save_dir: Optional[str] = None
    ):
        self.similarity_window_size = similarity_window_size
        self.drop_threshold = drop_threshold
        self.min_steps_between_interventions = min_steps_between_interventions
        self.freeze_duration_steps = freeze_duration_steps
        self.loss_rebalance_factor = loss_rebalance_factor
        self.similarity_smoothing_alpha = similarity_smoothing_alpha
        self.save_dir = Path(save_dir) if save_dir else None

        # Tracking variables
        self.similarity_history: Deque[float] = deque(maxlen=similarity_window_size)
        self.similarity_ema: Optional[float] = None
        self.current_step = 0
        self.last_intervention_step = -min_steps_between_interventions

        # Intervention states
        self.text_encoder_frozen = False
        self.vision_encoder_frozen = False
        self.text_freeze_until_step = 0
        self.vision_freeze_until_step = 0
        self.loss_boost_active = False
        self.loss_boost_until_step = 0
        self.current_cross_modal_weight_multiplier = 1.0

        # Statistics
        self.interventions_log = []
        self.similarity_stats = {
            'min': float('inf'),
            'max': float('-inf'),
            'mean': 0.0,
            'std': 0.0
        }

    def update_similarity(
        self,
        similarity_score: float,
        step: int
    ) -> Dict[str, any]:
        """
        Update similarity tracker and check for interventions

        Args:
            similarity_score: Current cross-modal similarity score
            step: Current training step

        Returns:
            Dict with intervention decisions and statistics
        """
        self.current_step = step
        self.similarity_history.append(similarity_score)

        # Update EMA
        if self.similarity_ema is None:
            self.similarity_ema = similarity_score
        else:
            self.similarity_ema = (
                self.similarity_smoothing_alpha * similarity_score +
                (1 - self.similarity_smoothing_alpha) * self.similarity_ema
            )

        # Update statistics
        self._update_statistics()

        # Check for sudden drops and decide interventions
        intervention_info = self._check_for_interventions(similarity_score, step)

        # Update freezing states
        self._update_freezing_states(step)

        # Log intervention if it occurred
        if intervention_info['intervention_triggered']:
            self._log_intervention(intervention_info, step)

        return {
            'similarity_ema': self.similarity_ema,
            'text_encoder_frozen': self.text_encoder_frozen,
            'vision_encoder_frozen': self.vision_encoder_frozen,
            'cross_modal_weight_multiplier': self.current_cross_modal_weight_multiplier,
            'intervention_info': intervention_info,
            'similarity_stats': self.similarity_stats.copy()
        }

    def _check_for_interventions(
        self,
        current_similarity: float,
        step: int
    ) -> Dict[str, any]:
        """Check if interventions should be triggered"""

        intervention_info = {
            'intervention_triggered': False,
            'intervention_type': None,
            'trigger_reason': None,
            'similarity_drop': 0.0
        }

        # Need enough history and cooldown period
        if (len(self.similarity_history) < self.similarity_window_size or
            step - self.last_intervention_step < self.min_steps_between_interventions):
            return intervention_info

        # Calculate trend and recent performance
        recent_window = int(self.similarity_window_size * 0.3)  # Last 30% of window
        recent_similarities = list(self.similarity_history)[-recent_window:]
        older_similarities = list(self.similarity_history)[:-recent_window]

        recent_mean = np.mean(recent_similarities)
        older_mean = np.mean(older_similarities)

        # Calculate relative drop
        if older_mean > 0:
            relative_drop = (older_mean - recent_mean) / older_mean
        else:
            relative_drop = 0.0

        # Check if drop exceeds threshold
        if relative_drop > self.drop_threshold:
            intervention_info.update({
                'intervention_triggered': True,
                'similarity_drop': relative_drop,
                'trigger_reason': f'Similarity dropped by {relative_drop:.3f} (threshold: {self.drop_threshold})'
            })

            # Decide intervention strategy based on current state and drop severity
            intervention_type = self._decide_intervention_strategy(relative_drop, step)
            intervention_info['intervention_type'] = intervention_type

            # Apply the intervention
            self._apply_intervention(intervention_type, step)
            self.last_intervention_step = step

        return intervention_info

    def _decide_intervention_strategy(
        self,
        drop_severity: float,
        step: int
    ) -> str:
        """Decide which intervention to apply based on drop severity and training stage"""

        # Early training (< 50k steps): Focus on preventing one modality from dominating
        if step < 50000:
            if drop_severity > 0.25:  # Severe drop
                return "freeze_both_encoders"
            elif drop_severity > 0.20:
                return "freeze_vision_encoder"  # Vision often dominates early
            else:
                return "rebalance_loss_only"

        # Mid training (50k - 150k steps): More targeted interventions
        elif step < 150000:
            if drop_severity > 0.30:
                return "freeze_text_encoder"  # Text might be overfitting
            elif drop_severity > 0.20:
                return "rebalance_loss_heavy"
            else:
                return "rebalance_loss_only"

        # Late training (> 150k steps): Gentle interventions
        else:
            if drop_severity > 0.25:
                return "rebalance_loss_heavy"
            else:
                return "rebalance_loss_gentle"

    def _apply_intervention(self, intervention_type: str, step: int):
        """Apply the decided intervention"""

        if intervention_type == "freeze_both_encoders":
            self.text_encoder_frozen = True
            self.vision_encoder_frozen = True
            self.text_freeze_until_step = step + self.freeze_duration_steps
            self.vision_freeze_until_step = step + self.freeze_duration_steps
            self.current_cross_modal_weight_multiplier = self.loss_rebalance_factor
            self.loss_boost_until_step = step + self.freeze_duration_steps

        elif intervention_type == "freeze_vision_encoder":
            self.vision_encoder_frozen = True
            self.vision_freeze_until_step = step + self.freeze_duration_steps
            self.current_cross_modal_weight_multiplier = self.loss_rebalance_factor * 1.5
            self.loss_boost_until_step = step + self.freeze_duration_steps

        elif intervention_type == "freeze_text_encoder":
            self.text_encoder_frozen = True
            self.text_freeze_until_step = step + self.freeze_duration_steps
            self.current_cross_modal_weight_multiplier = self.loss_rebalance_factor
            self.loss_boost_until_step = step + self.freeze_duration_steps

        elif intervention_type == "rebalance_loss_heavy":
            self.current_cross_modal_weight_multiplier = self.loss_rebalance_factor * 2.0
            self.loss_boost_until_step = step + self.freeze_duration_steps // 2

        elif intervention_type == "rebalance_loss_gentle":
            self.current_cross_modal_weight_multiplier = self.loss_rebalance_factor * 1.2
            self.loss_boost_until_step = step + self.freeze_duration_steps // 3

        else:  # rebalance_loss_only
            self.current_cross_modal_weight_multiplier = self.loss_rebalance_factor
            self.loss_boost_until_step = step + self.freeze_duration_steps // 2

        self.loss_boost_active = True

        logger.info(f"Step {step}: Applied intervention '{intervention_type}' - "
                   f"Text frozen: {self.text_encoder_frozen}, "
                   f"Vision frozen: {self.vision_encoder_frozen}, "
                   f"Loss multiplier: {self.current_cross_modal_weight_multiplier:.2f}")

    def _update_freezing_states(self, step: int):
        """Update freezing states based on current step"""

        # Unfreeze text encoder if duration expired
        if self.text_encoder_frozen and step >= self.text_freeze_until_step:
            self.text_encoder_frozen = False
            logger.info(f"Step {step}: Unfroze text encoder")

        # Unfreeze vision encoder if duration expired
        if self.vision_encoder_frozen and step >= self.vision_freeze_until_step:
            self.vision_encoder_frozen = False
            logger.info(f"Step {step}: Unfroze vision encoder")

        # Disable loss boost if duration expired
        if self.loss_boost_active and step >= self.loss_boost_until_step:
            self.loss_boost_active = False
            self.current_cross_modal_weight_multiplier = 1.0
            logger.info(f"Step {step}: Disabled loss boost")

    def _update_statistics(self):
        """Update similarity statistics"""
        if len(self.similarity_history) == 0:
            return

        similarities = list(self.similarity_history)
        self.similarity_stats.update({
            'min': min(similarities),
            'max': max(similarities),
            'mean': np.mean(similarities),
            'std': np.std(similarities)
        })

    def _log_intervention(self, intervention_info: Dict, step: int):
        """Log intervention details"""
        log_entry = {
            'step': step,
            'timestamp': step,  # Could be replaced with actual timestamp
            'intervention_type': intervention_info['intervention_type'],
            'trigger_reason': intervention_info['trigger_reason'],
            'similarity_drop': intervention_info['similarity_drop'],
            'similarity_ema': self.similarity_ema,
            'similarity_stats': self.similarity_stats.copy()
        }

        self.interventions_log.append(log_entry)

        # Save to file if directory provided
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            log_file = self.save_dir / "adaptive_interventions.json"
            with open(log_file, 'w') as f:
                json.dump(self.interventions_log, f, indent=2)

    def get_encoder_freeze_states(self) -> Dict[str, bool]:
        """Get current encoder freezing states"""
        return {
            'freeze_text_encoder': self.text_encoder_frozen,
            'freeze_vision_encoder': self.vision_encoder_frozen
        }

    def get_loss_weight_multiplier(self) -> float:
        """Get current cross-modal loss weight multiplier"""
        return self.current_cross_modal_weight_multiplier

    def save_state(self, filepath: str):
        """Save controller state"""
        state = {
            'similarity_history': list(self.similarity_history),
            'similarity_ema': self.similarity_ema,
            'current_step': self.current_step,
            'last_intervention_step': self.last_intervention_step,
            'interventions_log': self.interventions_log,
            'similarity_stats': self.similarity_stats,
            'text_encoder_frozen': self.text_encoder_frozen,
            'vision_encoder_frozen': self.vision_encoder_frozen,
            'text_freeze_until_step': self.text_freeze_until_step,
            'vision_freeze_until_step': self.vision_freeze_until_step,
            'loss_boost_active': self.loss_boost_active,
            'loss_boost_until_step': self.loss_boost_until_step,
            'current_cross_modal_weight_multiplier': self.current_cross_modal_weight_multiplier
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

    def load_state(self, filepath: str):
        """Load controller state"""
        with open(filepath, 'r') as f:
            state = json.load(f)

        self.similarity_history = deque(state['similarity_history'], maxlen=self.similarity_window_size)
        self.similarity_ema = state['similarity_ema']
        self.current_step = state['current_step']
        self.last_intervention_step = state['last_intervention_step']
        self.interventions_log = state['interventions_log']
        self.similarity_stats = state['similarity_stats']
        self.text_encoder_frozen = state['text_encoder_frozen']
        self.vision_encoder_frozen = state['vision_encoder_frozen']
        self.text_freeze_until_step = state['text_freeze_until_step']
        self.vision_freeze_until_step = state['vision_freeze_until_step']
        self.loss_boost_active = state['loss_boost_active']
        self.loss_boost_until_step = state['loss_boost_until_step']
        self.current_cross_modal_weight_multiplier = state['current_cross_modal_weight_multiplier']


def compute_cross_modal_similarity(
    text_features: torch.Tensor,
    vision_features: torch.Tensor
) -> float:
    """
    Compute cross-modal similarity score for monitoring

    Args:
        text_features: [batch_size, text_dim] or [batch_size, seq_len, text_dim]
        vision_features: [batch_size, vision_dim]

    Returns:
        Average cosine similarity between text and vision features
    """
    # Pool text features if needed
    if text_features.dim() == 3:
        text_features = text_features.mean(dim=1)  # [batch_size, text_dim]

    # Normalize features
    text_norm = F.normalize(text_features, p=2, dim=1)
    vision_norm = F.normalize(vision_features, p=2, dim=1)

    # Compute cosine similarity
    similarity = torch.sum(text_norm * vision_norm, dim=1)  # [batch_size]

    return similarity.mean().item()
