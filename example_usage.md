# HuggingFace Model Saving Examples

The training script now supports automatic model saving to HuggingFace format with the following new features:

## Command Line Arguments

### Save After Each Epoch (Default: Enabled)
```bash
# Save model after each epoch (default behavior)
python train_100M_tokens.py --config configs/bitmar_100M_tokens.yaml

# Explicitly enable epoch saving
python train_100M_tokens.py --config configs/bitmar_100M_tokens.yaml --save_after_epochs

# Disable epoch saving
python train_100M_tokens.py --config configs/bitmar_100M_tokens.yaml --no_save_after_epochs
```

### Save After Specific Number of Steps (Optional)
```bash
# Save model every 1000 steps
python train_100M_tokens.py --config configs/bitmar_100M_tokens.yaml --save_steps 1000

# Save model every 500 steps
python train_100M_tokens.py --config configs/bitmar_100M_tokens.yaml --save_steps 500

# Save both after epochs AND every 2000 steps
python train_100M_tokens.py --config configs/bitmar_100M_tokens.yaml --save_after_epochs --save_steps 2000
```

### Custom Save Directory
```bash
# Save to custom directory
python train_100M_tokens.py --config configs/bitmar_100M_tokens.yaml --hf_save_dir ./my_models

# Save to absolute path
python train_100M_tokens.py --config configs/bitmar_100M_tokens.yaml --hf_save_dir "D:\BitMar\Models"
```

### Combined Examples
```bash
# Full control: Save every 1000 steps, after epochs, to custom directory
python train_100M_tokens.py \
    --config configs/bitmar_100M_tokens.yaml \
    --save_steps 1000 \
    --save_after_epochs \
    --hf_save_dir ./huggingface_models

# Training without evaluation interference, with regular saves
python train_100M_tokens.py \
    --config configs/bitmar_100M_tokens.yaml \
    --save_steps 2000 \
    --save_after_epochs \
    --hf_save_dir ./saved_models
```

## Output Structure

Models will be saved in the following structure:
```
./saved_models/  (or your custom directory)
├── checkpoint-epoch-1/
│   ├── pytorch_model.bin      # Model weights
│   ├── config.json           # Model configuration
│   ├── training_info.json    # Training metadata
│   └── tokenizer files...    # If tokenizer available
├── checkpoint-epoch-2/
├── checkpoint-step-1000/      # If --save_steps enabled
├── checkpoint-step-2000/
└── ...
```

## What Gets Saved

Each checkpoint includes:

1. **pytorch_model.bin** - Full model state dict
2. **config.json** - HuggingFace-compatible model configuration including:
   - Standard transformer config (vocab_size, hidden_size, etc.)
   - BitMar-specific config (episodic memory, cross-modal fusion)
   - Training progress (step, epoch, tokens processed)
3. **training_info.json** - Training metadata:
   - Global step and epoch
   - Total tokens processed
   - Model parameter counts
   - Save timestamp
4. **Tokenizer files** - If tokenizer is available

## Evaluation Changes

**Important**: Inline evaluation during training has been disabled to separate training and evaluation workflows.

- All `run_step_evaluation()`, `run_tiny_model_evaluation()`, and `run_benchmark_evaluation()` calls are commented out
- This allows training to run without interruption
- Run evaluation separately on saved checkpoints using dedicated evaluation scripts

## Loading Saved Models

The saved models can be loaded using standard HuggingFace methods:

```python
from transformers import AutoModel, AutoTokenizer

# Load model
model = AutoModel.from_pretrained("./saved_models/checkpoint-epoch-5")

# Load tokenizer (if saved)
tokenizer = AutoTokenizer.from_pretrained("./saved_models/checkpoint-epoch-5")

# Check training info
import json
with open("./saved_models/checkpoint-epoch-5/training_info.json") as f:
    training_info = json.load(f)
    print(f"Model trained for {training_info['total_tokens_processed']} tokens")
```
