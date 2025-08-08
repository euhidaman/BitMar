# BabyLM Evaluation Pipeline Guide

## Overview
This guide explains how BitMar models are evaluated using both the 2024 and 2025 BabyLM evaluation pipelines.

## Evaluation Data Structure
```
evaluation_data/
├── fast_eval/          # Quick evaluation (after each epoch)
│   ├── blimp_fast/
│   ├── entity_tracking_fast/
│   ├── supplement_fast/
│   ├── wug_adj_nominalization/
│   └── wug_past_tense/
└── full_eval/          # Comprehensive evaluation (final model)
    ├── blimp_filtered/         # Linguistic acceptability
    ├── cdi_childes/           # Age of acquisition
    ├── comps/                 # Conceptual knowledge
    ├── entity_tracking/       # Entity state tracking
    ├── glue_filtered/         # General language understanding
    ├── reading/               # Reading comprehension
    ├── supplement_filtered/   # Additional linguistic tasks
    ├── vqa_filtered/          # Visual question answering
    ├── winoground_filtered/   # Visual-linguistic reasoning
    ├── wug_adj_nominalization/ # Morphological generalization
    └── wug_past_tense/        # Past tense formation
```

## Evaluation Flow

### 1. Training with Automatic Evaluation
```bash
# Your standard training command now includes automatic evaluation
python3 train_100M_tokens.py --config configs/bitmar_100M_tokens.yaml --device cuda:0
```

**What happens during training:**

1. **After Each Epoch**: Fast evaluation runs automatically
   - Uses `fast_eval/` datasets (smaller, quicker)
   - Evaluates on both 2025 (text + multimodal) and 2024 (multimodal only) pipelines
   - Results saved to `evaluation_results/epoch_N/`

2. **At Training End**: Full evaluation runs automatically
   - Uses `full_eval/` datasets (complete evaluation)
   - Comprehensive evaluation on all tasks
   - Results saved to `evaluation_results/final/`

### 2. Manual Evaluation
```bash
# Evaluate any saved checkpoint
./run_evaluation.sh --model_path checkpoints_100M_dataset/checkpoint_epoch_1_tokens_50000000.pt

# Fast evaluation only
./run_evaluation.sh --model_path latest_checkpoint.pt --eval_type fast
```

## Pipeline-Specific Tasks

### 2025 Pipeline (Text + Multimodal)
**Text-only tasks:**
- **BLiMP**: Linguistic acceptability judgments
- **Entity Tracking**: Following entity states through narratives
- **Reading**: Human reading time correlation
- **GLUE**: Fine-tuning on classification tasks
- **Morphology**: WUG tasks for morphological generalization
- **CDI**: Age of acquisition evaluation
- **COMPS**: Conceptual knowledge testing

**Multimodal tasks:**
- **Winoground**: Visual-linguistic reasoning
- **VQA**: Visual question answering

### 2024 Pipeline (Multimodal Only)
**Multimodal tasks:**
- **Winoground**: Visual-linguistic reasoning  
- **VQA**: Visual question answering

## Model Conversion Process

The evaluation scripts automatically handle model conversion:

1. **BitMar Checkpoint** → **HuggingFace Format**
   - Converts PyTorch checkpoint to HF-compatible format
   - Creates config.json, tokenizer files, and model files
   - Saves converted model in `model_path/hf_model/`

2. **Evaluation Execution**
   - 2025 pipeline uses custom evaluation modules
   - 2024 pipeline uses `lm_eval` framework
   - Both pipelines handle multimodal data differently

## Evaluation Metrics

### Text Tasks
- **Accuracy**: For classification tasks
- **BLEU/ROUGE**: For generation tasks  
- **Perplexity**: For language modeling
- **Correlation**: For reading time tasks

### Multimodal Tasks
- **Accuracy**: For VQA and Winoground
- **Retrieval Metrics**: Image-text matching
- **Cross-modal Similarity**: Alignment quality

## Results Structure
```
evaluation_results/
├── epoch_0/
│   ├── 2025_results/    # Text + multimodal (2025 pipeline)
│   │   ├── blimp_fast_results.json
│   │   ├── winoground_fast_results.json
│   │   └── ...
│   └── 2024_results/    # Multimodal only (2024 pipeline)
│       ├── winoground_filtered_results.json
│       └── vqa_filtered_results.json
├── epoch_1/
│   ├── 2025_results/
│   └── 2024_results/
└── final/
    ├── 2025_results/    # Full evaluation with 2025 pipeline
    └── 2024_results/    # Full evaluation with 2024 pipeline
```

## Dependencies and Setup

### Required Dependencies
```bash
# Install evaluation pipeline dependencies
python setup_evaluation_pipelines.py

# Download evaluation data  
python download_evaluation_data.py
```

### Key Dependencies by Pipeline:

**2024 Pipeline:**
- `lm_eval` framework
- `transformers`
- `torch` 
- `accelerate`
- `minicons`

**2025 Pipeline:**
- `transformers`
- `datasets`
- `scikit-learn`
- `nltk`
- `statsmodels`
- `wandb`

## Troubleshooting

### Common Issues:

1. **HuggingFace Authentication**
   ```bash
   huggingface-cli login
   ```
   Required for Winoground and some VQA datasets.

2. **Missing Evaluation Data**
   ```bash
   python download_evaluation_data.py
   ```

3. **Import Errors**
   ```bash
   python setup_evaluation_pipelines.py
   ```

4. **Model Conversion Failures**
   - Check BitMar checkpoint format
   - Verify model state_dict keys
   - Ensure config is properly saved

### Evaluation Failures:
- **2025 Pipeline**: Check if evaluation_pipeline module imports correctly
- **2024 Pipeline**: Check if lm_eval imports correctly  
- **Multimodal Tasks**: Verify image datasets are accessible

## Performance Expectations

### Fast Evaluation (per epoch):
- **Runtime**: ~30-60 minutes per epoch
- **Tasks**: Subset of full evaluation
- **Purpose**: Monitor training progress

### Full Evaluation (final model):
- **Runtime**: ~2-4 hours
- **Tasks**: Complete evaluation suite
- **Purpose**: Final model assessment

## Integration with Training

The evaluation is seamlessly integrated with your existing training workflow:

```python
# Your existing command works unchanged
python3 train_100M_tokens.py --config configs/bitmar_100M_tokens.yaml --device cuda:0

# With step-based checkpoints for testing
python3 train_100M_tokens.py --config configs/bitmar_100M_tokens.yaml --device cuda:0 --save_every_n_steps 1000

# Disable evaluation if needed
python3 train_100M_tokens.py --config configs/bitmar_100M_tokens.yaml --device cuda:0 --disable_fast_eval --disable_full_eval
```

The system automatically:
- Detects available evaluation pipelines
- Converts models to compatible formats
- Runs appropriate evaluations
- Saves results with timestamps
- Logs success/failure to Weights & Biases
