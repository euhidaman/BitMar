# BitMar: Vision-Language Episodic Memory Transformer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

BitMar is a **Vision-Language Episodic Memory Transformer** that combines BitNet-quantized text processing, DiNOv2 vision encoding, and Larimar's episodic memory mechanism. The model maintains cross-modal episodic memories to improve zero-shot image-language understanding.

## 🏗️ Architecture

```text
Text Input → BitNet Text Encoder → Text Latent (768D)
                                        ↓
Vision Input → Quantized ViT → Vision Latent (768D)  
                                        ↓
                            Cross-Modal Fusion
                                        ↓
                            Multimodal Latent (768D)
                                        ↓
                         Episodic Memory (512 slots)
                                        ↓
                         BitNet Decoder → Generated Text
```

## 🌟 Key Features

- **BitNet Quantization**: 1.58-bit quantized text encoder/decoder for efficient inference
- **Vision Processing**: Quantized Vision Transformer using pre-computed DiNOv2 features
- **Episodic Memory**: Larimar-inspired memory mechanism for cross-modal associations
- **Attention Analysis**: Comprehensive attention head analysis for interpretability
- **BabyLM Optimized**: Trained within BabyLM constraints (10 epochs max)
- **Cloud-Ready**: Designed for RunPod GPU training with local CPU testing

## 📋 Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA-capable GPU (for training on RunPod)
- 16GB+ RAM recommended

## 🚀 Quick Start

### 1. Installation

```bash
git clone <your-repo-url>
cd BitMar
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. CPU Compatibility Test

```bash
python test_cpu_compatibility.py
```

### 3. Dataset Compatibility Test

```bash
python test_dataset_compatibility.py
```

### 4. Training (RunPod/GPU)

```bash
# Full training with wandb logging
python train_bitmar.py --config configs/bitmar_config.yaml

# Quick test training (1 epoch)
python train_bitmar.py --config configs/bitmar_config.yaml --max_epochs 1 --batch_size 4
```

## 📊 BabyLM Multimodal Dataset

BitMar is designed for the **BabyLM Challenge Multimodal Track**, using the official dataset structure:

### Dataset Components

**Text-only Data:**
- `train_50M.zip` - 50M tokens of text-only training data

**Image-Caption Pairs:**
- **Captions**: `cc_3M_captions.json` - Conceptual Captions 3M captions
- **Visual Features**: Precomputed DiNOv2 embeddings (768D)
  - `cc_3M_dino_v2_states_1of2.npy` - First half of visual embeddings
  - `cc_3M_dino_v2_states_2of2.npy` - Second half of visual embeddings

### Data Sources
- **Localized Narratives**: OpenImage + MSCOCO training sets
- **Conceptual Captions 3M**: Training split only
- **Visual Embeddings**: DiNOv2 ViT-Base model (`facebook/dinov2-base`)

### Dataset Setup

1. **Automatic Setup** (if you have the files):
```bash
python download_babylm_data.py
```

2. **Manual Download** (recommended):
   - Download the files to `../babylm_dataset/`
   - Ensure files match the expected structure

3. **Test Dataset** (for development):
```bash
python download_babylm_data.py --test
```

### Data Verification
```bash
python test_dataset_compatibility.py
```

## 📊 Model Architecture Details

### BitNet Text Processing
- **Encoder**: 1.58-bit quantized transformer (based on BitNet b1.58)
- **Decoder**: 1.58-bit quantized GPT-style decoder
- **Quantization**: Ternary weights {-1, 0, +1} with 8-bit activations

### Vision Processing
- **Input**: Pre-computed DiNOv2 features (768D)
- **Encoder**: Quantized Vision Transformer
- **Features**: 3M image-caption pairs from BabyLM dataset

### Episodic Memory System
- **Memory Size**: 512 episodic slots
- **Content**: Cross-modal (text + vision) episode embeddings
- **Access**: Attention-based retrieval during inference
- **Updates**: Gradient-based memory writing during training

### Attention Analysis
- **Head Selection**: Identifies most important attention heads
- **Cross-Modal Attention**: Tracks vision-to-text attention patterns
- **Memory Attention**: Analyzes episodic memory access patterns

## 🔧 Configuration

Key parameters in `configs/bitmar_config.yaml`:

```yaml
model:
  # Text processing
  text_encoder: "microsoft/bitnet-b1.58-large"
  text_decoder: "microsoft/bitnet-b1.58-large"
  text_latent_size: 768
  
  # Vision processing
  vision_encoder_dim: 768
  vision_latent_size: 768
  vision_quantization: true
  
  # Memory system
  memory_size: 512
  episode_dim: 768
  memory_alpha: 0.1
  
  # Training
  learning_rate: 1e-4
  batch_size: 16
  max_epochs: 10
  gradient_clip: 1.0

# WandB logging
wandb:
  project: "bitmar-babylm"
  entity: "babylm-ntust-org"
  log_attention: true
  log_memory: true
```

## 📈 Training Details

### BabyLM Dataset
- **Text**: 3M captions from CC3M dataset
- **Vision**: Pre-computed DiNOv2 features (768D)
- **Pairing**: Image-caption pairs for multimodal training

### Training Strategy
1. **Phase 1**: Text-only pretraining (2 epochs)
2. **Phase 2**: Vision-text alignment (3 epochs)
3. **Phase 3**: Episodic memory training (5 epochs)

### Memory Training
- **Write Phase**: Store cross-modal episodes during training
- **Read Phase**: Retrieve relevant episodes for generation
- **Forgetting**: LRU-based memory slot replacement

## 🔍 Attention Analysis Features

### Important Attention Heads
- **Cross-Modal Heads**: Vision-to-text attention patterns
- **Memory Heads**: Episodic memory access patterns
- **Generation Heads**: Text generation attention patterns

### Visualization
- Attention heatmaps for cross-modal understanding
- Memory access patterns over time
- Head importance rankings

## 💾 Model Outputs

### Training Artifacts
- `checkpoints/`: Model checkpoints (best and last)
- `logs/`: Training logs and metrics
- `attention/`: Attention analysis results
- `memory/`: Episodic memory visualizations

### Evaluation Metrics
- **Cross-Modal Retrieval**: Image-text matching accuracy
- **Generation Quality**: BLEU, ROUGE scores
- **Memory Efficiency**: Memory access patterns
- **Quantization Quality**: BitNet compression metrics

## 🌐 RunPod Training Guide

### Setup on RunPod
1. **Instance**: RTX 4090 or A100 GPU pod
2. **Image**: PyTorch 2.0+ with CUDA 11.8+
3. **Storage**: 50GB+ for dataset and checkpoints

### Training Commands
```bash
# Clone and setup
git clone <your-repo-url>
cd BitMar
pip install -r requirements.txt

# Download dataset (if not present)
python download_babylm_data.py

# Start training with wandb
export WANDB_API_KEY="5fba3726e4e32540d9fcba403f880dfaad983051"
python train_bitmar.py --config configs/bitmar_config.yaml
```

## 📚 Research Context

### Motivation
Grounding language in vision requires linking to past visual experiences. Traditional approaches lack persistent memory of visual-text associations. BitMar addresses this through:

1. **Episodic Memory**: Concrete visual-text associations storage
2. **Quantization**: Efficient inference through BitNet compression
3. **Cross-Modal Understanding**: Joint vision-language reasoning

### Benefits
- **Efficiency**: 1.58-bit quantization enables local deployment
- **Interpretability**: Attention analysis reveals reasoning patterns
- **Adaptability**: One-shot episodic memory updates
- **Performance**: Strong zero-shot image-language tasks

### Cognitive Alignment
The episodic memory mechanism aligns with cognitive theories of grounding, where language understanding relies on recalled sensory experiences.

## 🎯 Evaluation Tasks

### Zero-Shot Capabilities
- **Visual Question Answering**: Answer questions about images
- **Image Captioning**: Generate descriptions for novel images
- **Cross-Modal Retrieval**: Find relevant images for text queries

### Memory Analysis
- **Episodic Recall**: Retrieve similar visual experiences
- **Memory Efficiency**: Utilization of memory slots
- **Forgetting Patterns**: Memory replacement strategies

## 🔬 Technical Implementation

### Quantization Details
- **Weights**: Ternary quantization {-1, 0, +1}
- **Activations**: 8-bit integer quantization
- **Gradients**: Full precision during training

### Memory Implementation
- **Storage**: Key-value memory with attention-based access
- **Updates**: Gradient-based memory slot updates
- **Retrieval**: Soft attention over memory slots

## 🐛 Troubleshooting

### Common Issues
1. **CUDA OOM**: Reduce batch size in config
2. **Dataset Missing**: Run `download_babylm_data.py`
3. **Quantization Errors**: Check PyTorch version compatibility

### CPU Testing
- Use `test_cpu_compatibility.py` for local testing
- Reduced model size for CPU inference
- Memory-efficient attention computation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **BitNet**: Microsoft Research for 1-bit LLM quantization
- **Larimar**: IBM Research for episodic memory mechanisms
- **DiNOv2**: Meta AI for self-supervised vision features
- **BabyLM**: EMNLP 2024 challenge for multimodal datasets

## 📞 Contact

For questions about this implementation, please open an issue or contact the development team.

---

**Note**: This model is designed for research purposes and BabyLM challenge participation. For production use, additional optimization and validation may be required.
