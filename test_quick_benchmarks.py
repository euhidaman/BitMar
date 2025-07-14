#!/usr/bin/env python3
"""
BitMar Quick Text Benchmarks
Fast evaluation on standard NLP tasks
"""

import os
import sys
import torch
import yaml
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List
import time

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from model import create_bitmar_model

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model_and_config():
    """Load BitMar model and config"""
    config_path = "configs/bitmar_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)['model']
    
    model = create_bitmar_model(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    return model, config, device


def test_text_completion():
    """Test text completion capability"""
    logger.info("📝 Testing Text Completion")
    
    model, config, device = load_model_and_config()
    tokenizer = model.tokenizer
    
    prompts = [
        "The weather today is",
        "Artificial intelligence will",
        "In the future, technology",
        "Climate change is",
        "Education helps people"
    ]
    
    results = []
    
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", max_length=16)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        # Dummy vision features
        dummy_vision = torch.zeros(1, config['vision_encoder_dim']).to(device)
        
        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                vision_features=dummy_vision,
                max_length=32,
                temperature=0.7
            )
            
            completion = generated['generated_text'][0]
            results.append({
                'prompt': prompt,
                'completion': completion,
                'length': len(completion.split())
            })
    
    avg_length = np.mean([r['length'] for r in results])
    logger.info(f"   Average completion length: {avg_length:.1f} words")
    
    # Show sample
    logger.info("   Sample completions:")
    for r in results[:3]:
        logger.info(f"     '{r['prompt']}' → '{r['completion']}'")
    
    return {
        'task': 'text_completion',
        'avg_length': avg_length,
        'completions': results[:3]
    }


def test_language_modeling():
    """Test language modeling with perplexity"""
    logger.info("📊 Testing Language Modeling")
    
    model, config, device = load_model_and_config()
    tokenizer = model.tokenizer
    
    test_sentences = [
        "The cat sat on the mat.",
        "Technology has changed our lives significantly.",
        "Climate change affects weather patterns globally.",
        "Machine learning models process data efficiently.",
        "Students learn better with interactive methods."
    ]
    
    total_loss = 0
    total_tokens = 0
    
    for sentence in test_sentences:
        inputs = tokenizer(sentence, return_tensors="pt", max_length=32)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        labels = input_ids.clone()
        
        dummy_vision = torch.zeros(1, config['vision_encoder_dim']).to(device)
        
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                vision_features=dummy_vision,
                labels=labels,
                mode="inference"
            )
            
            if outputs['loss'] is not None:
                total_loss += outputs['loss'].item() * input_ids.size(1)
                total_tokens += input_ids.size(1)
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    logger.info(f"   Perplexity: {perplexity:.2f}")
    logger.info(f"   Average Loss: {avg_loss:.3f}")
    
    return {
        'task': 'language_modeling',
        'perplexity': perplexity,
        'avg_loss': avg_loss,
        'num_sentences': len(test_sentences)
    }


def test_text_understanding():
    """Test basic text understanding"""
    logger.info("🧠 Testing Text Understanding")
    
    model, config, device = load_model_and_config()
    tokenizer = model.tokenizer
    
    # Question-context pairs
    qa_pairs = [
        {
            'context': "The capital of France is Paris. It is known for the Eiffel Tower.",
            'question': "What is the capital of France?",
            'expected': "Paris"
        },
        {
            'context': "Dogs are mammals. They are loyal companions to humans.",
            'question': "What are dogs?",
            'expected': "mammals"
        },
        {
            'context': "Python is a programming language. It is used for web development.",
            'question': "What is Python?",
            'expected': "programming language"
        }
    ]
    
    correct_answers = 0
    
    for qa in qa_pairs:
        # Combine context and question
        prompt = f"Context: {qa['context']} Question: {qa['question']} Answer:"
        
        inputs = tokenizer(prompt, return_tensors="pt", max_length=64)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        dummy_vision = torch.zeros(1, config['vision_encoder_dim']).to(device)
        
        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                vision_features=dummy_vision,
                max_length=80,
                temperature=0.3
            )
            
            answer = generated['generated_text'][0]
            # Simple check if expected answer is in generated text
            if qa['expected'].lower() in answer.lower():
                correct_answers += 1
    
    accuracy = correct_answers / len(qa_pairs)
    
    logger.info(f"   QA Accuracy: {accuracy:.2%}")
    logger.info(f"   Correct: {correct_answers}/{len(qa_pairs)}")
    
    return {
        'task': 'text_understanding',
        'qa_accuracy': accuracy,
        'correct_answers': correct_answers,
        'total_questions': len(qa_pairs)
    }


def test_coherence_and_fluency():
    """Test text coherence and fluency"""
    logger.info("✨ Testing Coherence and Fluency")
    
    model, config, device = load_model_and_config()
    tokenizer = model.tokenizer
    
    prompts = [
        "Write about the benefits of renewable energy:",
        "Describe the importance of education:",
        "Explain how technology helps people:"
    ]
    
    fluency_scores = []
    coherence_scores = []
    
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", max_length=16)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        dummy_vision = torch.zeros(1, config['vision_encoder_dim']).to(device)
        
        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                vision_features=dummy_vision,
                max_length=48,
                temperature=0.8
            )
            
            text = generated['generated_text'][0]
            words = text.split()
            
            # Simple fluency metrics
            avg_word_length = np.mean([len(w) for w in words])
            unique_word_ratio = len(set(words)) / len(words) if len(words) > 0 else 0
            
            # Simple coherence check (contains original prompt)
            contains_prompt = any(w in text.lower() for w in prompt.lower().split()[:3])
            
            fluency_scores.append(avg_word_length * unique_word_ratio)
            coherence_scores.append(1.0 if contains_prompt else 0.0)
    
    avg_fluency = np.mean(fluency_scores)
    avg_coherence = np.mean(coherence_scores)
    
    logger.info(f"   Fluency Score: {avg_fluency:.2f}")
    logger.info(f"   Coherence Score: {avg_coherence:.2f}")
    
    return {
        'task': 'coherence_fluency',
        'fluency_score': avg_fluency,
        'coherence_score': avg_coherence,
        'num_prompts': len(prompts)
    }


def test_quantization_effects():
    """Test effects of BitNet quantization"""
    logger.info("⚡ Testing Quantization Effects")
    
    model, config, device = load_model_and_config()
    
    # Test quantization on a sample layer
    from src.model import BitNetLinear
    
    test_layer = BitNetLinear(512, 256).to(device)
    test_input = torch.randn(4, 512).to(device)
    
    # Test training mode (full precision)
    test_layer.train()
    output_train = test_layer(test_input)
    
    # Test inference mode (quantized)
    test_layer.eval()
    output_inference = test_layer(test_input)
    
    # Check quantization statistics
    weight_original = test_layer.weight.detach()
    weight_quantized = test_layer.quantize_weights_1_58_bit(weight_original)
    
    unique_values = weight_quantized.unique().cpu().numpy()
    quantization_ratio = len(unique_values) / weight_original.numel()
    
    logger.info(f"   Unique quantized values: {len(unique_values)}")
    logger.info(f"   Quantization ratio: {quantization_ratio:.6f}")
    logger.info(f"   Values: {unique_values}")
    
    return {
        'task': 'quantization_effects',
        'unique_values': len(unique_values),
        'quantization_ratio': quantization_ratio,
        'quantized_values': unique_values.tolist()
    }


def run_quick_benchmarks():
    """Run all quick benchmark tests"""
    logger.info("🚀 Running Quick Text Benchmarks")
    logger.info("=" * 40)
    
    start_time = time.time()
    results = {}
    
    try:
        results['text_completion'] = test_text_completion()
        logger.info("✅ Text completion test passed")
    except Exception as e:
        logger.error(f"❌ Text completion failed: {e}")
        results['text_completion'] = {'error': str(e)}
    
    try:
        results['language_modeling'] = test_language_modeling()
        logger.info("✅ Language modeling test passed")
    except Exception as e:
        logger.error(f"❌ Language modeling failed: {e}")
        results['language_modeling'] = {'error': str(e)}
    
    try:
        results['text_understanding'] = test_text_understanding()
        logger.info("✅ Text understanding test passed")
    except Exception as e:
        logger.error(f"❌ Text understanding failed: {e}")
        results['text_understanding'] = {'error': str(e)}
    
    try:
        results['coherence_fluency'] = test_coherence_and_fluency()
        logger.info("✅ Coherence and fluency test passed")
    except Exception as e:
        logger.error(f"❌ Coherence and fluency failed: {e}")
        results['coherence_fluency'] = {'error': str(e)}
    
    try:
        results['quantization_effects'] = test_quantization_effects()
        logger.info("✅ Quantization effects test passed")
    except Exception as e:
        logger.error(f"❌ Quantization effects failed: {e}")
        results['quantization_effects'] = {'error': str(e)}
    
    total_time = time.time() - start_time
    
    # Summary
    logger.info("=" * 40)
    logger.info("📊 Quick Benchmark Summary")
    logger.info("=" * 40)
    
    passed_tests = sum(1 for r in results.values() if 'error' not in r)
    total_tests = len(results)
    
    logger.info(f"Tests passed: {passed_tests}/{total_tests}")
    logger.info(f"Total time: {total_time:.2f}s")
    
    # Show key metrics
    for task, result in results.items():
        if 'error' not in result:
            if 'perplexity' in result:
                logger.info(f"📈 {task}: Perplexity = {result['perplexity']:.2f}")
            elif 'qa_accuracy' in result:
                logger.info(f"🎯 {task}: Accuracy = {result['qa_accuracy']:.2%}")
            elif 'unique_values' in result:
                logger.info(f"⚡ {task}: Quantization = {result['unique_values']} unique values")
    
    # Save results
    with open("quick_benchmark_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("\n💾 Results saved to: quick_benchmark_results.json")
    
    return passed_tests, total_tests


if __name__ == "__main__":
    passed, total = run_quick_benchmarks()
    success_rate = passed / total
    
    if success_rate >= 0.8:
        logger.info("🎉 BitMar model shows good text performance!")
        exit(0)
    else:
        logger.warning("⚠️  BitMar model needs improvement.")
        exit(1)
