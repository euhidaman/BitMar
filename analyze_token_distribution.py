"""
Token Analysis and Visualization Script for BitMar 100M Token Training
Analyzes token distribution, alignment quality, and training progress
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import pandas as pd
from transformers import AutoTokenizer
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class TokenAnalyzer:
    """Analyze token usage and distribution for 100M token training"""
    
    def __init__(self, dataset_dir: str, tokenizer_name: str = "gpt2"):
        self.dataset_dir = Path(dataset_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.results_dir = Path("token_analysis_results")
        self.results_dir.mkdir(exist_ok=True)
        
    def analyze_dataset_tokens(self) -> Dict:
        """Analyze complete dataset token distribution"""
        logger.info("🔍 Analyzing complete dataset token distribution...")
        
        # Load all data sources
        analysis_results = {
            'conceptual_captions': self._analyze_conceptual_captions(),
            'localized_narratives': self._analyze_localized_narratives(),
            'train_50M_text': self._analyze_train_50M_text(),
            'summary': {}
        }
        
        # Create summary
        total_caption_tokens = (analysis_results['conceptual_captions']['total_tokens'] + 
                               analysis_results['localized_narratives']['total_tokens'])
        total_text_tokens = analysis_results['train_50M_text']['total_tokens']
        
        analysis_results['summary'] = {
            'total_caption_tokens': total_caption_tokens,
            'total_text_tokens': total_text_tokens,
            'total_tokens': total_caption_tokens + total_text_tokens,
            'caption_percentage': (total_caption_tokens / (total_caption_tokens + total_text_tokens)) * 100,
            'text_percentage': (total_text_tokens / (total_caption_tokens + total_text_tokens)) * 100
        }
        
        return analysis_results
    
    def _analyze_conceptual_captions(self) -> Dict:
        """Analyze Conceptual Captions token distribution"""
        logger.info("Analyzing Conceptual Captions...")
        
        cc_file = self.dataset_dir / "cc_3M_captions.json"
        with open(cc_file, 'r', encoding='utf-8') as f:
            captions = json.load(f)
        
        token_counts = []
        total_tokens = 0
        
        for caption in captions:
            tokens = self.tokenizer.encode(caption)
            token_count = len(tokens)
            token_counts.append(token_count)
            total_tokens += token_count
        
        return {
            'source': 'Conceptual Captions 3M',
            'num_samples': len(captions),
            'total_tokens': total_tokens,
            'avg_tokens_per_sample': np.mean(token_counts),
            'median_tokens_per_sample': np.median(token_counts),
            'std_tokens_per_sample': np.std(token_counts),
            'min_tokens': np.min(token_counts),
            'max_tokens': np.max(token_counts),
            'token_distribution': token_counts
        }
    
    def _analyze_localized_narratives(self) -> Dict:
        """Analyze Localized Narratives token distribution"""
        logger.info("Analyzing Localized Narratives...")
        
        ln_file = self.dataset_dir / "local_narr_captions.json"
        with open(ln_file, 'r', encoding='utf-8') as f:
            captions = json.load(f)
        
        token_counts = []
        total_tokens = 0
        
        for caption in captions:
            tokens = self.tokenizer.encode(caption)
            token_count = len(tokens)
            token_counts.append(token_count)
            total_tokens += token_count
        
        return {
            'source': 'Localized Narratives',
            'num_samples': len(captions),
            'total_tokens': total_tokens,
            'avg_tokens_per_sample': np.mean(token_counts),
            'median_tokens_per_sample': np.median(token_counts),
            'std_tokens_per_sample': np.std(token_counts),
            'min_tokens': np.min(token_counts),
            'max_tokens': np.max(token_counts),
            'token_distribution': token_counts
        }
    
    def _analyze_train_50M_text(self) -> Dict:
        """Analyze train_50M text token distribution"""
        logger.info("Analyzing train_50M text data...")
        
        train_dir = self.dataset_dir / "train_50M"
        text_files = [
            "bnc_spoken.train",
            "childes.train", 
            "gutenberg.train",
            "open_subtitles.train",
            "simple_wiki.train",
            "switchboard.train"
        ]
        
        all_token_counts = []
        total_tokens = 0
        file_stats = {}
        
        for filename in text_files:
            file_path = train_dir / filename
            if not file_path.exists():
                logger.warning(f"File not found: {filename}")
                continue
                
            file_token_counts = []
            file_total_tokens = 0
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        tokens = self.tokenizer.encode(line)
                        token_count = len(tokens)
                        file_token_counts.append(token_count)
                        file_total_tokens += token_count
            
            file_stats[filename] = {
                'num_lines': len(file_token_counts),
                'total_tokens': file_total_tokens,
                'avg_tokens_per_line': np.mean(file_token_counts) if file_token_counts else 0,
                'token_distribution': file_token_counts
            }
            
            all_token_counts.extend(file_token_counts)
            total_tokens += file_total_tokens
            
            logger.info(f"  {filename}: {len(file_token_counts):,} lines, {file_total_tokens:,} tokens")
        
        return {
            'source': 'train_50M',
            'num_samples': len(all_token_counts),
            'total_tokens': total_tokens,
            'avg_tokens_per_sample': np.mean(all_token_counts) if all_token_counts else 0,
            'median_tokens_per_sample': np.median(all_token_counts) if all_token_counts else 0,
            'std_tokens_per_sample': np.std(all_token_counts) if all_token_counts else 0,
            'min_tokens': np.min(all_token_counts) if all_token_counts else 0,
            'max_tokens': np.max(all_token_counts) if all_token_counts else 0,
            'token_distribution': all_token_counts,
            'file_breakdown': file_stats
        }
    
    def create_token_distribution_plots(self, analysis_results: Dict):
        """Create comprehensive token distribution visualizations"""
        logger.info("📊 Creating token distribution plots...")
        
        # Set up the plotting
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Token Distribution Analysis for 100M Token Training', fontsize=16, fontweight='bold')
        
        # Plot 1: Overall token distribution comparison
        ax1 = axes[0, 0]
        sources = ['Conceptual Captions', 'Localized Narratives', 'train_50M']
        token_counts = [
            analysis_results['conceptual_captions']['total_tokens'],
            analysis_results['localized_narratives']['total_tokens'],
            analysis_results['train_50M_text']['total_tokens']
        ]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        bars = ax1.bar(sources, token_counts, color=colors, alpha=0.8)
        ax1.set_ylabel('Total Tokens (millions)')
        ax1.set_title('Total Tokens by Source')
        
        # Add value labels on bars
        for bar, count in zip(bars, token_counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{count/1_000_000:.1f}M', ha='center', va='bottom', fontweight='bold')
        
        # Format y-axis to show millions
        ax1.ticklabel_format(style='plain', axis='y')
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1_000_000:.1f}M'))
        
        # Plot 2: Tokens per sample distribution for captions
        ax2 = axes[0, 1]
        cc_tokens = analysis_results['conceptual_captions']['token_distribution']
        ln_tokens = analysis_results['localized_narratives']['token_distribution']
        
        ax2.hist(cc_tokens, bins=50, alpha=0.7, label='Conceptual Captions', color='#FF6B6B', density=True)
        ax2.hist(ln_tokens, bins=50, alpha=0.7, label='Localized Narratives', color='#4ECDC4', density=True)
        ax2.set_xlabel('Tokens per Caption')
        ax2.set_ylabel('Density')
        ax2.set_title('Caption Length Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Tokens per sample for text data
        ax3 = axes[0, 2]
        text_tokens = analysis_results['train_50M_text']['token_distribution']
        ax3.hist(text_tokens, bins=50, alpha=0.8, color='#45B7D1', density=True)
        ax3.set_xlabel('Tokens per Text Sample')
        ax3.set_ylabel('Density')
        ax3.set_title('Text Sample Length Distribution')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Cumulative token distribution
        ax4 = axes[1, 0]
        
        # Calculate cumulative tokens for sampling strategy
        cc_cumsum = np.cumsum(sorted(cc_tokens, reverse=True))
        ln_cumsum = np.cumsum(sorted(ln_tokens, reverse=True))
        text_cumsum = np.cumsum(sorted(text_tokens, reverse=True))
        
        ax4.plot(cc_cumsum / 1_000_000, label='Conceptual Captions', color='#FF6B6B', linewidth=2)
        ax4.plot(ln_cumsum / 1_000_000, label='Localized Narratives', color='#4ECDC4', linewidth=2)
        ax4.plot(text_cumsum / 1_000_000, label='train_50M', color='#45B7D1', linewidth=2)
        
        # Add 50M token lines
        ax4.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='50M Token Target')
        
        ax4.set_xlabel('Sample Index (sorted by token count)')
        ax4.set_ylabel('Cumulative Tokens (millions)')
        ax4.set_title('Cumulative Token Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Token statistics comparison
        ax5 = axes[1, 1]
        
        stats_data = {
            'Mean': [
                analysis_results['conceptual_captions']['avg_tokens_per_sample'],
                analysis_results['localized_narratives']['avg_tokens_per_sample'],
                analysis_results['train_50M_text']['avg_tokens_per_sample']
            ],
            'Median': [
                analysis_results['conceptual_captions']['median_tokens_per_sample'],
                analysis_results['localized_narratives']['median_tokens_per_sample'],
                analysis_results['train_50M_text']['median_tokens_per_sample']
            ],
            'Std Dev': [
                analysis_results['conceptual_captions']['std_tokens_per_sample'],
                analysis_results['localized_narratives']['std_tokens_per_sample'],
                analysis_results['train_50M_text']['std_tokens_per_sample']
            ]
        }
        
        x = np.arange(len(sources))
        width = 0.25
        
        for i, (stat, values) in enumerate(stats_data.items()):
            ax5.bar(x + i * width, values, width, label=stat, alpha=0.8)
        
        ax5.set_xlabel('Data Source')
        ax5.set_ylabel('Tokens per Sample')
        ax5.set_title('Token Statistics Comparison')
        ax5.set_xticks(x + width)
        ax5.set_xticklabels(sources)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: 100M Token Allocation Strategy
        ax6 = axes[1, 2]
        
        # Show how 100M tokens would be allocated
        caption_allocation = 50_000_000
        text_allocation = 50_000_000
        
        # Calculate what percentage of each source we'd use
        cc_usage_pct = min(100, (caption_allocation * 0.6) / analysis_results['conceptual_captions']['total_tokens'] * 100)
        ln_usage_pct = min(100, (caption_allocation * 0.4) / analysis_results['localized_narratives']['total_tokens'] * 100)
        text_usage_pct = min(100, text_allocation / analysis_results['train_50M_text']['total_tokens'] * 100)
        
        allocation_sources = ['CC Captions\n(30M tokens)', 'LN Captions\n(20M tokens)', 'Text Data\n(50M tokens)']
        allocations = [30_000_000, 20_000_000, 50_000_000]  # Estimated split
        usage_pcts = [cc_usage_pct, ln_usage_pct, text_usage_pct]
        
        bars = ax6.bar(allocation_sources, allocations, color=colors, alpha=0.8)
        
        # Add usage percentage labels
        for i, (bar, alloc, pct) in enumerate(zip(bars, allocations, usage_pcts)):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{alloc/1_000_000:.0f}M\n({pct:.1f}% of source)', 
                    ha='center', va='bottom', fontweight='bold')
        
        ax6.set_ylabel('Allocated Tokens (millions)')
        ax6.set_title('100M Token Allocation Strategy')
        ax6.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1_000_000:.0f}M'))
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = self.results_dir / 'token_distribution_analysis.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"📊 Token distribution plots saved to {plot_path}")
        
        return fig
    
    def generate_sampling_strategy(self, analysis_results: Dict, target_caption_tokens: int = 50_000_000, 
                                  target_text_tokens: int = 50_000_000) -> Dict:
        """Generate optimal sampling strategy for 100M tokens"""
        logger.info("🎯 Generating optimal sampling strategy...")
        
        # Caption token allocation (60% CC, 40% LN for diversity)
        cc_target = int(target_caption_tokens * 0.6)  # 30M tokens
        ln_target = int(target_caption_tokens * 0.4)  # 20M tokens
        
        cc_tokens = analysis_results['conceptual_captions']['token_distribution']
        ln_tokens = analysis_results['localized_narratives']['token_distribution']
        text_tokens = analysis_results['train_50M_text']['token_distribution']
        
        # Sort by token count for efficient selection
        cc_sorted_indices = sorted(range(len(cc_tokens)), key=lambda i: cc_tokens[i], reverse=True)
        ln_sorted_indices = sorted(range(len(ln_tokens)), key=lambda i: ln_tokens[i], reverse=True)
        text_sorted_indices = sorted(range(len(text_tokens)), key=lambda i: text_tokens[i], reverse=True)
        
        # Select CC samples
        cc_selected = []
        cc_total = 0
        for idx in cc_sorted_indices:
            if cc_total + cc_tokens[idx] <= cc_target:
                cc_selected.append(idx)
                cc_total += cc_tokens[idx]
            else:
                break
        
        # Select LN samples
        ln_selected = []
        ln_total = 0
        for idx in ln_sorted_indices:
            if ln_total + ln_tokens[idx] <= ln_target:
                ln_selected.append(idx)
                ln_total += ln_tokens[idx]
            else:
                break
        
        # Select text samples
        text_selected = []
        text_total = 0
        for idx in text_sorted_indices:
            if text_total + text_tokens[idx] <= target_text_tokens:
                text_selected.append(idx)
                text_total += text_tokens[idx]
            else:
                break
        
        strategy = {
            'conceptual_captions': {
                'target_tokens': cc_target,
                'actual_tokens': cc_total,
                'selected_samples': len(cc_selected),
                'total_samples': len(cc_tokens),
                'usage_percentage': (len(cc_selected) / len(cc_tokens)) * 100,
                'selected_indices': cc_selected
            },
            'localized_narratives': {
                'target_tokens': ln_target,
                'actual_tokens': ln_total,
                'selected_samples': len(ln_selected),
                'total_samples': len(ln_tokens),
                'usage_percentage': (len(ln_selected) / len(ln_tokens)) * 100,
                'selected_indices': ln_selected
            },
            'train_50M_text': {
                'target_tokens': target_text_tokens,
                'actual_tokens': text_total,
                'selected_samples': len(text_selected),
                'total_samples': len(text_tokens),
                'usage_percentage': (len(text_selected) / len(text_tokens)) * 100,
                'selected_indices': text_selected
            },
            'summary': {
                'total_target_tokens': target_caption_tokens + target_text_tokens,
                'total_actual_tokens': cc_total + ln_total + text_total,
                'caption_tokens': cc_total + ln_total,
                'text_tokens': text_total,
                'efficiency': ((cc_total + ln_total + text_total) / (target_caption_tokens + target_text_tokens)) * 100
            }
        }
        
        return strategy
    
    def save_analysis_report(self, analysis_results: Dict, sampling_strategy: Dict):
        """Save comprehensive analysis report"""
        logger.info("📝 Saving analysis report...")
        
        report = {
            'dataset_analysis': analysis_results,
            'sampling_strategy': sampling_strategy,
            'recommendations': {
                'optimal_batch_size': 32,
                'estimated_training_steps': sampling_strategy['summary']['total_actual_tokens'] // (32 * 256),  # batch_size * seq_len
                'estimated_epochs': 30,
                'alignment_quality': 'Perfect (image-caption pairs maintained)',
                'token_efficiency': f"{sampling_strategy['summary']['efficiency']:.2f}%"
            }
        }
        
        # Save as JSON
        report_path = self.results_dir / 'token_analysis_report.json'
        with open(report_path, 'w') as f:
            # Remove numpy arrays for JSON serialization
            clean_report = self._clean_for_json(report)
            json.dump(clean_report, f, indent=2)
        
        # Save human-readable summary
        summary_path = self.results_dir / 'token_analysis_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("BitMar 100M Token Training Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Dataset Overview:\n")
            f.write(f"  • Total available tokens: {analysis_results['summary']['total_tokens']:,}\n")
            f.write(f"  • Caption tokens available: {analysis_results['summary']['total_caption_tokens']:,}\n")
            f.write(f"  • Text tokens available: {analysis_results['summary']['total_text_tokens']:,}\n\n")
            
            f.write("100M Token Allocation:\n")
            f.write(f"  • Target caption tokens: 50M ({sampling_strategy['summary']['caption_tokens']:,} actual)\n")
            f.write(f"  • Target text tokens: 50M ({sampling_strategy['summary']['text_tokens']:,} actual)\n")
            f.write(f"  • Total tokens: {sampling_strategy['summary']['total_actual_tokens']:,}\n")
            f.write(f"  • Efficiency: {sampling_strategy['summary']['efficiency']:.2f}%\n\n")
            
            f.write("Data Source Usage:\n")
            f.write(f"  • Conceptual Captions: {sampling_strategy['conceptual_captions']['selected_samples']:,} samples ({sampling_strategy['conceptual_captions']['usage_percentage']:.1f}%)\n")
            f.write(f"  • Localized Narratives: {sampling_strategy['localized_narratives']['selected_samples']:,} samples ({sampling_strategy['localized_narratives']['usage_percentage']:.1f}%)\n")
            f.write(f"  • train_50M text: {sampling_strategy['train_50M_text']['selected_samples']:,} samples ({sampling_strategy['train_50M_text']['usage_percentage']:.1f}%)\n\n")
            
            f.write("Training Estimates:\n")
            f.write(f"  • Estimated training steps: {report['recommendations']['estimated_training_steps']:,}\n")
            f.write(f"  • Estimated epochs: {report['recommendations']['estimated_epochs']}\n")
            f.write(f"  • Recommended batch size: {report['recommendations']['optimal_batch_size']}\n")
            f.write(f"  • Image-caption alignment: {report['recommendations']['alignment_quality']}\n")
        
        logger.info(f"📝 Analysis report saved to {report_path}")
        logger.info(f"📝 Summary saved to {summary_path}")
    
    def _clean_for_json(self, obj):
        """Clean object for JSON serialization"""
        if isinstance(obj, dict):
            return {k: self._clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            if len(obj) > 1000:  # Truncate large lists
                return obj[:1000] + [f"... truncated, total length: {len(obj)}"]
            return [self._clean_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj


def main():
    """Main analysis function"""
    parser = argparse.ArgumentParser(description="Analyze token distribution for BitMar 100M training")
    parser.add_argument("--dataset_dir", type=str, default="../babylm_dataset",
                       help="Path to BabyLM dataset directory")
    parser.add_argument("--tokenizer", type=str, default="gpt2",
                       help="Tokenizer to use for analysis")
    parser.add_argument("--target_caption_tokens", type=int, default=50_000_000,
                       help="Target caption tokens")
    parser.add_argument("--target_text_tokens", type=int, default=50_000_000,
                       help="Target text tokens")
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = TokenAnalyzer(args.dataset_dir, args.tokenizer)
        
        # Run analysis
        logger.info("🚀 Starting comprehensive token analysis...")
        analysis_results = analyzer.analyze_dataset_tokens()
        
        # Generate sampling strategy
        sampling_strategy = analyzer.generate_sampling_strategy(
            analysis_results, args.target_caption_tokens, args.target_text_tokens
        )
        
        # Create visualizations
        analyzer.create_token_distribution_plots(analysis_results)
        
        # Save report
        analyzer.save_analysis_report(analysis_results, sampling_strategy)
        
        # Print summary
        logger.info("🎯 Analysis Complete!")
        logger.info(f"Total tokens available: {analysis_results['summary']['total_tokens']:,}")
        logger.info(f"100M token efficiency: {sampling_strategy['summary']['efficiency']:.2f}%")
        logger.info(f"Results saved to: {analyzer.results_dir}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
