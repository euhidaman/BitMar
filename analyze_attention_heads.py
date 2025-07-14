"""
Attention Head Analysis Utility for BitMar
Analyze saved attention heads and generate reports
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
import json
from typing import List, Tuple, Dict, Any
import pandas as pd

class AttentionHeadReportGenerator:
    """Generate reports from saved attention head data"""
    
    def __init__(self, analysis_dir: str):
        self.analysis_dir = Path(analysis_dir)
        self.reports_dir = self.analysis_dir / "reports"
        self.reports_dir.mkdir(exist_ok=True)
        
    def load_attention_heads(self, attention_type: str = 'encoder') -> Dict[int, List[Tuple[int, int]]]:
        """Load attention heads from all saved steps"""
        head_files = list(self.analysis_dir.glob(f"top_heads_{attention_type}_step_*.npy"))
        
        heads_by_step = {}
        for file in head_files:
            # Extract step number from filename
            step = int(file.stem.split('_')[-1])
            heads = np.load(file)
            heads_by_step[step] = [(int(layer), int(head)) for layer, head in heads]
            
        return heads_by_step
        
    def analyze_head_consistency(self, attention_type: str = 'encoder') -> Dict[str, Any]:
        """Analyze consistency of top heads across training"""
        heads_by_step = self.load_attention_heads(attention_type)
        
        if not heads_by_step:
            return {}
            
        # Find heads that appear consistently
        all_heads = set()
        for heads in heads_by_step.values():
            all_heads.update(heads)
            
        head_counts = {head: 0 for head in all_heads}
        for heads in heads_by_step.values():
            for head in heads:
                head_counts[head] += 1
                
        total_steps = len(heads_by_step)
        consistency_threshold = 0.5  # Head appears in >50% of steps
        
        consistent_heads = [
            head for head, count in head_counts.items() 
            if count / total_steps >= consistency_threshold
        ]
        
        return {
            'total_unique_heads': len(all_heads),
            'consistent_heads': consistent_heads,
            'consistency_scores': {str(head): count/total_steps for head, count in head_counts.items()},
            'steps_analyzed': list(heads_by_step.keys()),
            'attention_type': attention_type
        }
        
    def create_head_evolution_plot(self, attention_type: str = 'encoder'):
        """Create plot showing evolution of top heads over training"""
        heads_by_step = self.load_attention_heads(attention_type)
        
        if not heads_by_step:
            print(f"No attention head data found for {attention_type}")
            return
            
        # Convert to DataFrame for easier plotting
        data = []
        for step, heads in heads_by_step.items():
            for rank, (layer, head) in enumerate(heads[:10]):  # Top 10 only
                data.append({
                    'step': step,
                    'layer': layer,
                    'head': head,
                    'rank': rank + 1,
                    'head_id': f'L{layer}H{head}'
                })
                
        df = pd.DataFrame(data)
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot 1: Head rankings over time
        top_heads = df.groupby('head_id').size().nlargest(10).index
        for head_id in top_heads:
            head_data = df[df['head_id'] == head_id]
            ax1.plot(head_data['step'], head_data['rank'], 
                    marker='o', label=head_id, linewidth=2, markersize=4)
            
        ax1.set_title(f'Top {attention_type.title()} Attention Heads Ranking Over Training')
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Rank (1 = Most Important)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.invert_yaxis()  # Lower rank numbers at top
        
        # Plot 2: Layer distribution of top heads
        layer_counts = df.groupby(['step', 'layer']).size().unstack(fill_value=0)
        im = ax2.imshow(layer_counts.T, aspect='auto', cmap='viridis')
        ax2.set_title(f'{attention_type.title()} Attention: Number of Top Heads per Layer')
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Layer Index')
        ax2.set_xticks(range(len(layer_counts.index)))
        ax2.set_xticklabels(layer_counts.index)
        plt.colorbar(im, ax=ax2, label='Number of Top Heads')
        
        plt.tight_layout()
        
        # Save plot
        save_path = self.reports_dir / f"{attention_type}_head_evolution.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved head evolution plot to {save_path}")
        plt.close()
        
    def create_head_stability_heatmap(self, attention_type: str = 'encoder'):
        """Create heatmap showing head stability across layers"""
        analysis = self.analyze_head_consistency(attention_type)
        
        if not analysis:
            print(f"No consistency data found for {attention_type}")
            return
            
        # Create matrix of consistency scores
        consistency_scores = analysis['consistency_scores']
        
        # Parse head coordinates
        head_coords = []
        scores = []
        for head_str, score in consistency_scores.items():
            head = eval(head_str)  # Convert "(layer, head)" string back to tuple
            head_coords.append(head)
            scores.append(score)
            
        if not head_coords:
            return
            
        # Determine matrix dimensions
        max_layer = max(coord[0] for coord in head_coords) + 1
        max_head = max(coord[1] for coord in head_coords) + 1
        
        # Create consistency matrix
        consistency_matrix = np.zeros((max_layer, max_head))
        for (layer, head), score in zip(head_coords, scores):
            consistency_matrix[layer, head] = score
            
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(consistency_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        
        ax.set_title(f'{attention_type.title()} Attention Head Consistency\n(Fraction of training steps where head was in top-K)')
        ax.set_xlabel('Head Index')
        ax.set_ylabel('Layer Index')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Consistency Score')
        
        # Add text annotations for high-consistency heads
        for i in range(max_layer):
            for j in range(max_head):
                if consistency_matrix[i, j] > 0.3:  # Only show significant scores
                    text = ax.text(j, i, f'{consistency_matrix[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontsize=8)
                                 
        plt.tight_layout()
        
        # Save plot
        save_path = self.reports_dir / f"{attention_type}_head_stability.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved head stability heatmap to {save_path}")
        plt.close()
        
    def generate_text_report(self) -> str:
        """Generate comprehensive text report"""
        report_lines = ["# BitMar Attention Head Analysis Report\n"]
        
        for attention_type in ['encoder', 'decoder', 'cross_modal']:
            analysis = self.analyze_head_consistency(attention_type)
            
            if not analysis:
                continue
                
            report_lines.append(f"## {attention_type.title()} Attention Analysis\n")
            report_lines.append(f"- **Total unique heads tracked**: {analysis['total_unique_heads']}")
            report_lines.append(f"- **Consistently important heads**: {len(analysis['consistent_heads'])}")
            report_lines.append(f"- **Training steps analyzed**: {len(analysis['steps_analyzed'])}")
            
            if analysis['consistent_heads']:
                report_lines.append("\n### Most Consistent Heads:")
                for head in analysis['consistent_heads'][:10]:  # Top 10
                    layer, head_idx = head
                    consistency = analysis['consistency_scores'][str(head)]
                    report_lines.append(f"- Layer {layer}, Head {head_idx}: {consistency:.3f} consistency")
                    
            report_lines.append("\n")
            
        # Add summary
        report_lines.append("## Summary\n")
        report_lines.append("This report shows which attention heads are consistently important during BitMar training.")
        report_lines.append("Heads with high consistency scores focus on specific linguistic or cross-modal patterns.")
        report_lines.append("Use this information to understand model behavior and potentially prune less important heads.\n")
        
        return "\\n".join(report_lines)
        
    def save_analysis_json(self):
        """Save detailed analysis to JSON file"""
        full_analysis = {}
        
        for attention_type in ['encoder', 'decoder', 'cross_modal']:
            analysis = self.analyze_head_consistency(attention_type)
            if analysis:
                full_analysis[attention_type] = analysis
                
        if full_analysis:
            save_path = self.reports_dir / "attention_analysis.json"
            with open(save_path, 'w') as f:
                json.dump(full_analysis, f, indent=2)
            print(f"Saved detailed analysis to {save_path}")
            
    def create_all_reports(self):
        """Generate all analysis reports and visualizations"""
        print("Generating BitMar attention head analysis reports...")
        
        # Create visualizations
        for attention_type in ['encoder', 'decoder', 'cross_modal']:
            self.create_head_evolution_plot(attention_type)
            self.create_head_stability_heatmap(attention_type)
            
        # Generate text report
        text_report = self.generate_text_report()
        report_path = self.reports_dir / "attention_analysis_report.md"
        with open(report_path, 'w') as f:
            f.write(text_report)
        print(f"Saved text report to {report_path}")
        
        # Save JSON analysis
        self.save_analysis_json()
        
        print(f"All reports saved to {self.reports_dir}")

def main():
    parser = argparse.ArgumentParser(description="Analyze BitMar attention heads")
    parser.add_argument(
        "--analysis_dir", 
        type=str, 
        default="./attention_analysis",
        help="Directory containing saved attention head data"
    )
    parser.add_argument(
        "--attention_type",
        type=str,
        choices=['encoder', 'decoder', 'cross_modal', 'all'],
        default='all',
        help="Type of attention to analyze"
    )
    
    args = parser.parse_args()
    
    # Create report generator
    generator = AttentionHeadReportGenerator(args.analysis_dir)
    
    if args.attention_type == 'all':
        generator.create_all_reports()
    else:
        generator.create_head_evolution_plot(args.attention_type)
        generator.create_head_stability_heatmap(args.attention_type)
        
        analysis = generator.analyze_head_consistency(args.attention_type)
        print(f"\\nAnalysis for {args.attention_type}:")
        print(f"Total unique heads: {analysis.get('total_unique_heads', 0)}")
        print(f"Consistent heads: {len(analysis.get('consistent_heads', []))}")

if __name__ == "__main__":
    main()
