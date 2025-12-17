"""
Visualization Suite for Tracking Data Analysis
==============================================

Football-first visualizations designed for non-technical stakeholders.
Every chart tells a story about what matters on the field.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec

# Professional styling
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'


class TrackingVisualizer:

    # Creates Visualizations

    def __init__(self, analyzer):
        # Initialize with analyzer object containing data and results.
        self.analyzer = analyzer
        self.df = analyzer.df
        self.colors = {
            'primary': '#1f77b4',
            'success': '#2ca02c', 
            'warning': '#ff7f0e',
            'danger': '#d62728',
            'neutral': '#7f7f7f',
            'elite': '#9467bd'
        }
    
    def plot_model_comparison(self, combine_r2=-0.155, save_path='model_comparison.png'):
        
        # Side-by-side comparison showing tracking beats combine.
        
        # Football Context:
        # This single chart tells the entire story - combine testing fails to predict
        # performance while tracking data captures what matters.

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Get tracking model results
        tracking_r2 = self.analyzer.results['metrics']['test_r2']
        
        # LEFT: Combine Model (your previous project)
        ax1 = axes[0]
        ax1.text(0.5, 0.95, 'Combine Metrics', 
                ha='center', va='top', fontsize=20, fontweight='bold',
                transform=ax1.transAxes)
        ax1.text(0.5, 0.85, '40-time • Vertical • Broad Jump • 3-Cone • Shuttle',
                ha='center', va='top', fontsize=11, style='italic',
                transform=ax1.transAxes, color='gray')
        
        # Big R² display
        r2_color = self.colors['danger']
        ax1.text(0.5, 0.55, f'R² = {combine_r2:.3f}',
                ha='center', va='center', fontsize=48, fontweight='bold',
                transform=ax1.transAxes, color=r2_color)
        
        # Interpretation
        ax1.text(0.5, 0.35, 'Worse than guessing\nthe average',
                ha='center', va='top', fontsize=14,
                transform=ax1.transAxes, color=r2_color)
        
        # What it misses
        ax1.text(0.5, 0.15, '• Route discipline\n• Separation ability\n• Ball tracking\n• YAC creation',
                ha='center', va='top', fontsize=11,
                transform=ax1.transAxes, bbox=dict(boxstyle='round', facecolor='#ffebee'))
        
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis('off')
        
        # RIGHT: Tracking Model
        ax2 = axes[1]
        ax2.text(0.5, 0.95, 'TRACKING METRICS', 
                ha='center', va='top', fontsize=20, fontweight='bold',
                transform=ax2.transAxes)
        ax2.text(0.5, 0.85, 'Speed • Separation • Route Running • Change of Direction',
                ha='center', va='top', fontsize=11, style='italic',
                transform=ax2.transAxes, color='gray')
        
        # Big R² display
        r2_color = self.colors['success'] if tracking_r2 > 0.2 else self.colors['warning']
        ax2.text(0.5, 0.55, f'R² = {tracking_r2:.3f}',
                ha='center', va='center', fontsize=48, fontweight='bold',
                transform=ax2.transAxes, color=r2_color)
        
        # Improvement calculation
        if combine_r2 < 0:
            improvement_pct = ((tracking_r2 - combine_r2) / abs(combine_r2)) * 100
        else:
            improvement_pct = ((tracking_r2 - combine_r2) / combine_r2) * 100
        
        ax2.text(0.5, 0.35, f'{improvement_pct:.0f}% improvement\nover combine testing',
                ha='center', va='top', fontsize=14, fontweight='bold',
                transform=ax2.transAxes, color=r2_color)
        
        # What it captures
        ax2.text(0.5, 0.15, '• Real game speed\n• Separation skills\n• Route precision\n• Playmaking ability',
                ha='center', va='top', fontsize=11,
                transform=ax2.transAxes, bbox=dict(boxstyle='round', facecolor='#e8f5e9'))
        
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        
        plt.suptitle('The Evolution of WR Evaluation: Static vs Dynamic Metrics',
                    fontsize=18, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" Saved: {save_path}")
        plt.show()
        
        return fig
    
    def plot_feature_importance_football(self, top_n=15, save_path='feature_importance.png'):

        # Feature importance with football context - what ACTUALLY predicts success.

        feature_imp = self.analyzer.results['feature_importance'].head(top_n).copy()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Map features to football concepts
        football_labels = {
            'speed_score': 'Top-End Speed',
            'separation_consistency': 'Get-Open Ability',
            'route_diversity': 'Route Tree Versatility',
            'yac_ability': 'Yards After Catch',
            'sharp_cut_ability': 'Cutting Speed',
            'man_coverage_win_rate': 'vs Man Coverage',
            'burst_rate': 'Acceleration Bursts',
            'qb_friendly': 'Reliable Hands',
            'cut_separation': 'Separation from Cuts',
            'route_bend_ability': 'Route Bending',
            'explosive_rate': 'Explosive Plays',
            'first_step_quickness': 'Release Speed',
            'distance_per_play': 'Route Depth',
            'brake_rate': 'Deceleration Control',
            'total_plays': 'Volume/Sample Size'
        }
        
        feature_imp['football_label'] = feature_imp['feature'].map(
            lambda x: football_labels.get(x, x.replace('_', ' ').title())
        )
        
        # Color by category
        colors = []
        for feat in feature_imp['feature']:
            if any(x in feat for x in ['speed', 'burst', 'quickness', 'brake']):
                colors.append(self.colors['elite'])  # Athleticism
            elif any(x in feat for x in ['separation', 'route', 'man_coverage']):
                colors.append(self.colors['primary'])  # Route Running
            elif any(x in feat for x in ['yac', 'qb_friendly', 'explosive']):
                colors.append(self.colors['success'])  # Playmaking
            elif any(x in feat for x in ['cut', 'bend']):
                colors.append(self.colors['warning'])  # COD
            else:
                colors.append(self.colors['neutral'])
        
        bars = ax.barh(feature_imp['football_label'], feature_imp['importance'], color=colors)
        
        ax.set_xlabel('Feature Importance (Impact on Prediction)', fontsize=12, fontweight='bold')
        ax.set_title('What Actually Predicts WR Success?\nTracking Metrics Ranked by Predictive Power',
                    fontsize=14, fontweight='bold', pad=20)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, feature_imp['importance'])):
            ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, 
                   f'{val:.3f}', va='center', fontsize=9, fontweight='bold')
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=self.colors['elite'], label='Athleticism'),
            Patch(facecolor=self.colors['primary'], label='Route Running'),
            Patch(facecolor=self.colors['success'], label='Playmaking'),
            Patch(facecolor=self.colors['warning'], label='Change of Direction'),
        ]
        ax.legend(handles=legend_elements, loc='lower right', frameon=True, fontsize=10)
        
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" Saved: {save_path}")
        plt.show()
        
        return fig
    
    def plot_player_archetypes(self, save_path='player_archetypes.png'):

        # Visualize distinct receiver types - helps scouts understand player profiles.
        if 'archetype' not in self.df.columns:
            print("Run identify_archetypes() first!")
            return None
        
        # Get players with archetype assignments
        df_archetypes = self.df.dropna(subset=['archetype'])
        
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)
        
        # Define archetype names and characteristics
        archetypes = {
            0: {'name': 'Deep Threat Burners', 'color': '#e74c3c', 
                'key_traits': ['speed_score', 'burst_rate']},
            1: {'name': 'Route Technicians', 'color': '#3498db',
                'key_traits': ['separation_consistency', 'route_diversity']},
            2: {'name': 'YAC Monsters', 'color': '#2ecc71',
                'key_traits': ['yac_ability', 'explosive_rate']},
            3: {'name': 'Complete Receivers', 'color': '#9b59b6',
                'key_traits': ['athleticism_score', 'route_running_grade']},
            4: {'name': 'Possession Specialists', 'color': '#f39c12',
                'key_traits': ['qb_friendly', 'contested_catch_rate']}
        }
        
        # Main scatter plot (top)
        ax_main = fig.add_subplot(gs[0, :])
        
        for arch_id, arch_info in archetypes.items():
            arch_data = df_archetypes[df_archetypes['archetype'] == arch_id]
            if len(arch_data) > 0 and 'speed_score' in arch_data and 'separation_consistency' in arch_data:
                ax_main.scatter(
                    arch_data['speed_score'], 
                    arch_data['separation_consistency'],
                    c=arch_info['color'], 
                    label=arch_info['name'],
                    s=100, 
                    alpha=0.6,
                    edgecolors='black',
                    linewidth=0.5
                )
        
        ax_main.set_xlabel('Speed Score (Top-End Speed)', fontsize=12, fontweight='bold')
        ax_main.set_ylabel('Separation Consistency (Get Open)', fontsize=12, fontweight='bold')
        ax_main.set_title('5 Distinct Receiver Archetypes: Speed vs Separation', 
                         fontsize=14, fontweight='bold')
        ax_main.legend(loc='best', frameon=True, fontsize=10)
        ax_main.grid(alpha=0.3)
        
        # Individual archetype profiles (bottom)
        positions = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1)]
        
        for idx, (arch_id, arch_info) in enumerate(archetypes.items()):
            if idx >= len(positions):
                break
                
            row, col = positions[idx]
            ax = fig.add_subplot(gs[row, col])
            
            arch_data = df_archetypes[df_archetypes['archetype'] == arch_id]
            
            # Radar chart of key metrics
            metrics = ['speed_score', 'separation_consistency', 'yac_ability', 
                      'route_diversity', 'explosive_rate']
            available_metrics = [m for m in metrics if m in arch_data.columns]
            
            if len(arch_data) > 0 and available_metrics:
                values = [arch_data[m].mean() for m in available_metrics]
                
                # Normalize to 0-100 scale
                normalized_values = []
                for m, v in zip(available_metrics, values):
                    col_data = df_archetypes[m].dropna()
                    if col_data.std() > 0:
                        norm_val = ((v - col_data.mean()) / col_data.std()) * 15 + 50
                        normalized_values.append(max(0, min(100, norm_val)))
                    else:
                        normalized_values.append(50)
                
                x = np.arange(len(available_metrics))
                ax.bar(x, normalized_values, color=arch_info['color'], alpha=0.7)
                ax.set_xticks(x)
                ax.set_xticklabels([m.replace('_', '\n') for m in available_metrics], 
                                  rotation=0, fontsize=8)
                ax.set_ylim(0, 100)
                ax.set_title(f"{arch_info['name']}\n({len(arch_data)} players)", 
                           fontsize=11, fontweight='bold')
                ax.axhline(50, color='gray', linestyle='--', alpha=0.5, linewidth=1)
                ax.grid(axis='y', alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'Insufficient\nData', 
                       ha='center', va='center', fontsize=12,
                       transform=ax.transAxes)
                ax.set_title(arch_info['name'], fontsize=11, fontweight='bold')
                ax.axis('off')
        
        plt.suptitle('Understanding Receiver Diversity: One Size Does Not Fit All',
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" Saved: {save_path}")
        plt.show()
        
        return fig
    
    def plot_actual_vs_predicted(self, save_path='actual_vs_predicted.png'):

        # Show prediction accuracy - how well the model captures reality.

        pred_data = self.analyzer.results['predictions']
        metrics = self.analyzer.results['metrics']
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Training set
        ax1 = axes[0]
        ax1.scatter(pred_data['y_train'], pred_data['y_pred_train'], 
                   alpha=0.6, s=80, color=self.colors['primary'],
                   edgecolors='black', linewidth=0.5)
        
        # Perfect prediction line
        min_val = min(pred_data['y_train'].min(), pred_data['y_pred_train'].min())
        max_val = max(pred_data['y_train'].max(), pred_data['y_pred_train'].max())
        ax1.plot([min_val, max_val], [min_val, max_val], 
                'r--', linewidth=2, label='Perfect Prediction')
        
        ax1.set_xlabel('Actual Performance', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Predicted Performance', fontsize=12, fontweight='bold')
        ax1.set_title(f'Training Set (n={len(pred_data["y_train"])})\nR² = {metrics["train_r2"]:.3f}',
                     fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(alpha=0.3)
        
        # Test set
        ax2 = axes[1]
        ax2.scatter(pred_data['y_test'], pred_data['y_pred_test'],
                   alpha=0.6, s=80, color=self.colors['success'],
                   edgecolors='black', linewidth=0.5)
        
        min_val = min(pred_data['y_test'].min(), pred_data['y_pred_test'].min())
        max_val = max(pred_data['y_test'].max(), pred_data['y_pred_test'].max())
        ax2.plot([min_val, max_val], [min_val, max_val],
                'r--', linewidth=2, label='Perfect Prediction')
        
        ax2.set_xlabel('Actual Performance', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Predicted Performance', fontsize=12, fontweight='bold')
        ax2.set_title(f'Test Set (n={len(pred_data["y_test"])})\nR² = {metrics["test_r2"]:.3f} | MAE = {metrics["test_mae"]:.3f}',
                     fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(alpha=0.3)
        
        plt.suptitle('Model Prediction Quality: Capturing Real Performance',
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.show()
        
        return fig
    
    def plot_context_performance(self, save_path='context_matters.png'):

        # Show how context affects metrics - volume, coverage type, route depth.
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Performance by Volume (fatigue effect)
        ax1 = fig.add_subplot(gs[0, 0])
        if 'volume_tier' in self.df.columns and 'yac_ability' in self.df.columns:
            volume_performance = self.df.groupby('volume_tier')['yac_ability'].mean().dropna()
            if len(volume_performance) > 0:
                volume_performance.plot(kind='bar', ax=ax1, color=self.colors['primary'])
                ax1.set_title('Does High Volume Hurt Performance?\nYAC Ability by Playing Time',
                            fontsize=12, fontweight='bold')
                ax1.set_xlabel('Volume Tier', fontsize=11)
                ax1.set_ylabel('Average YAC Over Expected', fontsize=11)
                ax1.axhline(0, color='black', linestyle='--', linewidth=1)
                ax1.grid(axis='y', alpha=0.3)
                plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 2. Man Coverage Performance Distribution
        ax2 = fig.add_subplot(gs[0, 1])
        if 'man_coverage_win_rate' in self.df.columns:
            man_data = self.df['man_coverage_win_rate'].dropna()
            if len(man_data) > 0:
                ax2.hist(man_data, bins=30, color=self.colors['warning'], 
                        alpha=0.7, edgecolor='black')
                ax2.axvline(man_data.median(), color='red', linestyle='--', 
                           linewidth=2, label=f'Median: {man_data.median():.2f} yards')
                ax2.set_title('Separation vs Man Coverage\nThe Ultimate Test of Route Running',
                            fontsize=12, fontweight='bold')
                ax2.set_xlabel('Separation at Throw (yards)', fontsize=11)
                ax2.set_ylabel('Number of Players', fontsize=11)
                ax2.legend(fontsize=10)
                ax2.grid(alpha=0.3)
        
        # 3. Route Depth vs Success
        ax3 = fig.add_subplot(gs[1, 0])
        if all(col in self.df.columns for col in ['distance_per_play', 'separation_consistency']):
            scatter_data = self.df[['distance_per_play', 'separation_consistency']].dropna()
            if len(scatter_data) > 0:
                ax3.scatter(scatter_data['distance_per_play'], 
                           scatter_data['separation_consistency'],
                           alpha=0.5, s=60, color=self.colors['success'],
                           edgecolors='black', linewidth=0.5)
                ax3.set_title('Route Depth vs Separation Ability\nDo Deep Threats Get Open?',
                            fontsize=12, fontweight='bold')
                ax3.set_xlabel('Average Route Depth (yards per play)', fontsize=11)
                ax3.set_ylabel('Separation Consistency (yards)', fontsize=11)
                ax3.grid(alpha=0.3)
        
        # 4. Speed vs YAC (does speed = playmaking?)
        ax4 = fig.add_subplot(gs[1, 1])
        if all(col in self.df.columns for col in ['speed_score', 'yac_ability']):
            yac_data = self.df[['speed_score', 'yac_ability']].dropna()
            if len(yac_data) > 0:
                ax4.scatter(yac_data['speed_score'], yac_data['yac_ability'],
                           alpha=0.5, s=60, color=self.colors['elite'],
                           edgecolors='black', linewidth=0.5)
                ax4.axhline(0, color='black', linestyle='--', linewidth=1)
                ax4.set_title('Speed vs Playmaking\nFast ≠ Always Productive',
                            fontsize=12, fontweight='bold')
                ax4.set_xlabel('Top-End Speed Score', fontsize=11)
                ax4.set_ylabel('YAC Over Expected', fontsize=11)
                ax4.grid(alpha=0.3)
        
        plt.suptitle('Context Matters: Understanding When & Why Metrics Change',
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.show()
        
        return fig
    
    def create_player_card(self, player_name, save_path=None):
        # Generate a scouting report card for a specific player.
        player_data = self.df[self.df['player_name'] == player_name]
        
        if len(player_data) == 0:
            print(f"Player '{player_name}' not found")
            return None
        
        player = player_data.iloc[0]
        
        fig = plt.figure(figsize=(12, 10))
        gs = GridSpec(4, 2, figure=fig, hspace=0.4, wspace=0.3)
        
        # Header
        ax_header = fig.add_subplot(gs[0, :])
        ax_header.text(0.5, 0.7, player_name.upper(),
                      ha='center', va='center', fontsize=24, fontweight='bold',
                      transform=ax_header.transAxes)
        ax_header.text(0.5, 0.3, 
                      f"{player.get('offense_team', 'N/A')} | {player.get('season', 'N/A')} | {player.get('total_plays', 0):.0f} plays",
                      ha='center', va='center', fontsize=14,
                      transform=ax_header.transAxes, color='gray')
        ax_header.axis('off')
        
        # Key metrics radar
        ax_radar = fig.add_subplot(gs[1:3, 0], projection='polar')
        
        metrics = ['speed_score', 'separation_consistency', 'yac_ability', 
                  'route_diversity', 'qb_friendly']
        available_metrics = [m for m in metrics if m in player.index and not pd.isna(player[m])]
        
        if available_metrics:
            # Get percentiles
            percentiles = []
            for m in available_metrics:
                col_data = self.df[m].dropna()
                if len(col_data) > 0:
                    percentile = (player[m] > col_data).sum() / len(col_data) * 100
                    percentiles.append(percentile)
                else:
                    percentiles.append(50)
            
            # Create radar
            angles = np.linspace(0, 2 * np.pi, len(available_metrics), endpoint=False).tolist()
            percentiles += percentiles[:1]  # Close the circle
            angles += angles[:1]
            
            ax_radar.plot(angles, percentiles, 'o-', linewidth=2, color=self.colors['primary'])
            ax_radar.fill(angles, percentiles, alpha=0.25, color=self.colors['primary'])
            ax_radar.set_xticks(angles[:-1])
            ax_radar.set_xticklabels([m.replace('_', '\n') for m in available_metrics], fontsize=9)
            ax_radar.set_ylim(0, 100)
            ax_radar.set_yticks([25, 50, 75, 100])
            ax_radar.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=8)
            ax_radar.grid(True)
            ax_radar.set_title('Performance Percentiles', fontsize=12, fontweight='bold', pad=20)
        
        # Stats table
        ax_stats = fig.add_subplot(gs[1:3, 1])
        ax_stats.axis('tight')
        ax_stats.axis('off')
        
        stats_data = [
            ['METRIC', 'VALUE'],
            ['Top Speed', f"{player.get('max_speed_99', 0):.1f} mph"],
            ['Separation (99th%)', f"{player.get('average_separation_99', 0):.2f} yds"],
            ['YAC Over Expected', f"{player.get('YACOE_MEAN', 0):.2f}"],
            ['Route Diversity', f"{player.get('route_diversity', 0):.1f}%"],
            ['Cut Separation', f"{player.get('cod_sep_generated_overall', 0):.2f} yds"],
            ['Total Plays', f"{player.get('total_plays', 0):.0f}"],
        ]
        
        table = ax_stats.table(cellText=stats_data, cellLoc='left',
                              colWidths=[0.6, 0.4], loc='center',
                              bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header row
        for i in range(2):
            table[(0, i)].set_facecolor('#3498db')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Strengths & Weaknesses
        ax_notes = fig.add_subplot(gs[3, :])
        ax_notes.axis('off')
        
        # Determine strengths
        strengths = []
        if player.get('speed_score', 0) > self.df['speed_score'].quantile(0.75):
            strengths.append("Elite speed threat")
        if player.get('separation_consistency', 0) > self.df['separation_consistency'].quantile(0.75):
            strengths.append("Consistent separator")
        if player.get('yac_ability', 0) > 0.5:
            strengths.append("YAC creator")
        
        weaknesses = []
        if player.get('speed_score', 0) < self.df['speed_score'].quantile(0.25):
            weaknesses.append("Limited top-end speed")
        if player.get('contested_catch_rate', 0) < 40:
            weaknesses.append("Struggles in traffic")
        
        notes_text = "Strengths:\n" + "\n".join(f"• {s}" for s in strengths[:3])
        notes_text += "\n\nWeaknesses:\n" + "\n".join(f"• {w}" for w in weaknesses[:2])
        
        ax_notes.text(0.5, 0.5, notes_text,
                     ha='center', va='center', fontsize=11,
                     transform=ax_notes.transAxes,
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.suptitle('Player Scouting Report', fontsize=16, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.show()
        return fig


# Helper function for easy use
def create_all_visuals(analyzer, output_dir='.'):
    # Generate all key visualizations.
    viz = TrackingVisualizer(analyzer)
    
    print("\n\nCreating Visualizations")
    
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate all plots
    viz.plot_model_comparison(save_path=f'{output_dir}/1_model_comparison.png')
    viz.plot_feature_importance_football(save_path=f'{output_dir}/2_feature_importance.png')
    viz.plot_actual_vs_predicted(save_path=f'{output_dir}/3_actual_vs_predicted.png')
    viz.plot_player_archetypes(save_path=f'{output_dir}/4_player_archetypes.png')
    viz.plot_context_performance(save_path=f'{output_dir}/5_context_matters.png')
    
    return viz