
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle, Patch
from matplotlib.gridspec import GridSpec
from scipy import stats

# Professional styling
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'


class TrackingVisualizer:
    """Creates football-focused visualizations with target-relevant insights."""

    def __init__(self, analyzer, analysis_type='nfl'):
        """
        Initialize with analyzer object containing data and results.
        
        Parameters:
        -----------
        analyzer : TrackingDataAnalyzer
            Analyzer object with processed data and model results
        analysis_type : str
            'nfl' for NFL rookie performance, 'draft' for draft prediction
        """
        self.analyzer = analyzer
        self.df = analyzer.df
        self.analysis_type = analysis_type
        self.colors = {
            'primary': '#1f77b4',
            'success': '#2ca02c', 
            'warning': '#ff7f0e',
            'danger': '#d62728',
            'neutral': '#7f7f7f',
            'elite': '#9467bd'
        }
        
        # Set target-specific labels
        if analysis_type == 'nfl':
            self.target_label = 'NFL Rookie Performance'
            self.target_unit = 'targets/game'
        else:
            self.target_label = 'Draft Capital'
            self.target_unit = 'draft points'
    
    def plot_model_comparison(self, combine_r2=-0.155, save_path='1_model_comparison.png'):
        """
        Side-by-side comparison showing tracking beats combine.
        
        Football Context:
        This single chart tells the entire story - combine testing fails to predict
        performance while tracking data captures what matters.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Get tracking model results
        tracking_r2 = self.analyzer.results['metrics']['test_r2']
        
        # Get sample size for context
        pred_data = self.analyzer.results.get('predictions', {})
        n_train = len(pred_data.get('y_train', []))
        n_test = len(pred_data.get('y_test', []))
        n_total = n_train + n_test
        
        # LEFT: Combine Model (previous project baseline)
        ax1 = axes[0]
        ax1.text(0.5, 0.95, 'COMBINE METRICS', 
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
        ax1.text(0.5, 0.15, '✗ Route discipline\n✗ Separation ability\n✗ Ball tracking\n✗ YAC creation',
                ha='center', va='top', fontsize=11,
                transform=ax1.transAxes, bbox=dict(boxstyle='round', facecolor='#ffebee'))
        
        # Sample size note for combine
        ax1.text(0.5, 0.02, 'n = 92 rookie WRs (2015-2023)',
                ha='center', va='bottom', fontsize=9, style='italic',
                transform=ax1.transAxes, color='gray')
        
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
            improvement_pct = ((tracking_r2 - combine_r2) / max(combine_r2, 0.001)) * 100
        
        ax2.text(0.5, 0.35, f'{improvement_pct:.0f}% improvement\nover combine testing',
                ha='center', va='top', fontsize=14, fontweight='bold',
                transform=ax2.transAxes, color=r2_color)
        
        # What it captures
        ax2.text(0.5, 0.15, 'Real game speed\nSeparation skills\nRoute precision\nPlaymaking ability',
                ha='center', va='top', fontsize=11,
                transform=ax2.transAxes, bbox=dict(boxstyle='round', facecolor='#e8f5e9'))
        
        # Sample size note for tracking
        sample_note = f'n = {n_total} players' if n_total > 0 else 'College tracking data'
        ax2.text(0.5, 0.02, sample_note,
                ha='center', va='bottom', fontsize=9, style='italic',
                transform=ax2.transAxes, color='gray')
        
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        
        plt.suptitle('The Evolution of WR Evaluation: Static vs Dynamic Metrics',
                    fontsize=18, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
        
        return fig
    
    def plot_feature_importance_football(self, top_n=15, save_path='2_feature_importance.png'):
        """
        Feature importance with football context - what ACTUALLY predicts success.
        """
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
            'total_plays': 'Volume/Sample Size',
            'max_speed_99': 'Max Speed 99',
            'cod_sep_generated_overall': 'Cod Sep Generated Overall',
            'changedir_route_MEAN': 'Changedir Route Mean'
        }
        
        feature_imp['football_label'] = feature_imp['feature'].map(
            lambda x: football_labels.get(x, x.replace('_', ' ').title())
        )
        
        # Color by category
        colors = []
        for feat in feature_imp['feature']:
            if any(x in feat.lower() for x in ['speed', 'burst', 'quickness', 'brake', 'max_speed']):
                colors.append(self.colors['elite'])  # Athleticism
            elif any(x in feat.lower() for x in ['separation', 'route', 'man_coverage', 'changedir']):
                colors.append(self.colors['primary'])  # Route Running
            elif any(x in feat.lower() for x in ['yac', 'qb_friendly', 'explosive', 'cpoe', 'yacoe']):
                colors.append(self.colors['success'])  # Playmaking
            elif any(x in feat.lower() for x in ['cut', 'bend', 'cod']):
                colors.append(self.colors['warning'])  # COD
            else:
                colors.append(self.colors['neutral'])
        
        bars = ax.barh(feature_imp['football_label'], feature_imp['importance'], color=colors)
        
        ax.set_xlabel('Feature Importance (Impact on Prediction)', fontsize=12, fontweight='bold')
        ax.set_title(f'What Actually Predicts WR Success?\nTracking Metrics Ranked by Predictive Power',
                    fontsize=14, fontweight='bold', pad=20)
        
        # Add value labels
        for bar, val in zip(bars, feature_imp['importance']):
            ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, 
                   f'{val:.3f}', va='center', fontsize=9, fontweight='bold')
        
        # Legend
        legend_elements = [
            Patch(facecolor=self.colors['elite'], label='Athleticism'),
            Patch(facecolor=self.colors['primary'], label='Route Running'),
            Patch(facecolor=self.colors['success'], label='Playmaking'),
            Patch(facecolor=self.colors['warning'], label='Change of Direction'),
        ]
        ax.legend(handles=legend_elements, loc='lower right', frameon=True, fontsize=10)
        
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()  # Highest importance at top
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
        
        return fig
    
    def plot_actual_vs_predicted(self, save_path='3_actual_vs_predicted.png'):
        """
        Show prediction accuracy - how well the model captures reality.
        Enhanced with target-specific labels.
        """
        pred_data = self.analyzer.results['predictions']
        metrics = self.analyzer.results['metrics']
        
        # Get target name for labels
        target_name = pred_data.get('target_name')
        if not target_name or target_name is None:
            target_name = 'targets_per_game' if self.analysis_type == 'nfl' else 'draft_value'
        
        if self.analysis_type == 'nfl':
            xlabel = f'Actual NFL Rookie {target_name.replace("_", " ").title()}'
            ylabel = f'Predicted NFL Rookie {target_name.replace("_", " ").title()}'
        else:
            xlabel = 'Actual Draft Value'
            ylabel = 'Predicted Draft Value'
        
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
        
        ax1.set_xlabel(xlabel, fontsize=12, fontweight='bold')
        ax1.set_ylabel(ylabel, fontsize=12, fontweight='bold')
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
        
        ax2.set_xlabel(xlabel, fontsize=12, fontweight='bold')
        ax2.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax2.set_title(f'Test Set (n={len(pred_data["y_test"])})\nR² = {metrics["test_r2"]:.3f} | MAE = {metrics["test_mae"]:.3f}',
                     fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(alpha=0.3)
        
        plt.suptitle('Model Prediction Quality: Capturing Real Performance',
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
        
        return fig
    
    def plot_player_archetypes(self, save_path='4_player_archetypes.png'):
        """
        Visualize distinct receiver types - helps scouts understand player profiles.
        """
        if 'archetype' not in self.df.columns:
            print("⚠ Archetypes not found. Skipping archetype visualization.")
            return None
        
        # Get players with archetype assignments
        df_archetypes = self.df.dropna(subset=['archetype'])
        
        if len(df_archetypes) < 5:
            print("⚠ Insufficient archetype data. Skipping.")
            return None
        
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)
        
        # Define archetype names and characteristics
        archetypes = {
            0: {'name': 'Deep Threat Burners', 'color': '#e74c3c'},
            1: {'name': 'Route Technicians', 'color': '#3498db'},
            2: {'name': 'YAC Monsters', 'color': '#2ecc71'},
            3: {'name': 'Complete Receivers', 'color': '#9b59b6'},
            4: {'name': 'Possession Specialists', 'color': '#f39c12'}
        }
        
        # Main scatter plot (top)
        ax_main = fig.add_subplot(gs[0, :])
        
        # Find available columns for scatter plot
        x_col = 'speed_score' if 'speed_score' in df_archetypes.columns else 'max_speed_99'
        y_col = 'separation_consistency' if 'separation_consistency' in df_archetypes.columns else 'average_separation_99'
        
        for arch_id, arch_info in archetypes.items():
            arch_data = df_archetypes[df_archetypes['archetype'] == arch_id]
            if len(arch_data) > 0 and x_col in arch_data.columns and y_col in arch_data.columns:
                plot_data = arch_data[[x_col, y_col]].dropna()
                if len(plot_data) > 0:
                    ax_main.scatter(
                        plot_data[x_col], 
                        plot_data[y_col],
                        c=arch_info['color'], 
                        label=arch_info['name'],
                        s=100, 
                        alpha=0.6,
                        edgecolors='black',
                        linewidth=0.5
                    )
        
        ax_main.set_xlabel(x_col.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax_main.set_ylabel(y_col.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax_main.set_title('5 Distinct Receiver Archetypes: Speed vs Separation', 
                         fontsize=14, fontweight='bold')
        ax_main.legend(loc='best', frameon=True, fontsize=10)
        ax_main.grid(alpha=0.3)
        
        # Individual archetype profiles (bottom)
        positions = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1)]
        
        # Find available metrics for bar charts
        metrics_to_check = ['speed_score', 'separation_consistency', 'yac_ability', 
                          'route_diversity', 'explosive_rate',
                          'max_speed_99', 'average_separation_99', 'YACOE_MEAN',
                          'changedir_route_MEAN', 'burst_rate']
        available_metrics = [m for m in metrics_to_check if m in df_archetypes.columns][:5]
        
        for idx, (arch_id, arch_info) in enumerate(archetypes.items()):
            if idx >= len(positions):
                break
                
            row, col = positions[idx]
            ax = fig.add_subplot(gs[row, col])
            
            arch_data = df_archetypes[df_archetypes['archetype'] == arch_id]
            
            if len(arch_data) > 0 and available_metrics:
                values = []
                for m in available_metrics:
                    if m in arch_data.columns:
                        val = arch_data[m].mean()
                        # Normalize to 0-100 scale
                        col_data = df_archetypes[m].dropna()
                        if len(col_data) > 0 and col_data.std() > 0:
                            norm_val = ((val - col_data.mean()) / col_data.std()) * 15 + 50
                            values.append(max(0, min(100, norm_val)))
                        else:
                            values.append(50)
                    else:
                        values.append(50)
                
                x = np.arange(len(available_metrics))
                ax.bar(x, values, color=arch_info['color'], alpha=0.7)
                ax.set_xticks(x)
                ax.set_xticklabels([m.replace('_', '\n')[:15] for m in available_metrics], 
                                  rotation=0, fontsize=7)
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
        
        plt.suptitle('Understanding Receiver Diversity: One Size Does NOT Fit All',
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
        
        return fig
    
    def plot_tracking_to_outcome(self, save_path='5_tracking_to_outcome.png'):
        """
        NEW VISUALIZATION: Show how top tracking features predict the actual target.
        
        This directly answers: "Do tracking metrics predict NFL success / draft position?"
        
        Football Context:
        - For NFL: Shows which college tracking metrics predict rookie production
        - For Draft: Shows which metrics scouts value (consciously or not)
        """
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
        
        # Get feature importance to pick top features
        feat_imp = self.analyzer.results.get('feature_importance', pd.DataFrame())
        
        if len(feat_imp) < 2:
            print("⚠ Insufficient feature importance data")
            return None
        
        # Get top 4 features
        top_features = feat_imp.head(4)['feature'].tolist()
        
        # Get the actual target that was used in modeling
        pred_data = self.analyzer.results.get('predictions', {})
        target_col = pred_data.get('target_name', None)
        
        # Fallback if target_name not stored
        if target_col is None or target_col not in self.df.columns:
            if self.analysis_type == 'nfl':
                target_candidates = ['targets_per_game', 'yards_per_game', 'receptions_per_game', 'catch_rate']
            else:
                target_candidates = ['draft_capital', 'rec_yards', 'production_score', 'draft_pick']
            
            for tc in target_candidates:
                if tc in self.df.columns and self.df[tc].notna().sum() > 10:
                    target_col = tc
                    break
        
        if target_col is None:
            print("⚠ No valid target column found")
            return None
        
        # Set label based on what target we're using
        if self.analysis_type == 'nfl':
            target_label = 'NFL Rookie Performance'
        else:
            if target_col == 'draft_capital':
                target_label = 'Draft Value'
            elif target_col == 'rec_yards':
                target_label = 'College Production'
            else:
                target_label = target_col.replace('_', ' ').title()
        
        # Create 4 scatter plots: top features vs target
        for idx, feature in enumerate(top_features[:4]):
            row, col = divmod(idx, 2)
            ax = fig.add_subplot(gs[row, col])
            
            if feature not in self.df.columns:
                ax.text(0.5, 0.5, f'{feature}\nNot Available', 
                       ha='center', va='center', fontsize=12,
                       transform=ax.transAxes)
                ax.axis('off')
                continue
            
            # Get valid data
            plot_data = self.df[[feature, target_col]].dropna()
            
            if len(plot_data) < 5:
                ax.text(0.5, 0.5, f'{feature}\nInsufficient Data', 
                       ha='center', va='center', fontsize=12,
                       transform=ax.transAxes)
                ax.axis('off')
                continue
            
            # Scatter plot
            ax.scatter(plot_data[feature], plot_data[target_col],
                      alpha=0.5, s=60, color=self.colors['primary'],
                      edgecolors='black', linewidth=0.3)
            
            # Add correlation and trend line
            corr = plot_data[feature].corr(plot_data[target_col])
            
            # Trend line
            z = np.polyfit(plot_data[feature], plot_data[target_col], 1)
            p = np.poly1d(z)
            x_line = np.linspace(plot_data[feature].min(), plot_data[feature].max(), 100)
            ax.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.7)
            
            # Feature importance from model
            feat_importance = feat_imp[feat_imp['feature'] == feature]['importance'].values
            imp_val = feat_importance[0] if len(feat_importance) > 0 else 0
            
            # Football-friendly label
            football_labels = {
                'speed_score': 'Top-End Speed',
                'separation_consistency': 'Get-Open Ability',
                'route_bend_ability': 'Route Bending',
                'explosive_rate': 'Explosive Plays',
                'burst_rate': 'Acceleration Bursts',
                'max_speed_99': 'Consistent Top Speed',
                'cod_sep_generated_overall': 'Separation from Cuts',
                'distance_per_play': 'Route Depth',
                'first_step_quickness': 'Release Speed',
                'route_diversity': 'Route Versatility'
            }
            
            feature_label = football_labels.get(feature, feature.replace('_', ' ').title())
            
            ax.set_xlabel(f'{feature_label}', fontsize=11, fontweight='bold')
            ax.set_ylabel(f'{target_col.replace("_", " ").title()}', fontsize=11, fontweight='bold')
            ax.set_title(f'{feature_label} → {target_label}\nr = {corr:.3f} | Importance: {imp_val:.3f}',
                        fontsize=11, fontweight='bold')
            ax.grid(alpha=0.3)
            
            # Color code correlation strength
            if abs(corr) > 0.3:
                ax.patch.set_facecolor('#e8f5e9')
                ax.patch.set_alpha(0.3)
            elif abs(corr) < 0.1:
                ax.patch.set_facecolor('#ffebee')
                ax.patch.set_alpha(0.3)
        
        # Suptitle based on analysis type
        if self.analysis_type == 'nfl':
            title = 'Tracking Metrics → NFL Rookie Production\nDo College Skills Translate to the Pros?'
        else:
            title = 'Tracking Metrics → Draft Capital\nWhat Do Scouts Really Value?'
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
        
        return fig
    
    def plot_value_discovery(self, save_path='6_value_discovery.png'):
        """
        NEW VISUALIZATION: Identify over/under-valued players.
        
        Football Context:
        Shows players who:
        - Outperformed their tracking profile (late bloomers, scheme fit)
        - Underperformed their tracking profile (injury, opportunity, bust risk)
        
        This is DIRECTLY useful for scouts identifying value picks.
        """
        pred_data = self.analyzer.results.get('predictions', {})
        
        if 'y_test' not in pred_data or 'y_pred_test' not in pred_data:
            print("⚠ Prediction data not available for value discovery")
            return None
        
        y_test = pred_data['y_test']
        y_pred = pred_data['y_pred_test']
        
        # Calculate residuals (actual - predicted)
        residuals = np.array(y_test) - np.array(y_pred)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # LEFT: Residual distribution
        ax1 = axes[0]
        ax1.hist(residuals, bins=20, color=self.colors['primary'], 
                alpha=0.7, edgecolor='black')
        ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
        ax1.axvline(residuals.mean(), color='green', linestyle='-', linewidth=2, 
                   label=f'Mean Error: {residuals.mean():.2f}')
        
        ax1.set_xlabel('Prediction Error (Actual - Predicted)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of Players', fontsize=12, fontweight='bold')
        ax1.set_title('Prediction Error Distribution\nPositive = Outperformed | Negative = Underperformed',
                     fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(alpha=0.3)
        
        # Add interpretation zones
        ax1.axvspan(residuals.min(), -residuals.std(), alpha=0.1, color='red', 
                   label='Underperformers')
        ax1.axvspan(residuals.std(), residuals.max(), alpha=0.1, color='green',
                   label='Overperformers')
        
        # RIGHT: Predicted vs Actual with value zones
        ax2 = axes[1]
        
        # Color points by residual (over/under performance)
        colors = ['green' if r > residuals.std() else 'red' if r < -residuals.std() else 'gray' 
                 for r in residuals]
        
        ax2.scatter(y_pred, y_test, c=colors, alpha=0.6, s=80, edgecolors='black', linewidth=0.5)
        
        # Perfect prediction line
        min_val = min(min(y_test), min(y_pred))
        max_val = max(max(y_test), max(y_pred))
        ax2.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect Fit')
        
        ax2.set_xlabel('Predicted Performance (from Tracking)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Actual Performance', fontsize=12, fontweight='bold')
        ax2.set_title('Value Discovery: Who Beat/Missed Expectations?\nGreen = Value Pick | Red = Bust Risk',
                     fontsize=13, fontweight='bold')
        ax2.grid(alpha=0.3)
        
        # Legend
        legend_elements = [
            Patch(facecolor='green', label='Outperformed (Value)'),
            Patch(facecolor='red', label='Underperformed (Risk)'),
            Patch(facecolor='gray', label='As Expected'),
        ]
        ax2.legend(handles=legend_elements, loc='upper left', fontsize=10)
        
        # Count players in each category
        overperformers = sum(1 for r in residuals if r > residuals.std())
        underperformers = sum(1 for r in residuals if r < -residuals.std())
        as_expected = len(residuals) - overperformers - underperformers
        
        # Add text summary
        summary = f"Overperformers: {overperformers} | As Expected: {as_expected} | Underperformers: {underperformers}"
        fig.text(0.5, 0.02, summary, ha='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('Finding Value: Tracking Profile vs Actual Outcome',
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" Saved: {save_path}")
        plt.close()
        
        return fig
    
    def create_player_card(self, player_name, save_path=None):
        """Generate a scouting report card for a specific player."""
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
        
        # Use available metrics
        metrics_to_check = ['speed_score', 'separation_consistency', 'yac_ability', 
                          'route_diversity', 'qb_friendly',
                          'max_speed_99', 'average_separation_99', 'YACOE_MEAN']
        available_metrics = [m for m in metrics_to_check if m in player.index and pd.notna(player.get(m))][:5]
        
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
            percentiles_closed = percentiles + percentiles[:1]
            angles_closed = angles + angles[:1]
            
            ax_radar.plot(angles_closed, percentiles_closed, 'o-', linewidth=2, color=self.colors['primary'])
            ax_radar.fill(angles_closed, percentiles_closed, alpha=0.25, color=self.colors['primary'])
            ax_radar.set_xticks(angles)
            ax_radar.set_xticklabels([m.replace('_', '\n')[:12] for m in available_metrics], fontsize=9)
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
            ['Top Speed', f"{player.get('max_speed_99', player.get('max_speed_max', 0)):.1f} mph"],
            ['Separation (99th%)', f"{player.get('average_separation_99', 0):.2f} yds"],
            ['YAC Over Expected', f"{player.get('YACOE_MEAN', player.get('yac_ability', 0)):.2f}"],
            ['Route Complexity', f"{player.get('changedir_route_MEAN', player.get('route_diversity', 0)):.1f}"],
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
        
        # Determine strengths based on available data
        strengths = []
        weaknesses = []
        
        speed_col = 'speed_score' if 'speed_score' in self.df.columns else 'max_speed_99'
        sep_col = 'separation_consistency' if 'separation_consistency' in self.df.columns else 'average_separation_99'
        
        if speed_col in player.index and pd.notna(player.get(speed_col)):
            if player[speed_col] > self.df[speed_col].quantile(0.75):
                strengths.append("Elite speed threat")
            elif player[speed_col] < self.df[speed_col].quantile(0.25):
                weaknesses.append("Limited top-end speed")
        
        if sep_col in player.index and pd.notna(player.get(sep_col)):
            if player[sep_col] > self.df[sep_col].quantile(0.75):
                strengths.append("Consistent separator")
            elif player[sep_col] < self.df[sep_col].quantile(0.25):
                weaknesses.append("Struggles to separate")
        
        yac_col = 'yac_ability' if 'yac_ability' in player.index else 'YACOE_MEAN'
        if yac_col in player.index and pd.notna(player.get(yac_col)):
            if player[yac_col] > 0.5:
                strengths.append("YAC creator")
            elif player[yac_col] < -0.5:
                weaknesses.append("Limited after catch")
        
        if not strengths:
            strengths = ["Well-rounded profile"]
        if not weaknesses:
            weaknesses = ["No major weaknesses identified"]
        
        notes_text = "STRENGTHS:\n" + "\n".join(f"• {s}" for s in strengths[:3])
        notes_text += "\n\nAREAS TO DEVELOP:\n" + "\n".join(f"• {w}" for w in weaknesses[:2])
        
        ax_notes.text(0.5, 0.5, notes_text,
                     ha='center', va='center', fontsize=11,
                     transform=ax_notes.transAxes,
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.suptitle('Player Scouting Report', fontsize=16, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
        return fig


def create_all_visuals(analyzer, output_dir='.', analysis_type=None):
    """
    Generate all key visualizations.
    
    Params: 
    analyzer : TrackingDataAnalyzer
        Analyzer object with processed data and results
    output_dir : str
        Directory to save visualizations
    analysis_type : str or None
        'nfl' for NFL rookie performance, 'draft' for draft prediction
        If None, auto-detects from the target variable used in modeling
    """
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Auto-detect analysis type from the target variable used in modeling
    if analysis_type is None:
        target_name = analyzer.results.get('predictions', {}).get('target_name', '')
        
        # NFL targets
        nfl_targets = ['targets_per_game', 'yards_per_game', 'receptions_per_game', 'catch_rate']
        # Draft targets  
        draft_targets = ['draft_capital', 'draft_pick', 'draft_round', 'rec_yards', 'production_score']
        
        if target_name in nfl_targets:
            analysis_type = 'nfl'
        elif target_name in draft_targets:
            analysis_type = 'draft'
        else:
            # Fallback: check which columns have more data
            nfl_data = sum(1 for col in nfl_targets if col in analyzer.df.columns and analyzer.df[col].notna().sum() > 10)
            draft_data = analyzer.df.get('draft_capital', pd.Series()).notna().sum()
            
            if nfl_data > 0 and analyzer.df.get('targets_per_game', pd.Series()).notna().sum() > 20:
                analysis_type = 'nfl'
            else:
                analysis_type = 'draft'
    
    # Set prefix based on analysis type
    prefix = 'nfl' if analysis_type == 'nfl' else 'draft'
    
    print(f"Creating Visualizations ({analysis_type.upper()} Analysis)")
    
    viz = TrackingVisualizer(analyzer, analysis_type=analysis_type)
    
    # Generate all plots
    viz.plot_model_comparison(save_path=f'{output_dir}/{prefix}_1_model_comparison.png')
    viz.plot_feature_importance_football(save_path=f'{output_dir}/{prefix}_2_feature_importance.png')
    viz.plot_actual_vs_predicted(save_path=f'{output_dir}/{prefix}_3_actual_vs_predicted.png')
    viz.plot_player_archetypes(save_path=f'{output_dir}/{prefix}_4_player_archetypes.png')
    viz.plot_tracking_to_outcome(save_path=f'{output_dir}/{prefix}_5_tracking_to_outcome.png')
    viz.plot_value_discovery(save_path=f'{output_dir}/{prefix}_6_value_discovery.png')
    
    print(f"\nAll visualizations saved to: {output_dir}/")
    
    return viz






