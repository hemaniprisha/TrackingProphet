"""
Tracking Prophet - Interactive Dashboard
NFL WR Performance Prediction from In-Game Tracking Data
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="Tracking Prophet | NFL Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.3rem;
        color: #666;
        text-align: center;
        padding-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .insight-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .success-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .tracking-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load analysis results
@st.cache_data
def load_analysis_results(result_path):
    """Load pre-computed analysis results from pickle file"""
    if os.path.exists(result_path):
        with open(result_path, 'rb') as f:
            return pickle.load(f)
    return None

# Load both analysis types
@st.cache_data
def load_all_results():
    """Load both draft and NFL tracking analysis results"""
    results = {
        'draft': None,
        'nfl': None,
        'combine': None
    }
    
    # Try to load draft tracking results
    if os.path.exists('tracking_draft_export.pkl'):
        results['draft'] = load_analysis_results('tracking_draft_export.pkl')
    
    # Try to load NFL tracking results
    if os.path.exists('tracking_nfl_export.pkl'):
        results['nfl'] = load_analysis_results('tracking_nfl_export.pkl')
    
    # Try to load combine results for comparison
    if os.path.exists('combine_analysis_export.pkl'):
        results['combine'] = load_analysis_results('combine_analysis_export.pkl')
    
    return results

# Load all results
all_results = load_all_results()

# Check if we have at least one dataset
has_data = any(all_results.values())

if not has_data:
    st.error("No analysis data found. Please run the analysis scripts first.")
    st.info("""
    **To generate required data files:**
    
    1. **For College Tracking → Draft Analysis:**
       ```
       python run_analysis_draft.py --data your_tracking_data.csv
       ```
       This will create: `tracking_draft_export.pkl`
    
    2. **For College Tracking → NFL Rookie Performance:**
       ```
       python run_analysis_nfl.py --data your_tracking_data.csv
       ```
       This will create: `tracking_nfl_export.pkl`
    
    3. **For Combine Comparison (optional):**
       ```
       python combine_analysis.py
       ```
       This will create: `combine_analysis_export.pkl`
    
    Once generated, refresh this page.
    """)
    st.stop()

# Extract available datasets
available_analyses = []
if all_results['draft']:
    available_analyses.append('Draft Prediction')
if all_results['nfl']:
    available_analyses.append('NFL Rookie Performance')
if all_results['combine']:
    available_analyses.append('Combine (Comparison)')

# Title section
st.markdown('<div class="main-header">🎯 Tracking Prophet</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predicting NFL Success with In-Game Tracking Data</div>', unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/a/a2/National_Football_League_logo.svg/400px-National_Football_League_logo.svg.png", width=180)
    
    st.title("Navigation")
    
    # Analysis type selector
    analysis_type = st.radio(
        "Select Analysis Type:",
        available_analyses,
        help="Choose which tracking analysis to explore"
    )
    
    st.markdown("---")
    
    # Page navigation
    page = st.radio(
        "Navigate To:",
        ["Overview", "Performance Comparison", "Data Explorer", "ML Model Results", "Player Insights"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Show relevant stats based on selected analysis
    if analysis_type == 'Draft Prediction' and all_results['draft']:
        data = all_results['draft']
        st.markdown(f"""
        ### Draft Analysis Stats
        
        **Dataset:** College Tracking Data  
        **Target:** Draft Position/Grade  
        **Sample:** {len(data.get('processed_data', []))} players  
        **Model R²:** {data.get('metrics', {}).get('test_r2', 0):.3f}  
        **MAE:** {data.get('metrics', {}).get('test_mae', 0):.2f}
        
        ---
        
        **Key Insight:**  
        Tracking data captures real game performance that predicts draft success.
        """)
    
    elif analysis_type == 'NFL Rookie Performance' and all_results['nfl']:
        data = all_results['nfl']
        st.markdown(f"""
        ### NFL Performance Stats
        
        **Dataset:** College Tracking → NFL Stats  
        **Target:** NFL Rookie Production  
        **Sample:** {len(data.get('processed_data', []))} players  
        **Model R²:** {data.get('metrics', {}).get('test_r2', 0):.3f}  
        **MAE:** {data.get('metrics', {}).get('test_mae', 0):.2f}
        
        ---
        
        **Key Insight:**  
        College tracking metrics predict NFL success far better than combine testing.
        """)
    
    elif analysis_type == 'Combine (Comparison)' and all_results['combine']:
        data = all_results['combine']
        st.markdown(f"""
        ### Combine Testing Stats
        
        **Dataset:** NFL Combine Metrics  
        **Target:** NFL Rookie Performance  
        **Sample:** {len(data.get('merged_data', []))} players  
        **Model R²:** {data.get('metrics', {}).get('test_r2', 0):.3f}  
        **MAE:** {data.get('metrics', {}).get('mae', 0):.1f} ypg
        
        ---
        
        **Limitation:**  
        Static testing misses in-game performance factors.
        """)

# Get current dataset based on selection
current_data = None
if analysis_type == 'Draft Prediction':
    current_data = all_results['draft']
elif analysis_type == 'NFL Rookie Performance':
    current_data = all_results['nfl']
elif analysis_type == 'Combine (Comparison)':
    current_data = all_results['combine']

# Extract key metrics
if current_data:
    metrics = current_data.get('metrics', {})
    test_r2 = metrics.get('test_r2', 0)
    test_mae = metrics.get('test_mae', 0) if 'test_mae' in metrics else metrics.get('mae', 0)
    
    # Success metrics at top
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_records = len(current_data.get('processed_data', [])) if 'processed_data' in current_data else len(current_data.get('merged_data', []))
        st.metric("Total Records", f"{total_records:,}")
    
    with col2:
        if 'processed_data' in current_data:
            unique_players = len(current_data['processed_data']['player_name'].unique()) if 'player_name' in current_data['processed_data'].columns else total_records
        else:
            unique_players = len(current_data['merged_data']['player_name'].unique()) if 'player_name' in current_data['merged_data'].columns else total_records
        st.metric("Unique Players", f"{unique_players:,}")
    
    with col3:
        st.metric("Model R²", f"{test_r2:.3f}")
    
    with col4:
        mae_label = "MAE" if analysis_type != 'Combine (Comparison)' else "MAE (ypg)"
        st.metric(mae_label, f"{test_mae:.2f}")
    
    st.markdown("---")

# ============================================================================
# PAGE 1: OVERVIEW
# ============================================================================
if page == "Overview":
    st.header("Project Overview")
    
    # Show different content based on analysis type
    if analysis_type in ['Draft Prediction', 'NFL Rookie Performance']:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if analysis_type == 'Draft Prediction':
                st.markdown(f"""
                ### Research Question
                
                **"Can college tracking data predict NFL draft success?"**
                
                This analysis uses in-game tracking metrics from college football to predict 
                draft position and grades. Unlike static combine testing, tracking data captures:
                
                - Real-game separation ability
                - Route running efficiency
                - Performance vs. different coverage types
                - Consistency across multiple games
                - Explosive play creation
                
                ### Key Findings
                
                - **Model R²:** {test_r2:.3f}
                - **Prediction Error:** {test_mae:.2f}
                - **Sample Size:** {len(current_data.get('processed_data', []))} players
                
                ### What This Means
                
                Tracking data provides objective, quantifiable metrics that capture the skills 
                scouts look for. This bridges the gap between subjective evaluation and data-driven decisions.
                """)
            else:  # NFL Rookie Performance
                st.markdown(f"""
                ### Research Question
                
                **"Can college tracking data predict NFL rookie performance?"**
                
                This analysis examines whether in-game tracking metrics from college predict 
                NFL rookie production. This directly answers: "Which college stats translate to the pros?"
                
                ### Key Findings
                
                - **Model R²:** {test_r2:.3f}
                - **Prediction Error:** {test_mae:.2f}
                - **Sample Size:** {len(current_data.get('processed_data', []))} players
                - **vs. Combine R²:** {all_results.get('combine', {}).get('metrics', {}).get('test_r2', -0.155) if all_results.get('combine') else 'N/A'}
                
                ### What This Means
                
                College tracking metrics successfully predict NFL success, validating that 
                in-game performance matters more than isolated athletic testing.
                
                ### Football Translation
                
                **Why This Matters for Teams:**
                - Identifies receivers who create separation consistently (not just run fast)
                - Reveals route-running precision that translates to NFL concepts
                - Shows YAC ability that predicts explosive play potential
                - Captures performance under pressure (tight windows, man coverage)
                - Measures skills that combine testing completely misses
                
                **Scouting Insights:**
                - High separation metrics → Can win at catch point
                - Route diversity → NFL versatility across formations
                - CPOE/YACOE → Reliable playmaker, not just volume receiver
                - COD efficiency → Can run full route tree at NFL speed
                """)
            
            # Key metrics info instead of sample data
            st.markdown("### What The Model Captures")
            st.markdown("""
            This analysis uses **in-game tracking data** that measures:
            - **Separation metrics**: Distance from defenders at key moments
            - **Speed profiles**: Max speed, acceleration patterns, route-specific speeds
            - **Route efficiency**: Change of direction ability, route depth consistency
            - **Performance context**: Success against man coverage, tight window catches
            - **Playmaking**: YAC over expected, catch percentage over expected
            
            These metrics reveal **how players actually perform in game situations**, 
            not just isolated athletic traits.
            """)
        
        with col2:
            st.markdown(f"""
            ### Key Statistics
            
            **Model Performance:**
            - R² Score: **{test_r2:.3f}**
            - MAE: **{test_mae:.2f}**
            - RMSE: **{metrics.get('test_rmse', 0):.2f}**
            
            **Dataset:**
            - Players: **{len(current_data.get('processed_data', []))}**
            - Features: **{len(current_data.get('feature_importance', []))}**
            
            ---
            
            ### Tracking Advantage
            
            Unlike combine testing, tracking data measures:
            
            ✅ Real-game speed  
            ✅ Separation ability  
            ✅ Route efficiency  
            ✅ Coverage performance  
            ✅ Explosive plays  
            ✅ Consistency  
            
            **Result:** Better predictions than static testing alone.
            """)
        
        st.markdown("---")
        
        # Comparison with combine (if available)
        if all_results['combine']:
            combine_r2 = all_results['combine']['metrics']['test_r2']
            tracking_r2 = test_r2
            
            improvement = ((tracking_r2 - combine_r2) / abs(combine_r2) * 100) if combine_r2 != 0 else 0
            
            st.markdown(f"""
            <div class="success-box">
            <h2 style="color: white; margin-top: 0;">📊 Tracking Data vs. Combine Testing</h2>
            <h3 style="color: white;">Tracking data achieves {abs(improvement):.0f}% better predictive accuracy</h3>
            <p style="font-size: 1.1rem;">
            <strong>Combine R²:</strong> {combine_r2:.3f} (explains {max(0, combine_r2*100):.1f}% of variance)<br>
            <strong>Tracking R²:</strong> {tracking_r2:.3f} (explains {tracking_r2*100:.1f}% of variance)<br>
            <strong>Improvement:</strong> {abs(improvement):.0f}% better prediction
            </p>
            <p>This demonstrates that in-game performance metrics capture what matters for NFL success.</p>
            </div>
            """, unsafe_allow_html=True)
    
    else:  # Combine comparison
        st.info("Select 'Draft Prediction' or 'NFL Rookie Performance' to see tracking data analysis.")

# ============================================================================
# PAGE 2: PERFORMANCE COMPARISON
# ============================================================================
elif page == "Performance Comparison":
    st.header("Model Performance Comparison")
    
    st.markdown("""
    Compare the predictive power of tracking data across different targets:
    - **Draft Prediction**: How well college tracking predicts draft capital
    - **NFL Rookie Performance**: How well college tracking predicts NFL production
    """)
    
    # Check if we have both analyses
    has_draft = all_results.get('draft') is not None
    has_nfl = all_results.get('nfl') is not None
    
    if has_draft and has_nfl:
        # Get metrics for both
        draft_metrics = all_results['draft'].get('metrics', {})
        nfl_metrics = all_results['nfl'].get('metrics', {})
        
        draft_r2 = draft_metrics.get('test_r2', 0)
        nfl_r2 = nfl_metrics.get('test_r2', 0)
        
        # Comparison metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
            <h3 style="color: white; margin: 0;">Draft Prediction</h3>
            <h2 style="color: white; margin: 10px 0;">{:.3f}</h2>
            <p style="color: white; margin: 0;">R² Score</p>
            </div>
            """.format(draft_r2), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="success-box">
            <h3 style="color: white; margin: 0;">NFL Performance</h3>
            <h2 style="color: white; margin: 10px 0;">{:.3f}</h2>
            <p style="color: white; margin: 0;">R² Score</p>
            </div>
            """.format(nfl_r2), unsafe_allow_html=True)
        
        with col3:
            better = "Draft" if draft_r2 > nfl_r2 else "NFL"
            diff = abs(draft_r2 - nfl_r2)
            st.markdown("""
            <div class="tracking-box">
            <h3 style="color: white; margin: 0;">Better Target</h3>
            <h2 style="color: white; margin: 10px 0;">{}</h2>
            <p style="color: white; margin: 0;">Δ = {:.3f}</p>
            </div>
            """.format(better, diff), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Side-by-side comparison chart
        st.subheader("Predictive Power Comparison")
        
        comparison_df = pd.DataFrame({
            'Target': ['Draft Capital', 'NFL Production'],
            'R² Score': [draft_r2, nfl_r2],
            'Variance Explained (%)': [draft_r2*100, nfl_r2*100],
            'Variance Unexplained (%)': [(1-draft_r2)*100, (1-nfl_r2)*100]
        })
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Variance Explained',
            x=comparison_df['Target'],
            y=comparison_df['Variance Explained (%)'],
            marker_color='#2ecc71',
            text=comparison_df['Variance Explained (%)'].apply(lambda x: f'{x:.1f}%'),
            textposition='inside'
        ))
        
        fig.add_trace(go.Bar(
            name='Variance Unexplained',
            x=comparison_df['Target'],
            y=comparison_df['Variance Unexplained (%)'],
            marker_color='#e74c3c',
            text=comparison_df['Variance Unexplained (%)'].apply(lambda x: f'{x:.1f}%'),
            textposition='inside'
        ))
        
        fig.update_layout(
            barmode='stack',
            title='Tracking Data: Draft vs NFL Performance Prediction',
            yaxis_title='Percentage (%)',
            height=450,
            template='plotly_white',
            showlegend=True
        )
        
        st.plotly_chart(fig, width='stretch')
        
        st.markdown("---")
        
        # What each predicts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Draft Capital Prediction
            
            **Target Variable:** Draft round/pick selection
            
            **What High R² Means:**
            - Tracking metrics align with NFL scout evaluations
            - College performance translates to draft stock
            - Objective data validates subjective scouting
            
            **Use Cases:**
            - Find draft value (over/undervalued prospects)
            - Project where players will be selected
            - Identify traits scouts prioritize
            """)
        
        with col2:
            st.markdown("""
            ### NFL Production Prediction
            
            **Target Variable:** NFL rookie performance metrics
            
            **What High R² Means:**
            - College skills directly translate to pros
            - Tracking captures NFL-translatable traits
            - Can forecast rookie impact
            
            **Use Cases:**
            - Predict actual NFL performance
            - Identify high-floor prospects
            - Build draft models based on outcomes, not opinions
            """)
        
        # Football insights
        st.markdown("---")
        st.subheader("Strategic Insights")
        
        if draft_r2 > nfl_r2:
            st.markdown(f"""
            <div class="insight-box">
            <h3 style="color: white; margin-top: 0;">Draft Prediction is Stronger</h3>
            <p style="font-size: 1.1rem;">
            Draft capital (R²={draft_r2:.3f}) is more predictable than NFL production (R²={nfl_r2:.3f}).
            </p>
            <p><strong>What This Means:</strong></p>
            <ul>
                <li>NFL scouts value the same traits that tracking captures</li>
                <li>College metrics align well with draft evaluations</li>
                <li>Gap between draft and NFL suggests other factors matter (scheme fit, coaching, opportunity)</li>
                <li><strong>Opportunity:</strong> Find players drafted lower than tracking suggests (value picks)</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="success-box">
            <h3 style="color: white; margin-top: 0;">NFL Performance is More Predictable</h3>
            <p style="font-size: 1.1rem;">
            NFL production (R²={nfl_r2:.3f}) is more predictable than draft capital (R²={draft_r2:.3f}).
            </p>
            <p><strong>What This Means:</strong></p>
            <ul>
                <li>College tracking metrics directly translate to NFL success</li>
                <li>NFL scouts may be missing traits that matter</li>
                <li>Draft doesn't always reflect who will succeed in NFL</li>
                <li><strong>Opportunity:</strong> Identify undervalued prospects who will outperform draft position</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
    elif has_draft or has_nfl:
        st.info("""
        Only one analysis type is currently loaded. 
        
        To enable comparison:
        - Run both `run_analysis_draft.py` and `run_analysis_nfl.py`
        - Both will use the same college tracking data
        - One predicts draft capital, the other predicts NFL performance
        """)
    else:
        st.error("No analysis data available for comparison.")

# ============================================================================
# PAGE 3: DATA EXPLORER
# ============================================================================
elif page == "Data Explorer":
    st.header("Data Explorer")
    
    if 'processed_data' in current_data:
        df = current_data['processed_data']
        
        tab1, tab2, tab3 = st.tabs(["📊 Distributions", "🔗 Correlations", "🎯 Feature Importance"])
        
        with tab1:
            st.subheader("Metric Distributions")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Filter out columns with no data
            valid_cols = []
            for col in numeric_cols:
                if df[col].notna().sum() > 0:  # Has at least some data
                    valid_cols.append(col)
            
            if valid_cols:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    selected_metric = st.selectbox(
                        "Select metric to visualize:",
                        valid_cols,
                        format_func=lambda x: x.replace('_', ' ').title()
                    )
                
                with col2:
                    bins = st.slider("Number of bins:", 10, 50, 30)
                
                fig = px.histogram(
                    df,
                    x=selected_metric,
                    nbins=bins,
                    title=f"Distribution of {selected_metric.replace('_', ' ').title()}",
                    color_discrete_sequence=['#667eea']
                )
                fig.update_layout(height=450, template='plotly_white', showlegend=False)
                st.plotly_chart(fig, width='stretch')
                
                # Statistics
                col1, col2, col3, col4, col5 = st.columns(5)
                valid_data = df[selected_metric].dropna()
                
                col1.metric("Mean", f"{valid_data.mean():.2f}")
                col2.metric("Median", f"{valid_data.median():.2f}")
                col3.metric("Std Dev", f"{valid_data.std():.2f}")
                col4.metric("Min", f"{valid_data.min():.2f}")
                col5.metric("Max", f"{valid_data.max():.2f}")
            else:
                st.warning("No valid numeric metrics found in dataset")
        
        with tab2:
            st.subheader("Correlation Analysis")
            
            if len(numeric_cols) > 1:
                # Select subset of key features
                key_features = numeric_cols[:15] if len(numeric_cols) > 15 else numeric_cols
                corr_matrix = df[key_features].corr()
                
                fig = px.imshow(
                    corr_matrix,
                    labels=dict(color="Correlation"),
                    x=[m.replace('_', ' ').title() for m in corr_matrix.columns],
                    y=[m.replace('_', ' ').title() for m in corr_matrix.columns],
                    color_continuous_scale='RdBu_r',
                    zmin=-1,
                    zmax=1,
                    aspect='auto'
                )
                fig.update_layout(height=600, title="Feature Correlation Heatmap")
                st.plotly_chart(fig, width='stretch')
        
        with tab3:
            st.subheader("Feature Importance")
            
            if 'feature_importance' in current_data:
                feat_imp = current_data['feature_importance'].head(15)
                
                fig = px.bar(
                    feat_imp,
                    x='importance',
                    y='feature',
                    orientation='h',
                    title="Top 15 Most Important Features",
                    color='importance',
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(height=500, showlegend=False, template='plotly_white')
                fig.update_yaxes(title='')
                st.plotly_chart(fig, width='stretch')
                
                st.dataframe(feat_imp, hide_index=True, width='stretch')

# ============================================================================
# PAGE 4: ML MODEL RESULTS
# ============================================================================
elif page == "ML Model Results":
    st.header("Machine Learning Model Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("R² Score", f"{test_r2:.3f}")
    col2.metric("MAE", f"{test_mae:.2f}")
    col3.metric("RMSE", f"{metrics.get('test_rmse', 0):.2f}")
    col4.metric("CV R²", f"{metrics.get('cv_mean', 0):.3f}")
    
    st.markdown("---")
    
    # Model interpretation with football context
    if test_r2 > 0.3:
        quality = "Excellent"
        color_class = "success-box"
        interpretation = "Strong predictive relationship. Model captures key traits that translate to target outcome."
    elif test_r2 > 0.15:
        quality = "Good"
        color_class = "tracking-box"
        interpretation = "Solid predictive power. Model identifies meaningful patterns in tracking data."
    else:
        quality = "Moderate"
        color_class = "insight-box"
        interpretation = "Moderate predictive power. Other factors beyond tracking contribute to outcomes."
    
    st.markdown(f"""
    <div class="{color_class}">
    <h3 style="color: white; margin-top: 0;">Model Performance: {quality}</h3>
    <p style="font-size: 1.1rem;">
    R² of {test_r2:.3f} means the model explains <strong>{test_r2*100:.1f}%</strong> of variance in the target variable.
    </p>
    <p><strong>{interpretation}</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Football context based on analysis type
    st.markdown("---")
    st.subheader("What This Means for Football Strategy")
    
    if analysis_type == 'Draft Prediction':
        st.markdown(f"""
        ### Scouting & Draft Strategy
        
        **Model Insight (R² = {test_r2:.3f}):**
        
        With {test_r2*100:.1f}% of draft capital explained by tracking metrics, teams can:
        
        1. **Find Value Picks**: Identify prospects whose tracking data suggests higher draft capital than consensus
        2. **Validate Scouting**: Use objective data to confirm/challenge subjective evaluations
        3. **Trade Strategy**: Know when to trade up for players tracking data strongly supports
        4. **Risk Assessment**: Quantify uncertainty in draft projections
        
        **Key Tracking Metrics That Drive Draft Capital:**
        - Separation ability (scouts prioritize this heavily)
        - Route diversity (shows NFL versatility)
        - Performance vs. man coverage (ultimate test)
        - Speed in context (not just 40-time, but game speed with direction changes)
        
        **The {(1-test_r2)*100:.1f}% Gap:**
        - Positional scarcity/need
        - Intangibles (leadership, work ethic)
        - Medical/character concerns
        - Scheme fit for specific teams
        - Draft day supply/demand dynamics
        """)
    
    elif analysis_type == 'NFL Rookie Performance':
        st.markdown(f"""
        ### Player Development & Roster Strategy
        
        **Model Insight (R² = {test_r2:.3f}):**
        
        With {test_r2*100:.1f}% of NFL production explained by college tracking, teams can:
        
        1. **Forecast Rookie Impact**: Project first-year contribution with quantified confidence
        2. **Free Agent Strategy**: Identify undervalued prospects other teams missed
        3. **Development Plans**: Know which skills translate vs. which need coaching
        4. **Roster Construction**: Build depth chart based on projected production
        
        **College Metrics That Predict NFL Success:**
        - Consistent separation (translates directly to NFL targets)
        - YAC ability (identifies true playmakers vs. volume receivers)
        - Tight window success (predicts success against NFL press coverage)
        - Route efficiency at depth (shows ability to stretch NFL defenses)
        
        **The {(1-test_r2)*100:.1f}% Gap:**
        - Opportunity (volume depends on team situation)
        - Scheme fit (some offenses feature WRs more)
        - QB play (great QBs elevate WR stats)
        - Adjustment period (learning NFL speed/complexity)
        - Health/durability factors
        """)
    
    st.markdown("---")
    if 'predictions' in current_data:
        st.subheader("Actual vs. Predicted Values")
        
        pred_data = current_data['predictions']
        
        fig = go.Figure()
        
        # Test set
        fig.add_trace(go.Scatter(
            x=pred_data['y_test'],
            y=pred_data['y_pred_test'],
            mode='markers',
            name='Test Set',
            marker=dict(size=8, color='#667eea', opacity=0.6)
        ))
        
        # Perfect prediction line
        min_val = min(pred_data['y_test'].min(), pred_data['y_pred_test'].min())
        max_val = max(pred_data['y_test'].max(), pred_data['y_pred_test'].max())
        
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash', width=2)
        ))
        
        fig.update_layout(
            title=f'Actual vs. Predicted (R² = {test_r2:.3f})',
            xaxis_title='Actual Values',
            yaxis_title='Predicted Values',
            height=500,
            template='plotly_white',
            showlegend=True
        )
        
        st.plotly_chart(fig, width='stretch')

# ============================================================================
# PAGE 5: PLAYER INSIGHTS
# ============================================================================
elif page == "Player Insights":
    st.header("Player-Level Insights")
    
    if 'processed_data' in current_data:
        df = current_data['processed_data']
        
        if 'player_name' in df.columns:
            # Player selector
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Clean player names and filter out NaN/None
                valid_players = df['player_name'].dropna().unique()
                # Convert to strings and filter out any non-string values
                valid_players = [str(p) for p in valid_players if pd.notna(p)]
                
                if len(valid_players) > 0:
                    player_name = st.selectbox(
                        "Select Player:",
                        sorted(valid_players)
                    )
                else:
                    st.warning("No valid player names found in dataset")
                    st.stop()
            
            with col2:
                st.markdown("")
                st.markdown("")
                if st.button("Random Player"):
                    player_name = np.random.choice(df['player_name'].unique())
            
            # Get player data
            player_data = df[df['player_name'] == player_name].iloc[0]
            
            st.markdown(f"## {player_name}")
            st.markdown("---")
            
            # Display key metrics
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Define key metrics based on analysis type
            if analysis_type == 'NFL Rookie Performance':
                # For NFL analysis, show NFL production metrics
                key_metrics = [c for c in ['targets_per_game', 'receptions', 'rec_yards', 'rec_tds', 'yards_per_rec', 'avg_separation', 'max_speed'] if c in numeric_cols][:6]
            else:
                # For draft analysis, show college tracking metrics
                key_metrics = [c for c in ['targets', 'receptions', 'avg_separation', 'max_speed', 'draft_pick', 'draft_round'] if c in numeric_cols][:6]
            
            if not key_metrics:
                key_metrics = numeric_cols[:6]  # Fallback to first 6 numeric columns
            
            cols = st.columns(len(key_metrics))
            for idx, metric in enumerate(key_metrics):
                if metric in player_data:
                    cols[idx].metric(
                        metric.replace('_', ' ').title(),
                        f"{player_data[metric]:.2f}" if isinstance(player_data[metric], (int, float)) else player_data[metric]
                    )
            
            st.markdown("---")
            
            # Percentile rankings
            st.subheader("Percentile Rankings vs. Dataset")
            
            percentiles = []
            labels = []
            
            for metric in key_metrics[:6]:
                if metric in df.columns and pd.notna(player_data.get(metric)):
                    percentile = (df[metric] < player_data[metric]).sum() / len(df[metric].dropna()) * 100
                    percentiles.append(percentile)
                    
                    # Create readable labels
                    label = metric.replace('_', ' ').title()
                    if analysis_type == 'NFL Rookie Performance':
                        # Add context for NFL metrics
                        if 'per_game' in metric:
                            label = label.replace('Per Game', '/G')
                        elif metric == 'rec_yards':
                            label = 'Rec Yards'
                        elif metric == 'rec_tds':
                            label = 'Rec TDs'
                    labels.append(label)
            
            if percentiles:
                fig = go.Figure()
                
                fig.add_trace(go.Scatterpolar(
                    r=percentiles,
                    theta=labels,
                    fill='toself',
                    fillcolor='rgba(102, 126, 234, 0.5)',
                    line=dict(color='rgb(102, 126, 234)', width=2)
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100],
                            ticksuffix='%'
                        )
                    ),
                    showlegend=False,
                    height=400,
                    title="Player Rankings vs. Dataset"
                )
                
                st.plotly_chart(fig, width='stretch')
            else:
                st.info("Insufficient metrics available for radar chart")
                
            # Add player insights based on analysis type
            st.markdown("---")
            st.subheader("Player Profile Insights")
            
            if analysis_type == 'NFL Rookie Performance':
                st.markdown(f"""
                **NFL Production Analysis:**
                
                This player's college tracking metrics suggest:
                """)
                
                # Analyze key NFL metrics
                if 'targets_per_game' in player_data and pd.notna(player_data['targets_per_game']):
                    tpg = player_data['targets_per_game']
                    tpg_pct = (df['targets_per_game'] < tpg).sum() / len(df['targets_per_game'].dropna()) * 100
                    
                    if tpg_pct > 75:
                        st.markdown(f"**High-Volume Receiver**: {tpg:.1f} targets/game ({tpg_pct:.0f}th percentile) - Projects as featured target in NFL")
                    elif tpg_pct > 50:
                        st.markdown(f"➡️ **Solid Target Share**: {tpg:.1f} targets/game - Projects as reliable WR2/WR3")
                    else:
                        st.markdown(f"**Limited Volume**: {tpg:.1f} targets/game - May need development or better opportunity")
                
                if 'avg_separation' in player_data and pd.notna(player_data['avg_separation']):
                    sep = player_data['avg_separation']
                    sep_pct = (df['avg_separation'] < sep).sum() / len(df['avg_separation'].dropna()) * 100
                    
                    if sep_pct > 70:
                        st.markdown(f"**Elite Separator**: {sep:.2f} yards avg separation - Can win at NFL catch point")
                    elif sep_pct > 40:
                        st.markdown(f"➡️ **Adequate Separation**: {sep:.2f} yards - NFL-caliber route runner")
                
            elif analysis_type == 'Draft Prediction':
                st.markdown(f"""
                **Draft Profile Analysis:**
                
                Based on college tracking metrics:
                """)
                
                if 'draft_pick' in player_data and pd.notna(player_data['draft_pick']):
                    pick = player_data['draft_pick']
                    round_num = player_data.get('draft_round', 'Unknown')
                    
                    st.markdown(f"**Actual Draft Position**: Round {round_num}, Pick {pick}")
                
                if 'avg_separation' in player_data and pd.notna(player_data['avg_separation']):
                    sep = player_data['avg_separation']
                    sep_pct = (df['avg_separation'] < sep).sum() / len(df['avg_separation'].dropna()) * 100
                    
                    if sep_pct > 75:
                        st.markdown(f"**Top Separator**: {sep:.2f} yards - Trait scouts prioritize heavily")
                    
                if 'max_speed' in player_data and pd.notna(player_data['max_speed']):
                    speed = player_data['max_speed']
                    speed_pct = (df['max_speed'] < speed).sum() / len(df['max_speed'].dropna()) * 100
                    
                    if speed_pct > 80:
                        st.markdown(f"**Elite Game Speed**: {speed:.1f} mph in-game - Elite deep threat potential")
        else:
            st.info("Player names not available in dataset")
    else:
        st.info("Processed data not available for player insights")

# Footer
st.markdown("---")
st.markdown("---")

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown(f"""
    <div style='text-align: center; padding: 2rem 0;'>
        <h4>Tracking Prophet</h4>
        <p style='color: #888;'>
            NFL Wide Receiver Performance Prediction via In-Game Tracking Data<br>
            Predicting {analysis_type if 'analysis_type' in locals() else 'NFL Success'} from College Metrics
        </p>
        <p style='color: #888; margin-top: 1rem;'>
            <strong>Analysis Types Available:</strong><br>
            {', '.join(available_analyses) if available_analyses else 'None'}
        </p>
        <p style='margin-top: 1rem;'>
            <a href='https://github.com/hemaniprisha' target='_blank' style='margin: 0 10px; color: #667eea; text-decoration: none;'>GitHub</a> | 
            <a href='https://www.linkedin.com/in/prisha-hemani-4194a8257/' target='_blank' style='margin: 0 10px; color: #667eea; text-decoration: none;'>LinkedIn</a> | 
            <a href='mailto:hemaniprisha1@gmail.com' style='margin: 0 10px; color: #667eea; text-decoration: none;'>Contact</a>
        </p>
    </div>
    """, unsafe_allow_html=True)