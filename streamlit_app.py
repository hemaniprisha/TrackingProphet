"""
Tracking Prophet - Interactive Dashboard
Predicts NFL WR performance from college tracking data (because 40-times don't tell the whole story)
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
import logging

# Page config - wide layout works better for charts
st.set_page_config(
    page_title="Tracking Prophet | NFL Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling - gradients make everything look fancier
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

@st.cache_data
def load_pkl(fpath):
    """Load pickled results - cached so we don't reload every interaction."""
    if os.path.exists(fpath):
        with open(fpath, 'rb') as f:
            return pickle.load(f)
    return None

@st.cache_data
def get_all_data():
    """Try to load all three analysis types if they exist."""
    res = {'draft': None, 'nfl': None, 'combine': None}
    
    if os.path.exists('tracking_draft_export.pkl'):
        res['draft'] = load_pkl('tracking_draft_export.pkl')
    
    if os.path.exists('tracking_nfl_export.pkl'):
        res['nfl'] = load_pkl('tracking_nfl_export.pkl')
    
    if os.path.exists('combine_analysis_export.pkl'):
        res['combine'] = load_pkl('combine_analysis_export.pkl')
    
    return res

data_pkg = get_all_data()
has_anything = any(data_pkg.values())

if not has_anything:
    st.error("No data found. Run the analysis scripts first.")
    st.info("""
    **Generate data files:**
    
    1. College Tracking → Draft: `python run_analysis_draft.py --data tracking.csv`
    2. College Tracking → NFL: `python run_analysis_nfl.py --data tracking.csv`
    3. Combine (optional): `python combine_analysis.py`
    """)
    st.stop()

# Build available analyses list
avail = []
if data_pkg['draft']:
    avail.append('Draft Prediction')
if data_pkg['nfl']:
    avail.append('NFL Rookie Performance')
if data_pkg['combine']:
    avail.append('Combine (Comparison)')

# Header
st.markdown('<div class="main-header">🎯 Tracking Prophet</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predicting NFL Success with In-Game Tracking Data</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/a/a2/National_Football_League_logo.svg/400px-National_Football_League_logo.svg.png", width=180)
    
    st.title("Navigation")
    
    analysis_sel = st.radio("Select Analysis Type:", avail, help="Choose analysis to explore")
    # Resolve current dataset EARLY (needed by sidebar)
    curr = None
    if analysis_sel == 'Draft Prediction':
        curr = data_pkg.get('draft')
    elif analysis_sel == 'NFL Rookie Performance':
        curr = data_pkg.get('nfl')
    elif analysis_sel == 'Combine (Comparison)':
        curr = data_pkg.get('combine')

    # Safe defaults
    mets = {}
    r2 = 0.0
    mae = 0.0

    if curr:
        mets = curr.get('metrics', {})
        r2 = mets.get('test_r2', 0.0)
        mae = mets.get('test_mae', mets.get('mae', 0.0))

    st.markdown("---")
    pg = st.radio("Navigate To:", ["Overview", "Performance Comparison", "Data Explorer", "ML Model Results", "Player Insights"], label_visibility="collapsed")
    st.markdown("---")
    
    if analysis_sel == 'Draft Prediction' and data_pkg['draft']:
        d = data_pkg['draft']
        st.markdown(f"""
        ### Draft Analysis
        **Dataset:** College Tracking  
        **Target:** Draft Position  
        **Sample:** {len(d.get('processed_data', []))} players  
        **R²:** {d.get('metrics', {}).get('test_r2', 0):.3f}  
        **MAE:** {d.get('metrics', {}).get('test_mae', 0):.2f}
        
        ---
        **Insight:** Tracking captures real performance scouts value.
        """)
    elif analysis_sel == 'NFL Rookie Performance' and data_pkg['nfl']:
        d = data_pkg['nfl']
        st.markdown(f"""
        ### NFL Performance
        **Dataset:** College → NFL  
        **Target:** NFL Production  
        **Sample:** {len(d.get('processed_data', []))} players  
        **R²:** {d.get('metrics', {}).get('test_r2', 0):.3f}  
        **MAE:** {d.get('metrics', {}).get('test_mae', 0):.2f}
        
        ---
        **Insight:** College tracking beats combine testing.
        """)
    elif analysis_sel == 'Combine (Comparison)' and data_pkg['combine']:
        d = data_pkg['combine']
        st.markdown(f"""
        ### Combine Testing
        **Dataset:** Combine Metrics  
        **Target:** NFL Performance  
        **Sample:** {len(d.get('merged_data', []))} players  
        **R²:** {d.get('metrics', {}).get('test_r2', 0):.3f}  
        **MAE:** {d.get('metrics', {}).get('mae', 0):.1f} ypg
        
        ---
        **Limitation:** Static tests miss game skills.
        """)

# Get current dataset
# Resolve current dataset based on selection

if curr:
    c1, c2, c3, c4 = st.columns(4)

    # Total records
    if 'processed_data' in curr:
        df = curr['processed_data']
    else:
        df = curr.get('merged_data')

    tot = len(df) if df is not None else 0

    with c1:
        st.metric("Total Records", f"{tot:,}")

    # Unique players
    with c2:
        if df is not None and 'player_name' in df.columns:
            uniq = df['player_name'].nunique()
        else:
            uniq = tot
        st.metric("Unique Players", f"{uniq:,}")

    # Model performance
    with c3:
        st.metric("Model R²", f"{r2:.3f}")

    with c4:
        lbl = "MAE" if analysis_sel != 'Combine (Comparison)' else "MAE (ypg)"
        st.metric(lbl, f"{mae:.2f}")

    st.markdown("---")

# PAGE 1: OVERVIEW
if pg == "Overview":
    st.header("Project Overview")
    
    if analysis_sel in ['Draft Prediction', 'NFL Rookie Performance']:
        c1, c2 = st.columns([2, 1])
        
        with c1:
            if analysis_sel == 'Draft Prediction':
                st.markdown(f"""
                ### Research Question
                
                **"Can college tracking predict NFL draft position?"**
                
                Uses in-game tracking from college to predict draft position.
                Unlike combine, tracking captures:
                
                - Real-game separation ability
                - Route running precision
                - Performance vs coverage types
                - Game consistency
                - Explosive play creation
                
                ### Key Findings
                
                - **R²:** {r2:.3f}
                - **Error:** {mae:.2f}
                - **Sample:** {len(curr.get('processed_data', []))} players
                
                ### What This Means
                
                Tracking provides objective metrics scouts look for.
                Bridges subjective evaluation with data.
                """)
            else:  # NFL performance
                st.markdown(f"""
                ### Research Question
                
                **"Can college tracking predict NFL rookie success?"**
                
                Tests if in-game college metrics translate to pros.
                Answers: "Which college stats matter in NFL?"
                
                ### Key Findings
                
                - **R²:** {r2:.3f}
                - **Error:** {mae:.2f}
                - **Sample:** {len(curr.get('processed_data', []))} players
                - **vs Combine:** {data_pkg.get('combine', {}).get('metrics', {}).get('test_r2', -0.155) if data_pkg.get('combine') else 'N/A'}
                
                ### Football Translation
                
                **Why Teams Care:**
                - Identifies true separators (not just fast)
                - Shows route precision for NFL concepts
                - Reveals YAC ability = explosive plays
                - Captures tight window performance
                - Measures skills combine misses
                
                **Scouting Value:**
                - High separation → wins at catch point
                - Route diversity → NFL versatility
                - CPOE/YACOE → reliable playmaker
                - COD efficiency → full route tree
                """)
            
            st.markdown("### What Model Captures")
            st.markdown("""
            Uses **real game tracking**:
            - **Separation**: Distance from defenders
            - **Speed profiles**: Max speed, acceleration, route speeds
            - **Route efficiency**: COD ability, depth consistency
            - **Context**: Man coverage, tight windows
            - **Playmaking**: YAC/catch % over expected
            
            Reveals **actual game performance**, not just athletic traits.
            """)
        
        with c2:
            st.markdown(f"""
            ### Key Stats
            
            **Model:**
            - R²: **{r2:.3f}**
            - MAE: **{mae:.2f}**
            - RMSE: **{mets.get('test_rmse', 0):.2f}**
            
            **Dataset:**
            - Players: **{len(curr.get('processed_data', []))}**
            - Features: **{len(curr.get('feature_importance', []))}**
            
            ---
            
            ### Tracking Advantage
            
            Unlike combine:
            
            - Real game speed  
            - Separation creation  
            - Route precision  
            - Coverage performance  
            - Explosive ability  
            - Consistency  
            
            **Result:** Better than static tests.
            """)
        
        st.markdown("---")
        
        # Compare to combine if available
        if data_pkg['combine']:
            comb_r2 = data_pkg['combine']['metrics']['test_r2']
            track_r2 = r2
            improve = ((track_r2 - comb_r2) / abs(comb_r2) * 100) if comb_r2 != 0 else 0
            
            st.markdown(f"""
            <div class="success-box">
            <h2 style="color: white; margin-top: 0;">Tracking vs Combine</h2>
            <h3 style="color: white;">Tracking achieves {abs(improve):.0f}% better accuracy</h3>
            <p style="font-size: 1.1rem;">
            <strong>Combine R²:</strong> {comb_r2:.3f} (explains {max(0, comb_r2*100):.1f}%)<br>
            <strong>Tracking R²:</strong> {track_r2:.3f} (explains {track_r2*100:.1f}%)<br>
            <strong>Improvement:</strong> {abs(improve):.0f}% better
            </p>
            <p>In-game metrics capture what matters for NFL success.</p>
            </div>
            """, unsafe_allow_html=True)
        st.subheader("Football Strategy Implications")
    
    if analysis_sel == 'Draft Prediction':
        st.markdown(f"""
        ### Scouting & Draft
        
        **Model Insight (R² = {r2:.3f}):**
        
        With {r2*100:.1f}% of draft capital explained:
        
        1. **Value Picks**: Find prospects with better tracking than consensus
        2. **Validate Scouting**: Objective data confirms/challenges evaluations
        3. **Trade Strategy**: Know when to trade up based on data
        4. **Risk Assessment**: Quantify projection uncertainty
        
        **Tracking Metrics Scouts Value:**
        - Separation ability (top priority)
        - Route diversity (versatility)
        - Man coverage performance (ultimate test)
        - Functional game speed (not just 40-time)
        
        **The {(1-r2)*100:.1f}% Gap:**
        - Positional scarcity
        - Intangibles (leadership, etc.)
        - Medical/character flags
        - Scheme fit
        - Draft dynamics
        """)
    
    elif analysis_sel == 'NFL Rookie Performance':
        st.markdown(f"""
        ### Player Development & Roster
        
        **Model Insight (R² = {r2:.3f}):**
        
        With {r2*100:.1f}% of NFL production explained:
        
        1. **Forecast Impact**: Project rookie contribution
        2. **FA Strategy**: Find undervalued prospects
        3. **Development**: Know what translates vs needs coaching
        4. **Roster Build**: Depth chart based on projections
        
        **College Metrics That Translate:**
        - Consistent separation (→ NFL targets)
        - YAC ability (true playmakers)
        - Tight window success (vs press coverage)
        - Route efficiency at depth (stretch defense)
        
        **The {(1-r2)*100:.1f}% Gap:**
        - Opportunity (volume varies by team)
        - Scheme fit (WR usage)
        - QB play (elevates WR stats)
        - Adjustment period (NFL speed)
        - Health/durability
        """)

    else:
        st.info("Select 'Draft Prediction' or 'NFL Rookie Performance' for tracking analysis.")

# PAGE 2: PERFORMANCE COMPARISON
elif pg == "Performance Comparison":
    st.header("Model Performance Comparison")
    
    st.markdown("""
    Compare tracking data across targets:
    - **Draft**: College tracking → draft capital
    - **NFL**: College tracking → NFL production
    """)
    
    has_draft = data_pkg.get('draft') is not None
    has_nfl = data_pkg.get('nfl') is not None
    
    if has_draft and has_nfl:
        draft_r2 = data_pkg['draft'].get('metrics', {}).get('test_r2', 0)
        nfl_r2 = data_pkg['nfl'].get('metrics', {}).get('test_r2', 0)
        
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.markdown(f"""
            <div class="metric-card">
            <h3 style="color: white; margin: 0;">Draft Prediction</h3>
            <h2 style="color: white; margin: 10px 0;">{draft_r2:.3f}</h2>
            <p style="color: white; margin: 0;">R² Score</p>
            </div>
            """, unsafe_allow_html=True)
        
        with c2:
            st.markdown(f"""
            <div class="success-box">
            <h3 style="color: white; margin: 0;">NFL Performance</h3>
            <h2 style="color: white; margin: 10px 0;">{nfl_r2:.3f}</h2>
            <p style="color: white; margin: 0;">R² Score</p>
            </div>
            """, unsafe_allow_html=True)
        
        with c3:
            winner = "Draft" if draft_r2 > nfl_r2 else "NFL"
            gap = abs(draft_r2 - nfl_r2)
            st.markdown(f"""
            <div class="tracking-box">
            <h3 style="color: white; margin: 0;">Better Target</h3>
            <h2 style="color: white; margin: 10px 0;">{winner}</h2>
            <p style="color: white; margin: 0;">Δ = {gap:.3f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Stacked bar chart
        st.subheader("Predictive Power")
        
        comp_df = pd.DataFrame({
            'Target': ['Draft Capital', 'NFL Production'],
            'Explained (%)': [draft_r2*100, nfl_r2*100],
            'Unexplained (%)': [(1-draft_r2)*100, (1-nfl_r2)*100]
        })
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Explained',
            x=comp_df['Target'],
            y=comp_df['Explained (%)'],
            marker_color='#2ecc71',
            text=comp_df['Explained (%)'].apply(lambda x: f'{x:.1f}%'),
            textposition='inside'
        ))
        
        fig.add_trace(go.Bar(
            name='Unexplained',
            x=comp_df['Target'],
            y=comp_df['Unexplained (%)'],
            marker_color='#e74c3c',
            text=comp_df['Unexplained (%)'].apply(lambda x: f'{x:.1f}%'),
            textposition='inside'
        ))
        
        fig.update_layout(
            barmode='stack',
            title='Tracking Data: Draft vs NFL Prediction',
            yaxis_title='Percentage (%)',
            height=450,
            template='plotly_white',
            showlegend=True
        )
        
        st.plotly_chart(fig, width='stretch')
        
        st.markdown("---")
        
        # What each predicts
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("""
            ### Draft Capital Prediction
            
            **Target:** Draft round/pick
            
            **High R² means:**
            - Tracking aligns with scout evals
            - College performance → draft stock
            - Objective data validates scouting
            
            **Use cases:**
            - Find draft value
            - Project selection range
            - Identify scout priorities
            """)
        
        with c2:
            st.markdown("""
            ### NFL Production Prediction
            
            **Target:** NFL rookie stats
            
            **High R² means:**
            - College skills translate
            - Tracking captures NFL traits
            - Can forecast rookie impact
            
            **Use cases:**
            - Predict actual performance
            - Identify high-floor prospects
            - Build outcome-based models
            """)
        
    elif has_draft or has_nfl:
        st.info("Only one analysis loaded. Run both scripts for comparison.")
    else:
        st.error("No data for comparison.")

# PAGE 3: DATA EXPLORER
elif pg == "Data Explorer":
    st.header("Data Explorer")
    
    if 'processed_data' in curr:
        df = curr['processed_data']
        
        tab1, tab2, tab3 = st.tabs(["Distributions", "Correlations", "Feature Importance"])
        
        with tab1:
            st.subheader("Metric Distributions")
            
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            valid_cols = [c for c in num_cols if df[c].notna().sum() > 0]
            
            if valid_cols:
                c1, c2 = st.columns([2, 1])
                
                with c1:
                    sel = st.selectbox("Select metric:", valid_cols, format_func=lambda x: x.replace('_', ' ').title())
                
                with c2:
                    n_bins = st.slider("Bins:", 10, 50, 30)
                
                fig = px.histogram(df, x=sel, nbins=n_bins, title=f"Distribution: {sel.replace('_', ' ').title()}", color_discrete_sequence=['#667eea'])
                fig.update_layout(height=450, template='plotly_white', showlegend=False)
                st.plotly_chart(fig, width='stretch')
                
                # Stats
                c1, c2, c3, c4, c5 = st.columns(5)
                vdata = df[sel].dropna()
                
                c1.metric("Mean", f"{vdata.mean():.2f}")
                c2.metric("Median", f"{vdata.median():.2f}")
                c3.metric("SD", f"{vdata.std():.2f}")
                c4.metric("Min", f"{vdata.min():.2f}")
                c5.metric("Max", f"{vdata.max():.2f}")
            else:
                st.warning("No valid numeric metrics")
        
        with tab2:
            st.subheader("Correlation Analysis")
            
            if len(num_cols) > 1:
                key_feats = num_cols[:15] if len(num_cols) > 15 else num_cols
                corr_mat = df[key_feats].corr()
                
                fig = px.imshow(
                    corr_mat,
                    labels=dict(color="Correlation"),
                    x=[m.replace('_', ' ').title() for m in corr_mat.columns],
                    y=[m.replace('_', ' ').title() for m in corr_mat.columns],
                    color_continuous_scale='RdBu_r',
                    zmin=-1, zmax=1, aspect='auto'
                )
                fig.update_layout(height=600, title="Feature Correlation Heatmap")
                st.plotly_chart(fig, width='stretch')
        
        with tab3:
            st.subheader("Feature Importance")
            
            if 'feature_importance' in curr:
                feat_imp = curr['feature_importance'].head(15)
                
                fig = px.bar(feat_imp, x='importance', y='feature', orientation='h', title="Top 15 Features", color='importance', color_continuous_scale='Viridis')
                fig.update_layout(height=500, showlegend=False, template='plotly_white')
                fig.update_yaxes(title='')
                st.plotly_chart(fig, width='stretch')
                
                st.dataframe(feat_imp, hide_index=True, width='stretch')

# PAGE 4: ML MODEL RESULTS
elif pg == "ML Model Results":
    st.header("ML Model Results")
    
    c1, c2, c3, c4 = st.columns(4)
    
    c1.metric("R²", f"{r2:.3f}")
    c2.metric("MAE", f"{mae:.2f}")
    c3.metric("RMSE", f"{mets.get('test_rmse', 0):.2f}")
    c4.metric("CV R²", f"{mets.get('cv_mean', 0):.3f}")
    
    st.markdown("---")
    
    # Interpret quality
    if r2 > 0.3:
        qual = "Excellent"
        box_cls = "success-box"
        interp = "Strong predictive relationship. Captures key translatable traits."
    elif r2 > 0.15:
        qual = "Good"
        box_cls = "tracking-box"
        interp = "Solid predictive power. Identifies meaningful patterns."
    else:
        qual = "Moderate"
        box_cls = "insight-box"
        interp = "Moderate power. Other factors beyond tracking contribute."
    
    st.markdown(f"""
    <div class="{box_cls}">
    <h3 style="color: white; margin-top: 0;">Performance: {qual}</h3>
    <p style="font-size: 1.1rem;">
    R² of {r2:.3f} = model explains <strong>{r2*100:.1f}%</strong> of variance.
    </p>
    <p><strong>{interp}</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Actual vs predicted scatter
    if 'predictions' in curr:
        st.subheader("Actual vs Predicted")
        
        pred = curr['predictions']
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=pred['y_test'],
            y=pred['y_pred_test'],
            mode='markers',
            name='Test Set',
            marker=dict(size=8, color='#667eea', opacity=0.6)
        ))
        
        # Perfect prediction line
        min_val = min(pred['y_test'].min(), pred['y_pred_test'].min())
        max_val = max(pred['y_test'].max(), pred['y_pred_test'].max())
        
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash', width=2)
        ))
        
        fig.update_layout(
            title=f'Actual vs Predicted (R² = {r2:.3f})',
            xaxis_title='Actual',
            yaxis_title='Predicted',
            height=500,
            template='plotly_white',
            showlegend=True
        )
        
        st.plotly_chart(fig, width='stretch')

# PAGE 5: PLAYER INSIGHTS
elif pg == "Player Insights":
    st.header("Player Insights")
    
    if 'processed_data' in curr:
        df = curr['processed_data']
        
        if 'player_name' in df.columns:
            c1, c2 = st.columns([3, 1])
            
            with c1:
                valid_players = [str(p) for p in df['player_name'].dropna().unique() if pd.notna(p)]
                
                if len(valid_players) > 0:
                    plyr = st.selectbox("Select Player:", sorted(valid_players))
                else:
                    st.warning("No valid player names")
                    st.stop()
            
            with c2:
                st.markdown("")
                st.markdown("")
                if st.button("Random Player"):
                    plyr = np.random.choice(valid_players)
            
            # Get player data
            plyr_data = df[df['player_name'] == plyr].iloc[0]
            st.markdown(f"## {plyr}")
            st.markdown("---")
            
            # Display key metrics
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Key metrics based on analysis type
            if analysis_sel == 'NFL Rookie Performance':
                key_mets = [c for c in ['targets_per_game', 'receptions', 'rec_yards', 'rec_tds', 'yards_per_rec', 'avg_separation', 'max_speed'] if c in num_cols][:6]
            else:
                key_mets = [c for c in ['targets', 'receptions', 'avg_separation', 'max_speed', 'draft_pick', 'draft_round'] if c in num_cols][:6]
            
            if not key_mets:
                key_mets = num_cols[:6]
            
            cols = st.columns(len(key_mets))
            for idx, met in enumerate(key_mets):
                if met in plyr_data.index:
                    val = plyr_data[met]
                    cols[idx].metric(
                        met.replace('_', ' ').title(),
                        f"{val:.2f}" if isinstance(val, (int, float)) else val
                    )
            
            st.markdown("---")
            
            # Percentile rankings
            st.subheader("Percentile Rankings")
            
            pcts = []
            lbls = []
            
            for met in key_mets[:6]:
                if met in df.columns and pd.notna(plyr_data.get(met)):
                    pct = (df[met] < plyr_data[met]).sum() / len(df[met].dropna()) * 100
                    pcts.append(pct)
                    
                    lbl = met.replace('_', ' ').title()
                    if analysis_sel == 'NFL Rookie Performance':
                        if 'per_game' in met:
                            lbl = lbl.replace('Per Game', '/G')
                        elif met == 'rec_yards':
                            lbl = 'Rec Yards'
                        elif met == 'rec_tds':
                            lbl = 'Rec TDs'
                    lbls.append(lbl)
            
            if pcts:
                fig = go.Figure()
                
                fig.add_trace(go.Scatterpolar(
                    r=pcts,
                    theta=lbls,
                    fill='toself',
                    fillcolor='rgba(102, 126, 234, 0.5)',
                    line=dict(color='rgb(102, 126, 234)', width=2)
                ))
                
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100], ticksuffix='%')),
                    showlegend=False,
                    height=400,
                    title="Player Rankings vs Dataset"
                )
                
                st.plotly_chart(fig, width='stretch')
            else:
                st.info("Insufficient metrics for radar chart")
            
            # Player insights based on analysis type
            st.markdown("---")
            st.subheader("Player Profile Insights")
            
            avail_cols = set(plyr_data.index)
            
            if analysis_sel == 'NFL Rookie Performance':
                st.markdown("""
                **NFL Production Analysis:**
                
                This player's college tracking metrics suggest:
                """)
                
                insights_found = False
                
                # Targets per game
                tpg_cols = ['targets_per_game', 'targets', 'target_share']
                tpg_col = next((col for col in tpg_cols if col in avail_cols and pd.notna(plyr_data.get(col))), None)
                
                if tpg_col:
                    insights_found = True
                    tpg = plyr_data[tpg_col]
                    tpg_pct = (df[tpg_col] < tpg).sum() / len(df[tpg_col].dropna()) * 100
                    
                    if tpg_pct > 75:
                        st.markdown(f"**High-Volume Receiver**: {tpg:.1f} {tpg_col.replace('_', ' ')} ({tpg_pct:.0f}th percentile) - Projects as featured target in NFL")
                    elif tpg_pct > 50:
                        st.markdown(f"**Solid Target Share**: {tpg:.1f} {tpg_col.replace('_', ' ')} ({tpg_pct:.0f}th percentile) - Projects as reliable WR2/WR3")
                    else:
                        st.markdown(f"**Limited Volume**: {tpg:.1f} {tpg_col.replace('_', ' ')} ({tpg_pct:.0f}th percentile) - May need development or better opportunity")
                
                # Separation
                sep_cols = ['avg_separation', 'average_separation_99', 'sep_consistency', 'average_separation']
                sep_col = next((col for col in sep_cols if col in avail_cols and pd.notna(plyr_data.get(col))), None)
                
                if sep_col:
                    insights_found = True
                    sep = plyr_data[sep_col]
                    sep_pct = (df[sep_col] < sep).sum() / len(df[sep_col].dropna()) * 100
                    
                    if sep_pct > 70:
                        st.markdown(f"**Elite Separator**: {sep:.2f} yards avg separation ({sep_pct:.0f}th percentile) - Can win at NFL catch point")
                    elif sep_pct > 40:
                        st.markdown(f"**Adequate Separation**: {sep:.2f} yards ({sep_pct:.0f}th percentile) - NFL-caliber route runner")
                    else:
                        st.markdown(f"**Separation Concern**: {sep:.2f} yards ({sep_pct:.0f}th percentile) - May struggle vs tight NFL coverage")
                
                # Speed
                speed_cols = ['max_speed', 'max_speed_99', 'spd_score']
                speed_col = next((col for col in speed_cols if col in avail_cols and pd.notna(plyr_data.get(col))), None)
                
                if speed_col:
                    insights_found = True
                    speed = plyr_data[speed_col]
                    speed_pct = (df[speed_col] < speed).sum() / len(df[speed_col].dropna()) * 100
                    
                    if speed_pct > 80:
                        st.markdown(f"**Elite Speed**: {speed:.1f} ({speed_col.replace('_', ' ')}, {speed_pct:.0f}th percentile) - Elite deep threat potential")
                    elif speed_pct > 50:
                        st.markdown(f"**Above Average Speed**: {speed:.1f} ({speed_col.replace('_', ' ')}, {speed_pct:.0f}th percentile)")
                
                # YAC ability
                yac_cols = ['YACOE_MEAN', 'yac_ability', 'yards_after_catch']
                yac_col = next((col for col in yac_cols if col in avail_cols and pd.notna(plyr_data.get(col))), None)
                
                if yac_col:
                    insights_found = True
                    yac = plyr_data[yac_col]
                    yac_pct = (df[yac_col] < yac).sum() / len(df[yac_col].dropna()) * 100
                    
                    if yac_pct > 70:
                        st.markdown(f"**Elite Playmaker**: {yac:.2f} YAC over expected ({yac_pct:.0f}th percentile) - Creates after catch")
                    elif yac > 0:
                        st.markdown(f"**Positive YAC**: {yac:.2f} over expected ({yac_pct:.0f}th percentile)")
                
                if not insights_found:
                    st.info("Unable to generate detailed insights - key metrics not available for this player")
            
            elif analysis_sel == 'Draft Prediction':
                st.markdown("""
                **Draft Profile Analysis:**
                
                Based on college tracking metrics:
                """)
                
                insights_found = False
                
                # Draft position
                if 'draft_pick' in avail_cols and pd.notna(plyr_data.get('draft_pick')):
                    insights_found = True
                    pick = plyr_data['draft_pick']
                    round_num = plyr_data.get('draft_round', 'Unknown')
                    st.markdown(f"**Actual Draft Position**: Round {round_num}, Pick {int(pick)}")
                
                # Separation
                sep_cols = ['avg_separation', 'average_separation_99', 'sep_consistency']
                sep_col = next((col for col in sep_cols if col in avail_cols and pd.notna(plyr_data.get(col))), None)
                
                if sep_col:
                    insights_found = True
                    sep = plyr_data[sep_col]
                    sep_pct = (df[sep_col] < sep).sum() / len(df[sep_col].dropna()) * 100
                    
                    if sep_pct > 75:
                        st.markdown(f"**Top Separator**: {sep:.2f} yards ({sep_pct:.0f}th percentile) - Trait scouts prioritize heavily")
                    elif sep_pct > 50:
                        st.markdown(f"**Good Separator**: {sep:.2f} yards ({sep_pct:.0f}th percentile)")
                
                # Speed
                speed_cols = ['max_speed', 'max_speed_99', 'spd_score']
                speed_col = next((col for col in speed_cols if col in avail_cols and pd.notna(plyr_data.get(col))), None)
                
                if speed_col:
                    insights_found = True
                    speed = plyr_data[speed_col]
                    speed_pct = (df[speed_col] < speed).sum() / len(df[speed_col].dropna()) * 100
                    
                    if speed_pct > 80:
                        st.markdown(f"**Elite Game Speed**: {speed:.1f} ({speed_col.replace('_', ' ')}, {speed_pct:.0f}th percentile) - Elite deep threat potential")
                    elif speed_pct > 60:
                        st.markdown(f"**Above Average Speed**: {speed:.1f} ({speed_pct:.0f}th percentile)")
                
                # Production score
                if 'production_score' in avail_cols and pd.notna(plyr_data.get('production_score')):
                    insights_found = True
                    prod = plyr_data['production_score']
                    prod_pct = (df['production_score'] < prod).sum() / len(df['production_score'].dropna()) * 100
                    
                    if prod_pct > 75:
                        st.markdown(f"**High Producer**: {prod:.1f} production score ({prod_pct:.0f}th percentile)")
                
                if not insights_found:
                    st.info("Unable to generate detailed insights - key metrics not available for this player")

# Footer
st.markdown("---")
st.markdown("---")

c1, c2, c3 = st.columns([1, 2, 1])

with c2:
    st.markdown(f"""
    <div style='text-align: center; padding: 2rem 0;'>
        <h4>Tracking Prophet</h4>
        <p style='color: #888;'>
            NFL Wide Receiver Performance Prediction via In-Game Tracking Data<br>
            Predicting {analysis_sel if 'analysis_sel' in locals() else 'NFL Success'} from College Metrics
        </p>
        <p style='color: #888; margin-top: 1rem;'>
            <strong>Analysis Types Available:</strong><br>
            {', '.join(avail) if avail else 'None'}
        </p>
        <p style='margin-top: 1rem;'>
            <a href='https://github.com/hemaniprisha' target='_blank' style='margin: 0 10px; color: #667eea; text-decoration: none;'>GitHub</a> | 
            <a href='https://www.linkedin.com/in/prisha-hemani-4194a8257/' target='_blank' style='margin: 0 10px; color: #667eea; text-decoration: none;'>LinkedIn</a> | 
            <a href='mailto:hemaniprisha1@gmail.com' style='margin: 0 10px; color: #667eea; text-decoration: none;'>Contact</a>
        </p>
    </div>
    """, unsafe_allow_html=True)
