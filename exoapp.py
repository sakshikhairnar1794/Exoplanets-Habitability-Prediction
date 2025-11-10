import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, silhouette_score
from sklearn.decomposition import PCA
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Exoplanet Habitability Prediction",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with advanced styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600;700&family=Space+Grotesk:wght@300;400;600;700&display=swap');
    
    /* Global Styling */
    .main {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: #ffffff;
    }
    
    /* Header Styling */
    .main-header {
        font-family: 'Orbitron', sans-serif;
        font-size: 4rem;
        font-weight: 900;
        background: linear-gradient(45deg, #00d2ff 0%, #3a7bd5 50%, #ff00e5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 0 0 30px rgba(0, 210, 255, 0.5);
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { filter: drop-shadow(0 0 10px #00d2ff); }
        to { filter: drop-shadow(0 0 20px #ff00e5); }
    }
    
    .sub-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.3rem;
        text-align: center;
        color: #b8b8ff;
        margin-bottom: 2rem;
        letter-spacing: 2px;
    }
    
    .sub-header {
        font-family: 'Rajdhani', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #00d2ff;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #3a7bd5;
        padding-bottom: 0.5rem;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0,0,0,0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(102, 126, 234, 0.4);
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(255, 255, 255, 0.05);
        padding: 10px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        border-radius: 10px;
        padding: 10px 20px;
        color: #ffffff;
        font-family: 'Rajdhani', sans-serif;
        font-weight: 600;
        font-size: 1.1rem;
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(135deg, #2a5298 0%, #1e3c72 100%);
        border: 2px solid #00d2ff;
        transform: translateY(-2px);
    }
    
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 100%);
        border: 2px solid #ffffff;
        box-shadow: 0 4px 12px rgba(0, 210, 255, 0.5);
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    [data-testid="stSidebar"] .css-1d391kg {
        color: white;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 30px;
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(102, 126, 234, 0.6);
    }
    
    /* Insight Box */
    .insight-box {
        background: linear-gradient(135deg, rgba(0, 210, 255, 0.1) 0%, rgba(58, 123, 213, 0.1) 100%);
        padding: 1.5rem;
        border-left: 4px solid #00d2ff;
        border-radius: 10px;
        margin: 1rem 0;
        font-family: 'Space Grotesk', sans-serif;
        box-shadow: 0 4px 12px rgba(0, 210, 255, 0.2);
    }
    
    /* Success/Warning/Error Boxes */
    .stSuccess, .stWarning, .stError, .stInfo {
        font-family: 'Space Grotesk', sans-serif;
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Dataframe Styling */
    .dataframe {
        font-family: 'Space Grotesk', sans-serif;
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Metric Styling */
    [data-testid="stMetricValue"] {
        font-family: 'Orbitron', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(45deg, #00d2ff 0%, #3a7bd5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    [data-testid="stMetricLabel"] {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.1rem;
        color: #b8b8ff;
    }
    
    /* Download Button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        border-radius: 10px;
        font-family: 'Rajdhani', sans-serif;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #38ef7d 0%, #11998e 100%);
        transform: scale(1.05);
    }
    
    /* Select Box Styling */
    .stSelectbox label {
        font-family: 'Rajdhani', sans-serif;
        color: #00d2ff;
        font-weight: 600;
    }
    
    /* Slider Styling */
    .stSlider label {
        font-family: 'Rajdhani', sans-serif;
        color: #00d2ff;
        font-weight: 600;
    }
    
    /* Section Divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #00d2ff, transparent);
        margin: 2rem 0;
    }
    
    /* Card Container */
    .card-container {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 20px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    /* Animated Background */
    .main::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: radial-gradient(circle at 20% 50%, rgba(0, 210, 255, 0.1) 0%, transparent 50%),
                    radial-gradient(circle at 80% 80%, rgba(255, 0, 229, 0.1) 0%, transparent 50%);
        pointer-events: none;
        z-index: -1;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1a2e;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Radio Button Styling */
    .stRadio label {
        font-family: 'Rajdhani', sans-serif;
        color: #b8b8ff;
        font-size: 1.1rem;
    }
    
    /* Text Input */
    .stTextInput label {
        font-family: 'Rajdhani', sans-serif;
        color: #00d2ff;
        font-weight: 600;
    }
    
    /* Number Input */
    .stNumberInput label {
        font-family: 'Rajdhani', sans-serif;
        color: #00d2ff;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Generate synthetic exoplanet dataset
@st.cache_data
def generate_exoplanet_data():
    np.random.seed(42)
    n_samples = 500
    
    data = {
        'planet_name': [f'Planet_{i}' for i in range(n_samples)],
        'mass': np.random.exponential(5, n_samples),
        'radius': np.random.exponential(2, n_samples),
        'orbital_period': np.random.exponential(100, n_samples),
        'distance_from_star': np.random.exponential(1.5, n_samples),
        'stellar_temperature': np.random.normal(5500, 1000, n_samples),
        'equilibrium_temperature': np.random.normal(300, 150, n_samples),
        'discovery_year': np.random.randint(1995, 2024, n_samples),
        'detection_method': np.random.choice(['Transit', 'Radial Velocity', 'Direct Imaging', 'Microlensing'], n_samples),
        'star_system': np.random.choice([f'System_{i}' for i in range(50)], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Add some missing values
    missing_indices = np.random.choice(df.index, size=int(0.1 * n_samples), replace=False)
    df.loc[missing_indices[:len(missing_indices)//2], 'mass'] = np.nan
    df.loc[missing_indices[len(missing_indices)//2:], 'equilibrium_temperature'] = np.nan
    
    # Add duplicates
    duplicates = df.sample(10)
    df = pd.concat([df, duplicates], ignore_index=True)
    
    # Create habitability label based on conditions
    df['habitable'] = (
        (df['equilibrium_temperature'].fillna(0) > 200) & 
        (df['equilibrium_temperature'].fillna(0) < 400) &
        (df['mass'].fillna(0) > 0.5) & 
        (df['mass'].fillna(0) < 10) &
        (df['radius'].fillna(0) > 0.5) & 
        (df['radius'].fillna(0) < 2.5)
    ).astype(int)
    
    return df

# Advanced preprocessing with visualizations
def preprocess_data(df):
    st.subheader("🔧 Advanced Data Preprocessing")
    
    # Original stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Original Records", len(df), help="Total number of records before preprocessing")
    with col2:
        st.metric("Duplicate Records", df.duplicated().sum(), delta=f"-{df.duplicated().sum()}", delta_color="inverse")
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum(), delta=f"-{df.isnull().sum().sum()}", delta_color="inverse")
    with col4:
        st.metric("Features", len(df.columns), help="Total number of features")
    
    st.write("---")
    
    # Visualize missing data
    tab1, tab2, tab3, tab4 = st.tabs(["Missing Data Analysis", "Outlier Detection", "Distribution Analysis", "Correlation Heatmap"])
    
    with tab1:
        col_vis1, col_vis2 = st.columns(2)
        
        with col_vis1:
            # Missing data heatmap
            missing_data = df.isnull().sum()
            missing_data = missing_data[missing_data > 0]
            
            if len(missing_data) > 0:
                fig = go.Figure(data=[go.Bar(
                    x=missing_data.index,
                    y=missing_data.values,
                    marker_color='indianred'
                )])
                fig.update_layout(title='Missing Values by Column', xaxis_title='Column', yaxis_title='Count')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("No missing values found!")
        
        with col_vis2:
            # Missing data pattern
            fig = go.Figure(data=go.Heatmap(
                z=df.isnull().astype(int),
                colorscale=[[0, 'lightblue'], [1, 'darkred']],
                showscale=False
            ))
            fig.update_layout(title='Missing Data Pattern (Red = Missing)', height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Outlier detection using IQR
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_counts = {}
        
        for col in numeric_cols:
            if col != 'discovery_year' and col != 'habitable':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
                outlier_counts[col] = outliers
        
        col_out1, col_out2 = st.columns(2)
        
        with col_out1:
            outlier_df = pd.DataFrame(list(outlier_counts.items()), columns=['Feature', 'Outliers'])
            fig = px.bar(outlier_df, x='Feature', y='Outliers', title='Outlier Count by Feature')
            st.plotly_chart(fig, use_container_width=True)
        
        with col_out2:
            selected_feature = st.selectbox("Select feature for box plot:", list(outlier_counts.keys()))
            fig = go.Figure()
            fig.add_trace(go.Box(y=df[selected_feature].dropna(), name=selected_feature))
            fig.update_layout(title=f'Box Plot: {selected_feature}')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Distribution analysis
        col_dist1, col_dist2 = st.columns(2)
        
        with col_dist1:
            feature_for_dist = st.selectbox("Select feature for distribution:", numeric_cols)
            
            fig = make_subplots(rows=1, cols=2, subplot_titles=('Histogram', 'Q-Q Plot'))
            
            # Histogram
            fig.add_trace(
                go.Histogram(x=df[feature_for_dist].dropna(), name='Distribution', nbinsx=30),
                row=1, col=1
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col_dist2:
            # Statistical tests
            st.write("**Statistical Summary**")
            stats_data = df[feature_for_dist].dropna()
            
            stat_metrics = {
                'Mean': stats_data.mean(),
                'Median': stats_data.median(),
                'Std Dev': stats_data.std(),
                'Skewness': stats_data.skew(),
                'Kurtosis': stats_data.kurtosis()
            }
            
            for metric, value in stat_metrics.items():
                st.metric(metric, f"{value:.2f}")
    
    with tab4:
        # Correlation heatmap
        corr_features = ['mass', 'radius', 'orbital_period', 'distance_from_star', 
                        'stellar_temperature', 'equilibrium_temperature', 'habitable']
        corr_matrix = df[corr_features].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=corr_features,
            y=corr_features,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.round(2),
            texttemplate='%{text}',
            textfont={"size": 10}
        ))
        fig.update_layout(title='Feature Correlation Matrix', height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Key insights
        st.markdown("""
        <div class="insight-box">
        <strong>🔍 Key Insights:</strong><br>
        • Strong correlations indicate feature redundancy<br>
        • Weak correlations with target suggest low predictive power<br>
        • Look for correlation > 0.7 or < -0.7 for feature engineering
        </div>
        """, unsafe_allow_html=True)
    
    # Perform preprocessing
    st.write("---")
    st.subheader("Applying Preprocessing Steps...")
    
    df_clean = df.drop_duplicates()
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        df_clean[col].fillna(df_clean[col].median(), inplace=True)
    
    scaler = StandardScaler()
    numeric_features = ['mass', 'radius', 'orbital_period', 'distance_from_star', 
                       'stellar_temperature', 'equilibrium_temperature']
    df_clean[numeric_features] = scaler.fit_transform(df_clean[numeric_features])
    
    col_final1, col_final2, col_final3 = st.columns(3)
    with col_final1:
        st.metric("Clean Records", len(df_clean))
    with col_final2:
        st.metric("Removed Duplicates", len(df) - len(df_clean))
    with col_final3:
        st.metric("Features Normalized", len(numeric_features))
    
    st.success("✅ Preprocessing completed successfully!")
    
    return df_clean, scaler

# Enhanced OLAP Operations
def olap_operations(df):
    st.subheader("📊 Advanced OLAP Operations")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Slicing", "Dicing", "Drill-Down", "Pivot Analysis"])
    
    with tab1:
        st.write("**Slicing: Multi-dimensional Filtering**")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            slice_dimension = st.selectbox("Slice by:", ['star_system', 'detection_method', 'discovery_year'])
            
            if slice_dimension == 'discovery_year':
                unique_vals = sorted(df[slice_dimension].unique())
                selected_value = st.select_slider("Select value:", options=unique_vals)
            else:
                selected_value = st.selectbox("Select value:", df[slice_dimension].unique()[:15])
            
            sliced_data = df[df[slice_dimension] == selected_value]
            
            st.metric("Filtered Records", len(sliced_data))
            st.metric("Habitable %", f"{(sliced_data['habitable'].sum()/len(sliced_data)*100):.1f}%")
        
        with col2:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Mass vs Temperature', 'Radius Distribution'),
                specs=[[{'type': 'scatter'}, {'type': 'histogram'}]]
            )
            
            fig.add_trace(
                go.Scatter(x=sliced_data['mass'], y=sliced_data['equilibrium_temperature'],
                          mode='markers', marker=dict(color=sliced_data['habitable'], 
                          colorscale='Viridis', size=8)),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Histogram(x=sliced_data['radius'], nbinsx=20, marker_color='lightblue'),
                row=1, col=2
            )
            
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.write("**Dicing: Interactive Multi-dimensional Cube**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            temp_range = st.slider("Temperature (K)", 
                                  int(df['equilibrium_temperature'].min()), 
                                  int(df['equilibrium_temperature'].max()),
                                  (200, 400))
        with col2:
            mass_range = st.slider("Mass (Earth masses)",
                                  float(df['mass'].min()),
                                  float(df['mass'].max()),
                                  (0.5, 10.0))
        with col3:
            radius_range = st.slider("Radius (Earth radii)",
                                    float(df['radius'].min()),
                                    float(df['radius'].max()),
                                    (0.5, 2.5))
        
        diced_data = df[
            (df['equilibrium_temperature'] >= temp_range[0]) &
            (df['equilibrium_temperature'] <= temp_range[1]) &
            (df['mass'] >= mass_range[0]) &
            (df['mass'] <= mass_range[1]) &
            (df['radius'] >= radius_range[0]) &
            (df['radius'] <= radius_range[1])
        ]
        
        col_dice1, col_dice2 = st.columns([1, 2])
        
        with col_dice1:
            st.metric("Matching Planets", len(diced_data))
            st.metric("Habitable Count", diced_data['habitable'].sum())
            st.metric("Habitability Rate", f"{(diced_data['habitable'].sum()/len(diced_data)*100 if len(diced_data) > 0 else 0):.1f}%")
        
        with col_dice2:
            fig = px.scatter_3d(diced_data, x='mass', y='radius', z='equilibrium_temperature',
                               color='habitable', title="3D Data Cube Visualization",
                               color_continuous_scale='Viridis')
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.write("**Drill-Down & Roll-Up Analysis**")
        
        # System level aggregation
        system_summary = df.groupby('star_system').agg({
            'planet_name': 'count',
            'habitable': 'sum',
            'mass': 'mean',
            'equilibrium_temperature': 'mean'
        }).rename(columns={'planet_name': 'total_planets', 'habitable': 'habitable_planets'})
        system_summary['habitability_rate'] = (system_summary['habitable_planets'] / 
                                               system_summary['total_planets'] * 100).round(2)
        
        col_drill1, col_drill2 = st.columns([1, 1])
        
        with col_drill1:
            st.write("**System-Level Summary (Roll-Up)**")
            top_systems = system_summary.sort_values('habitability_rate', ascending=False).head(10)
            st.dataframe(top_systems, use_container_width=True)
        
        with col_drill2:
            fig = px.treemap(
                system_summary.reset_index().head(20),
                path=['star_system'],
                values='total_planets',
                color='habitability_rate',
                title='System Hierarchy (Size=Planets, Color=Habitability)',
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Drill down
        st.write("**Drill-Down to Planet Level**")
        selected_system = st.selectbox("Select system to drill down:", system_summary.index[:15])
        
        planet_detail = df[df['star_system'] == selected_system]
        
        col_detail1, col_detail2 = st.columns([1, 1])
        
        with col_detail1:
            st.dataframe(planet_detail[['planet_name', 'mass', 'radius', 'equilibrium_temperature', 'habitable']])
        
        with col_detail2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=planet_detail['mass'],
                y=planet_detail['equilibrium_temperature'],
                mode='markers+text',
                marker=dict(size=planet_detail['radius']*10, color=planet_detail['habitable'],
                           colorscale='Viridis'),
                text=planet_detail['planet_name'],
                textposition="top center"
            ))
            fig.update_layout(title=f'Planets in {selected_system}',
                            xaxis_title='Mass', yaxis_title='Temperature')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.write("**Pivot Table Analysis**")
        
        col_pivot1, col_pivot2 = st.columns(2)
        
        with col_pivot1:
            pivot_row = st.selectbox("Row dimension:", ['detection_method', 'star_system', 'discovery_year'])
            pivot_col = st.selectbox("Column dimension:", ['habitable', 'detection_method'])
        
        with col_pivot2:
            pivot_value = st.selectbox("Value to aggregate:", ['mass', 'radius', 'equilibrium_temperature'])
            pivot_agg = st.selectbox("Aggregation function:", ['mean', 'sum', 'count', 'median'])
        
        # Create pivot table
        if pivot_row == 'star_system':
            pivot_data = df.head(100)  # Limit for better visualization
        else:
            pivot_data = df
        
        pivot_table = pd.pivot_table(pivot_data, 
                                     values=pivot_value,
                                     index=pivot_row,
                                     columns=pivot_col,
                                     aggfunc=pivot_agg,
                                     fill_value=0)
        
        st.write("**Pivot Table:**")
        st.dataframe(pivot_table, use_container_width=True)
        
        # Visualize pivot
        fig = go.Figure(data=go.Heatmap(
            z=pivot_table.values,
            x=pivot_table.columns,
            y=pivot_table.index,
            colorscale='Viridis',
            text=pivot_table.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 10}
        ))
        fig.update_layout(title='Pivot Table Heatmap', height=500)
        st.plotly_chart(fig, use_container_width=True)

# Enhanced K-Means Clustering
def kmeans_clustering(df):
    st.subheader("🎯 Advanced K-Means Clustering Analysis")
    
    features = ['mass', 'radius', 'equilibrium_temperature', 'distance_from_star']
    X = df[features].fillna(df[features].median())
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        n_clusters = st.slider("Number of Clusters", 2, 8, 3)
        
        # Elbow method
        st.write("**Elbow Method Analysis**")
        inertias = []
        silhouette_scores = []
        K_range = range(2, 9)
        
        for k in K_range:
            kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans_temp.fit(X)
            inertias.append(kmeans_temp.inertia_)
            silhouette_scores.append(silhouette_score(X, kmeans_temp.labels_))
        
        fig = make_subplots(rows=2, cols=1, subplot_titles=('Elbow Curve', 'Silhouette Score'))
        
        fig.add_trace(go.Scatter(x=list(K_range), y=inertias, mode='lines+markers',
                                name='Inertia', line=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(x=list(K_range), y=silhouette_scores, mode='lines+markers',
                                name='Silhouette', line=dict(color='green')), row=2, col=1)
        
        fig.update_xaxes(title_text="Number of Clusters", row=2, col=1)
        fig.update_yaxes(title_text="Inertia", row=1, col=1)
        fig.update_yaxes(title_text="Silhouette Score", row=2, col=1)
        fig.update_layout(height=500, showlegend=False)
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X)
    
    with col2:
        # 3D scatter with clusters
        fig = px.scatter_3d(df, x='mass', y='radius', z='equilibrium_temperature',
                           color='cluster', symbol='habitable',
                           title="3D Clustering Visualization",
                           labels={'cluster': 'Cluster', 'habitable': 'Habitable'})
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # Cluster analysis
    st.write("---")
    st.write("**Detailed Cluster Analysis**")
    
    tab1, tab2, tab3 = st.tabs(["Cluster Statistics", "Cluster Profiles", "PCA Visualization"])
    
    with tab1:
        cluster_stats = df.groupby('cluster').agg({
            'planet_name': 'count',
            'mass': ['mean', 'std'],
            'radius': ['mean', 'std'],
            'equilibrium_temperature': ['mean', 'std'],
            'habitable': ['sum', 'mean']
        }).round(2)
        
        st.dataframe(cluster_stats, use_container_width=True)
    
    with tab2:
        # Radar chart for cluster profiles
        cluster_centers = kmeans.cluster_centers_
        
        fig = go.Figure()
        
        for i in range(n_clusters):
            fig.add_trace(go.Scatterpolar(
                r=cluster_centers[i],
                theta=features,
                fill='toself',
                name=f'Cluster {i}'
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            showlegend=True,
            title="Cluster Profiles (Radar Chart)"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Box plots by cluster
        selected_feature = st.selectbox("Select feature for comparison:", features)
        fig = px.box(df, x='cluster', y=selected_feature, color='habitable',
                    title=f'{selected_feature} Distribution by Cluster')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
        df_pca['cluster'] = df['cluster'].values
        df_pca['habitable'] = df['habitable'].values
        
        col_pca1, col_pca2 = st.columns(2)
        
        with col_pca1:
            fig = px.scatter(df_pca, x='PC1', y='PC2', color='cluster',
                           title='PCA: Clusters in 2D Space',
                           labels={'cluster': 'Cluster'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col_pca2:
            st.write("**PCA Explained Variance**")
            variance_explained = pca.explained_variance_ratio_
            st.metric("PC1 Variance", f"{variance_explained[0]*100:.1f}%")
            st.metric("PC2 Variance", f"{variance_explained[1]*100:.1f}%")
            st.metric("Total Variance", f"{sum(variance_explained)*100:.1f}%")
            
            # Component loadings
            loadings = pd.DataFrame(
                pca.components_.T,
                columns=['PC1', 'PC2'],
                index=features
            )
            st.write("**Feature Loadings**")
            st.dataframe(loadings.round(3))

# Association Rules Mining (keeping previous implementation)
def association_rules_mining(df):
    st.subheader("🛒 Association Rule Mining (Apriori)")
    
    df_disc = df.copy()
    
    df_disc['mass_cat'] = pd.cut(df_disc['mass'], bins=3, labels=['Low_Mass', 'Medium_Mass', 'High_Mass'])
    df_disc['temp_cat'] = pd.cut(df_disc['equilibrium_temperature'], bins=3, 
                                 labels=['Cold', 'Moderate_Temp', 'Hot'])
    df_disc['radius_cat'] = pd.cut(df_disc['radius'], bins=3, 
                                   labels=['Small_Radius', 'Medium_Radius', 'Large_Radius'])
    df_disc['habitable_cat'] = df_disc['habitable'].map({0: 'Non_Habitable', 1: 'Habitable'})
    
    transactions = df_disc[['mass_cat', 'temp_cat', 'radius_cat', 'habitable_cat']].values.tolist()
    transactions = [[str(item) for item in transaction if pd.notna(item)] for transaction in transactions]
    
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        min_support = st.slider("Minimum Support", 0.01, 0.3, 0.1)
        min_confidence = st.slider("Minimum Confidence", 0.1, 0.9, 0.5)
        min_lift = st.slider("Minimum Lift", 1.0, 3.0, 1.2)
    
    frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
    
    if len(frequent_itemsets) > 0:
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        rules = rules[rules['lift'] >= min_lift]
        
        with col2:
            st.write(f"**Found {len(rules)} Association Rules**")
            
            if len(rules) > 0:
                rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
                rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
                
                display_rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].round(3)
                st.dataframe(display_rules.sort_values('confidence', ascending=False).head(15))
                
                # Enhanced visualization
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('Support vs Confidence', 'Lift Distribution')
                )
                
                fig.add_trace(
                    go.Scatter(x=rules['support'], y=rules['confidence'], 
                              mode='markers', marker=dict(size=rules['lift']*5, 
                              color=rules['lift'], colorscale='Viridis'),
                              text=rules['antecedents'] + ' → ' + rules['consequents'],
                              hovertemplate='<b>%{text}</b><br>Support: %{x}<br>Confidence: %{y}'),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Histogram(x=rules['lift'], nbinsx=20, marker_color='lightcoral'),
                    row=1, col=2
                )
                
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No rules found with current thresholds.")
    else:
        st.warning("No frequent itemsets found. Try lowering the minimum support.")

# Enhanced KNN Classification
def knn_classification(df):
    st.subheader("🩺 K-Nearest Neighbors Classification")
    
    features = ['mass', 'radius', 'orbital_period', 'distance_from_star', 
                'stellar_temperature', 'equilibrium_temperature']
    
    X = df[features].fillna(df[features].median())
    y = df['habitable']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        k_value = st.slider("Number of Neighbors (K)", 1, 20, 5)
        
        knn = KNeighborsClassifier(n_neighbors=k_value)
        knn.fit(X_train, y_train)
        
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        st.metric("Accuracy", f"{accuracy*100:.2f}%")
        
        # Cross-validation
        cv_scores = cross_val_score(knn, X, y, cv=5)
        st.metric("Cross-Val Score", f"{cv_scores.mean()*100:.2f}%")
        st.metric("CV Std Dev", f"{cv_scores.std()*100:.2f}%")
        
        st.write("**Classification Report**")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose().round(2))
    
    with col2:
        # Create subplots for multiple visualizations
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Confusion Matrix', 'K vs Accuracy', 
                          'Precision-Recall-F1', 'ROC-style Curve'),
            specs=[[{'type': 'heatmap'}, {'type': 'scatter'}],
                   [{'type': 'bar'}, {'type': 'scatter'}]]
        )
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig.add_trace(
            go.Heatmap(z=cm, x=['Non-Hab', 'Hab'], y=['Non-Hab', 'Hab'],
                      colorscale='Blues', text=cm, texttemplate='%{text}',
                      showscale=False),
            row=1, col=1
        )
        
        # K vs Accuracy
        k_range = range(1, 21)
        accuracies = []
        for k in k_range:
            knn_temp = KNeighborsClassifier(n_neighbors=k)
            knn_temp.fit(X_train, y_train)
            accuracies.append(knn_temp.score(X_test, y_test))
        
        fig.add_trace(
            go.Scatter(x=list(k_range), y=accuracies, mode='lines+markers',
                      line=dict(color='green'), name='Accuracy'),
            row=1, col=2
        )
        
        # Precision, Recall, F1
        metrics_df = pd.DataFrame(report).T.loc[['0', '1'], ['precision', 'recall', 'f1-score']]
        metrics_df.index = ['Non-Habitable', 'Habitable']
        
        for col_name in ['precision', 'recall', 'f1-score']:
            fig.add_trace(
                go.Bar(x=metrics_df.index, y=metrics_df[col_name], name=col_name),
                row=2, col=1
            )
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': [abs(X[col].corr(y)) for col in features]
        }).sort_values('importance', ascending=True)
        
        fig.add_trace(
            go.Bar(x=feature_importance['importance'], 
                  y=feature_importance['feature'],
                  orientation='h', marker_color='coral'),
            row=2, col=2
        )
        
        fig.update_layout(height=700, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    # Prediction interface
    st.write("---")
    st.write("**🔮 Predict New Exoplanet Habitability**")
    
    pred_col1, pred_col2, pred_col3 = st.columns(3)
    
    with pred_col1:
        pred_mass = st.number_input("Mass (Earth masses)", 0.1, 100.0, 5.0)
        pred_radius = st.number_input("Radius (Earth radii)", 0.1, 50.0, 2.0)
    
    with pred_col2:
        pred_period = st.number_input("Orbital Period (days)", 1.0, 1000.0, 100.0)
        pred_distance = st.number_input("Distance from Star (AU)", 0.1, 10.0, 1.0)
    
    with pred_col3:
        pred_stellar_temp = st.number_input("Stellar Temperature (K)", 2000, 10000, 5500)
        pred_eq_temp = st.number_input("Equilibrium Temperature (K)", 0, 2000, 300)
    
    if st.button("Predict Habitability", type="primary"):
        new_data = np.array([[pred_mass, pred_radius, pred_period, pred_distance, 
                            pred_stellar_temp, pred_eq_temp]])
        
        prediction = knn.predict(new_data)[0]
        probabilities = knn.predict_proba(new_data)[0]
        
        result_col1, result_col2, result_col3 = st.columns(3)
        
        with result_col1:
            if prediction == 1:
                st.success("🌍 **Potentially Habitable!**")
            else:
                st.error("🔴 **Not Habitable**")
        
        with result_col2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probabilities[1]*100,
                title={'text': "Habitability Score"},
                gauge={'axis': {'range': [None, 100]},
                      'bar': {'color': "darkgreen" if probabilities[1] > 0.5 else "darkred"},
                      'steps': [
                          {'range': [0, 50], 'color': "lightgray"},
                          {'range': [50, 100], 'color': "lightgreen"}]}
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
        
        with result_col3:
            st.write("**Confidence Scores:**")
            st.metric("Non-Habitable", f"{probabilities[0]*100:.2f}%")
            st.metric("Habitable", f"{probabilities[1]*100:.2f}%")

# Interactive WEKA-style Analysis
def weka_analysis(df):
    st.subheader("⚙️ Interactive WEKA-Style Analysis")
    
    st.write("""
    This interactive module replicates WEKA functionality directly in your browser!
    Perform classification, clustering, and visualization without leaving the app.
    """)
    
    # Prepare data
    df_weka = df.copy()
    weka_features = ['mass', 'radius', 'orbital_period', 'distance_from_star',
                    'stellar_temperature', 'equilibrium_temperature', 'habitable']
    df_weka = df_weka[weka_features].fillna(df_weka[weka_features].median())
    
    # Main tabs for WEKA operations
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Preprocess", 
        "🎯 Classify", 
        "🔮 Cluster", 
        "📊 Visualize"
    ])
    
    with tab1:
        st.write("### Data Preprocessing")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Dataset Information**")
            st.metric("Instances", len(df_weka))
            st.metric("Attributes", len(weka_features))
            st.metric("Class Distribution", f"{df_weka['habitable'].value_counts().to_dict()}")
            
            st.write("**Attribute Statistics**")
            st.dataframe(df_weka.describe().T, use_container_width=True)
        
        with col2:
            st.write("**Filter Options**")
            
            # Normalize option
            if st.checkbox("Normalize features (0-1 range)"):
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
                df_weka_filtered = df_weka.copy()
                df_weka_filtered[weka_features[:-1]] = scaler.fit_transform(df_weka[weka_features[:-1]])
                st.success("✅ Features normalized")
                st.dataframe(df_weka_filtered.head(), use_container_width=True)
            
            # Remove outliers
            if st.checkbox("Remove outliers (IQR method)"):
                Q1 = df_weka.quantile(0.25)
                Q3 = df_weka.quantile(0.75)
                IQR = Q3 - Q1
                df_filtered = df_weka[~((df_weka < (Q1 - 1.5 * IQR)) | (df_weka > (Q3 + 1.5 * IQR))).any(axis=1)]
                st.success(f"✅ Removed {len(df_weka) - len(df_filtered)} outliers")
                st.metric("Remaining instances", len(df_filtered))
            
            # Feature selection
            st.write("**Feature Selection**")
            selected_features = st.multiselect(
                "Select features for analysis:",
                weka_features[:-1],
                default=weka_features[:-1]
            )
    
    with tab2:
        st.write("### Classification Algorithms")
        
        classifier_type = st.selectbox(
            "Select Classifier:",
            ["K-Nearest Neighbors", "Decision Tree", "Naive Bayes", "Random Forest"]
        )
        
        col_class1, col_class2 = st.columns([1, 2])
        
        with col_class1:
            test_size = st.slider("Test set size (%)", 10, 50, 30) / 100
            
            if classifier_type == "K-Nearest Neighbors":
                k_val = st.slider("K (neighbors)", 1, 20, 5)
            elif classifier_type == "Decision Tree":
                max_depth = st.slider("Max depth", 1, 20, 5)
                min_samples = st.slider("Min samples split", 2, 20, 2)
            elif classifier_type == "Random Forest":
                n_estimators = st.slider("Number of trees", 10, 200, 100)
                max_depth_rf = st.slider("Max depth", 1, 20, 10)
        
        if st.button("Run Classification", type="primary"):
            X = df_weka[weka_features[:-1]]
            y = df_weka['habitable']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            # Train classifier
            if classifier_type == "K-Nearest Neighbors":
                clf = KNeighborsClassifier(n_neighbors=k_val)
            elif classifier_type == "Decision Tree":
                clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples, random_state=42)
            elif classifier_type == "Naive Bayes":
                clf = GaussianNB()
            else:  # Random Forest
                clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth_rf, random_state=42)
            
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            
            with col_class2:
                # Results
                accuracy = accuracy_score(y_test, y_pred)
                
                st.write("**Classification Results**")
                
                # Metrics in columns
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                with metric_col1:
                    st.metric("Accuracy", f"{accuracy*100:.2f}%")
                with metric_col2:
                    st.metric("Precision", f"{classification_report(y_test, y_pred, output_dict=True)['1']['precision']*100:.2f}%")
                with metric_col3:
                    st.metric("Recall", f"{classification_report(y_test, y_pred, output_dict=True)['1']['recall']*100:.2f}%")
                
                # Confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                fig = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=['Non-Habitable', 'Habitable'],
                    y=['Non-Habitable', 'Habitable'],
                    colorscale='Blues',
                    text=cm,
                    texttemplate='%{text}',
                    textfont={"size": 20}
                ))
                fig.update_layout(title='Confusion Matrix', height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed report
                st.write("**Detailed Classification Report**")
                report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T
                st.dataframe(report_df.round(3), use_container_width=True)
                
                # Feature importance (for tree-based models)
                if classifier_type in ["Decision Tree", "Random Forest"]:
                    st.write("**Feature Importance**")
                    importance_df = pd.DataFrame({
                        'Feature': weka_features[:-1],
                        'Importance': clf.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    fig = px.bar(importance_df, x='Importance', y='Feature', 
                                orientation='h', title='Feature Importance')
                    st.plotly_chart(fig, use_container_width=True)
                
                # Decision tree visualization
                if classifier_type == "Decision Tree":
                    st.write("**Decision Tree Visualization**")
                    fig, ax = plt.subplots(figsize=(20, 10))
                    plot_tree(clf, feature_names=weka_features[:-1], 
                             class_names=['Non-Habitable', 'Habitable'],
                             filled=True, ax=ax, fontsize=10)
                    st.pyplot(fig)
    
    with tab3:
        st.write("### Clustering Analysis")
        
        col_clust1, col_clust2 = st.columns([1, 2])
        
        with col_clust1:
            n_clusters_weka = st.slider("Number of clusters", 2, 8, 3)
            
            cluster_features = st.multiselect(
                "Select features for clustering:",
                weka_features[:-1],
                default=weka_features[:-1][:4]
            )
        
        if st.button("Run Clustering", type="primary"):
            X_cluster = df_weka[cluster_features]
            
            # Perform clustering
            kmeans_weka = KMeans(n_clusters=n_clusters_weka, random_state=42, n_init=10)
            clusters = kmeans_weka.fit_predict(X_cluster)
            
            df_weka['cluster'] = clusters
            
            with col_clust2:
                st.write("**Clustering Results**")
                
                # Metrics
                silhouette = silhouette_score(X_cluster, clusters)
                st.metric("Silhouette Score", f"{silhouette:.3f}")
                
                # Cluster distribution
                cluster_counts = pd.Series(clusters).value_counts().sort_index()
                fig = px.bar(x=cluster_counts.index, y=cluster_counts.values,
                           labels={'x': 'Cluster', 'y': 'Count'},
                           title='Cluster Distribution')
                st.plotly_chart(fig, use_container_width=True)
                
                # Cluster centers
                st.write("**Cluster Centers**")
                centers_df = pd.DataFrame(kmeans_weka.cluster_centers_, 
                                         columns=cluster_features)
                st.dataframe(centers_df.round(3), use_container_width=True)
            
            # Visualization
            st.write("**Cluster Visualization**")
            
            if len(cluster_features) >= 3:
                fig = px.scatter_3d(df_weka, 
                                   x=cluster_features[0], 
                                   y=cluster_features[1], 
                                   z=cluster_features[2],
                                   color='cluster',
                                   symbol='habitable',
                                   title='3D Cluster Visualization')
                st.plotly_chart(fig, use_container_width=True)
            elif len(cluster_features) >= 2:
                fig = px.scatter(df_weka, 
                               x=cluster_features[0], 
                               y=cluster_features[1],
                               color='cluster',
                               symbol='habitable',
                               title='2D Cluster Visualization')
                st.plotly_chart(fig, use_container_width=True)
            
            # Cluster statistics
            st.write("**Cluster Statistics by Habitability**")
            cluster_hab = pd.crosstab(df_weka['cluster'], df_weka['habitable'])
            st.dataframe(cluster_hab, use_container_width=True)
    
    with tab4:
        st.write("### Data Visualization")
        
        viz_type = st.selectbox(
            "Select Visualization Type:",
            ["Scatter Plot Matrix", "Parallel Coordinates", "Box Plots", 
             "Correlation Matrix", "3D Scatter"]
        )
        
        if viz_type == "Scatter Plot Matrix":
            selected_attrs = st.multiselect(
                "Select attributes (max 5):",
                weka_features,
                default=weka_features[:4]
            )
            
            if len(selected_attrs) > 1:
                fig = px.scatter_matrix(df_weka, dimensions=selected_attrs,
                                       color='habitable',
                                       title='Scatter Plot Matrix')
                fig.update_layout(height=800)
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Parallel Coordinates":
            fig = px.parallel_coordinates(df_weka, color='habitable',
                                         dimensions=weka_features,
                                         title='Parallel Coordinates Plot')
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Box Plots":
            col_box1, col_box2 = st.columns(2)
            
            for i, feature in enumerate(weka_features[:-1]):
                col = col_box1 if i % 2 == 0 else col_box2
                with col:
                    fig = px.box(df_weka, y=feature, x='habitable',
                               title=f'{feature} by Habitability')
                    st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Correlation Matrix":
            corr = df_weka.corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr,
                x=corr.columns,
                y=corr.columns,
                colorscale='RdBu',
                zmid=0,
                text=corr.round(2),
                texttemplate='%{text}',
                textfont={"size": 10}
            ))
            fig.update_layout(title='Correlation Matrix', height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        else:  # 3D Scatter
            col_3d1, col_3d2, col_3d3 = st.columns(3)
            
            with col_3d1:
                x_axis = st.selectbox("X-axis:", weka_features[:-1], index=0)
            with col_3d2:
                y_axis = st.selectbox("Y-axis:", weka_features[:-1], index=1)
            with col_3d3:
                z_axis = st.selectbox("Z-axis:", weka_features[:-1], index=2)
            
            fig = px.scatter_3d(df_weka, x=x_axis, y=y_axis, z=z_axis,
                               color='habitable',
                               title=f'3D Scatter: {x_axis} vs {y_axis} vs {z_axis}')
            fig.update_layout(height=700)
            st.plotly_chart(fig, use_container_width=True)
    
    # Export section
    st.write("---")
    st.write("### 📥 Export Data for External WEKA")
    
    download_col1, download_col2, download_col3 = st.columns(3)
    
    with download_col1:
        csv = df_weka.to_csv(index=False)
        st.download_button(
            label="📄 Download CSV",
            data=csv,
            file_name="exoplanet_data.csv",
            mime="text/csv"
        )
    
    with download_col2:
        # Generate ARFF
        arff_content = "@RELATION exoplanet_habitability\n\n"
        for col in weka_features[:-1]:
            arff_content += f"@ATTRIBUTE {col} NUMERIC\n"
        arff_content += "@ATTRIBUTE habitable {0,1}\n\n@DATA\n"
        for _, row in df_weka.iterrows():
            arff_content += ",".join([str(row[col]) for col in weka_features]) + "\n"
        
        st.download_button(
            label="📊 Download ARFF",
            data=arff_content,
            file_name="exoplanet_data.arff",
            mime="text/plain"
        )
    
    with download_col3:
        # JSON export
        json_data = df_weka.to_json(orient='records', indent=2)
        st.download_button(
            label="🔗 Download JSON",
            data=json_data,
            file_name="exoplanet_data.json",
            mime="application/json"
        )

# Main application
def main():
    st.markdown('<h1 class="main-header">🪐 EXOPLANET HABITABILITY PREDICTION</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Advanced Data Mining & Warehousing Analytics Platform</p>', unsafe_allow_html=True)
    
    # Sidebar navigation with enhanced styling
    st.sidebar.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <h1 style='font-family: Orbitron; color: #00d2ff; font-size: 1.8rem;'>🧭 NAVIGATION</h1>
        </div>
    """, unsafe_allow_html=True)
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio("Select Analysis Module", [
        "🏠 Home & Overview",
        "🔧 Preprocessing",
        "📊 OLAP Operations",
        "🎯 K-Means Clustering",
        "🛒 Association Rules",
        "🩺 KNN Classification",
        "⚙️ WEKA Analysis"
    ])
    
    # Load data
    df_raw = generate_exoplanet_data()
    
    if page == "🏠 Home & Overview":
        # Animated welcome section
        st.markdown("""
            <div class='card-container' style='text-align: center; padding: 40px; margin-bottom: 30px;'>
                <h2 style='font-family: Space Grotesk; color: #00d2ff; font-size: 2rem;'>
                    Welcome to the Exoplanet Analytics Platform
                </h2>
                <p style='font-family: Rajdhani; color: #b8b8ff; font-size: 1.2rem; margin-top: 20px;'>
                    Explore the cosmos through advanced data mining techniques and discover potentially habitable worlds
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.write("---")
        
        # Metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Planets", len(df_raw), help="Total exoplanets in dataset")
        with col2:
            st.metric("Habitable", df_raw['habitable'].sum(), 
                     delta=f"{(df_raw['habitable'].sum()/len(df_raw)*100):.1f}%")
        with col3:
            st.metric("Star Systems", df_raw['star_system'].nunique())
        with col4:
            st.metric("Features", len(df_raw.columns) - 2)
        with col5:
            st.metric("Time Span", f"{df_raw['discovery_year'].min()}-{df_raw['discovery_year'].max()}")
        
        st.write("---")
        
        # Overview tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 Dataset", "📈 Statistics", "🎨 Visualizations", "📚 Project Info"
        ])
        
        with tab1:
            st.write("**Raw Exoplanet Dataset**")
            st.dataframe(df_raw.head(50), use_container_width=True, height=400)
            
            col_info1, col_info2 = st.columns(2)
            
            with col_info1:
                st.write("**Column Information**")
                info_df = pd.DataFrame({
                    'Column': df_raw.columns,
                    'Type': df_raw.dtypes.values,
                    'Non-Null': df_raw.count().values,
                    'Null': df_raw.isnull().sum().values
                })
                st.dataframe(info_df, use_container_width=True)
            
            with col_info2:
                st.write("**Quick Summary**")
                st.write(f"- **Detection Methods**: {df_raw['detection_method'].nunique()}")
                st.write(f"- **Most Common Method**: {df_raw['detection_method'].mode()[0]}")
                st.write(f"- **Average Mass**: {df_raw['mass'].mean():.2f} Earth masses")
                st.write(f"- **Average Radius**: {df_raw['radius'].mean():.2f} Earth radii")
                st.write(f"- **Temperature Range**: {df_raw['equilibrium_temperature'].min():.0f}K - {df_raw['equilibrium_temperature'].max():.0f}K")
        
        with tab2:
            st.write("**Comprehensive Statistical Analysis**")
            st.dataframe(df_raw.describe(), use_container_width=True)
            
            col_stat1, col_stat2 = st.columns(2)
            
            with col_stat1:
                st.write("**Categorical Features**")
                cat_summary = pd.DataFrame({
                    'Detection Method': df_raw['detection_method'].value_counts()
                })
                st.dataframe(cat_summary)
            
            with col_stat2:
                st.write("**Habitability Analysis**")
                hab_stats = df_raw.groupby('habitable').agg({
                    'mass': ['mean', 'std'],
                    'radius': ['mean', 'std'],
                    'equilibrium_temperature': ['mean', 'std']
                }).round(2)
                st.dataframe(hab_stats)
        
        with tab3:
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                # Habitability distribution
                fig1 = px.pie(df_raw, names='habitable', 
                             title='Habitability Distribution',
                             color_discrete_sequence=['#FF6B6B', '#4ECDC4'])
                fig1.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig1, use_container_width=True)
                
                # Detection methods
                fig2 = px.histogram(df_raw, x='detection_method', color='habitable',
                                  title='Planets by Detection Method',
                                  barmode='group')
                st.plotly_chart(fig2, use_container_width=True)
                
                # Discovery timeline
                timeline = df_raw.groupby('discovery_year')['habitable'].agg(['count', 'sum'])
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(x=timeline.index, y=timeline['count'],
                                         mode='lines+markers', name='Total',
                                         line=dict(color='blue', width=3)))
                fig3.add_trace(go.Scatter(x=timeline.index, y=timeline['sum'],
                                         mode='lines+markers', name='Habitable',
                                         line=dict(color='green', width=3)))
                fig3.update_layout(title='Discovery Timeline', 
                                  xaxis_title='Year', yaxis_title='Count')
                st.plotly_chart(fig3, use_container_width=True)
            
            with viz_col2:
                # 3D scatter
                fig4 = px.scatter_3d(df_raw, x='mass', y='radius', 
                                    z='equilibrium_temperature',
                                    color='habitable', 
                                    title='3D Feature Space',
                                    labels={'habitable': 'Habitable'})
                st.plotly_chart(fig4, use_container_width=True)
                
                # Violin plot
                fig5 = go.Figure()
                for hab in [0, 1]:
                    fig5.add_trace(go.Violin(
                        y=df_raw[df_raw['habitable'] == hab]['equilibrium_temperature'],
                        name=f"{'Habitable' if hab else 'Non-Habitable'}",
                        box_visible=True,
                        meanline_visible=True
                    ))
                fig5.update_layout(title='Temperature Distribution by Habitability')
                st.plotly_chart(fig5, use_container_width=True)
                
                # Sunburst chart
                df_sun = df_raw.groupby(['detection_method', 'habitable']).size().reset_index(name='count')
                df_sun['habitable'] = df_sun['habitable'].map({0: 'Non-Habitable', 1: 'Habitable'})
                fig6 = px.sunburst(df_sun, path=['detection_method', 'habitable'], 
                                  values='count',
                                  title='Hierarchical View: Method > Habitability')
                st.plotly_chart(fig6, use_container_width=True)
        
        with tab4:
            st.write("## 📚 Project Documentation")
            
            st.markdown("""
            ### 🎯 Project Overview
            This comprehensive data mining project demonstrates advanced analytics on exoplanet data
            to predict habitability using multiple machine learning and data warehousing techniques.
            
            ### ✨ Key Features
            
            #### 1. **Data Preprocessing** 🔧
            - Duplicate detection and removal
            - Missing value imputation (median strategy)
            - Feature normalization using StandardScaler
            - Statistical analysis and outlier detection
            - Advanced visualizations for data quality assessment
            
            #### 2. **OLAP Operations** 📊
            - **Slicing**: Filter data by single dimension (star system, detection method, year)
            - **Dicing**: Multi-dimensional filtering with interactive controls
            - **Drill-Down/Roll-Up**: Navigate from system level to individual planets
            - **Pivot Analysis**: Dynamic cross-tabulation with custom aggregations
            
            #### 3. **K-Means Clustering** 🎯
            - Unsupervised grouping of exoplanets
            - Elbow method and Silhouette score for optimal K
            - 3D cluster visualization
            - PCA-based dimensionality reduction
            - Cluster profiling with radar charts
            
            #### 4. **Association Rule Mining** 🛒
            - Apriori algorithm implementation
            - Feature discretization for categorical rules
            - Interactive support/confidence/lift thresholds
            - Rule visualization and interpretation
            
            #### 5. **KNN Classification** 🩺
            - Supervised learning for habitability prediction
            - Cross-validation for model validation
            - Comprehensive performance metrics
            - Interactive prediction interface
            - Feature importance analysis
            
            #### 6. **WEKA-Style Analysis** ⚙️
            - **Browser-based data mining without WEKA installation**
            - Multiple classification algorithms (KNN, Decision Tree, Naive Bayes, Random Forest)
            - Interactive clustering with visualization
            - Comprehensive data visualization tools
            - Export functionality (CSV, ARFF, JSON)
            
            ### 🛠️ Technologies Used
            - **Python**: Core programming language
            - **Streamlit**: Interactive web framework
            - **Scikit-learn**: Machine learning algorithms
            - **Plotly**: Advanced interactive visualizations
            - **Pandas/NumPy**: Data manipulation
            - **MLxtend**: Association rule mining
            
            ### 📖 How to Use
            1. Navigate through sections using the sidebar
            2. Interact with sliders, dropdowns, and buttons
            3. Explore visualizations by hovering and zooming
            4. Download processed data for external analysis
            5. Make predictions using the interactive interface
            
            ### 🎓 Educational Value
            - Demonstrates end-to-end data mining pipeline
            - Covers preprocessing, exploration, modeling, and evaluation
            - Suitable for academic projects and presentations
            - Includes multiple algorithms for comparison
            
            ### 💡 Future Enhancements
            - Real exoplanet data integration (NASA Exoplanet Archive)
            - Deep learning models (Neural Networks)
            - More clustering algorithms (DBSCAN, Hierarchical)
            - Time series analysis of discoveries
            - Collaborative filtering
            """)
            
            st.info("👈 **Get started by selecting a module from the sidebar!**")
    
    elif page == "🔧 Preprocessing":
        df_clean, scaler = preprocess_data(df_raw.copy())
        st.session_state['df_clean'] = df_clean
        
    elif page == "📊 OLAP Operations":
        if 'df_clean' not in st.session_state:
            df_clean, _ = preprocess_data(df_raw.copy())
            st.session_state['df_clean'] = df_clean
        olap_operations(df_raw.copy())
        
    elif page == "🎯 K-Means Clustering":
        if 'df_clean' not in st.session_state:
            df_clean, _ = preprocess_data(df_raw.copy())
            st.session_state['df_clean'] = df_clean
        kmeans_clustering(st.session_state['df_clean'].copy())
        
    elif page == "🛒 Association Rules":
        if 'df_clean' not in st.session_state:
            df_clean, _ = preprocess_data(df_raw.copy())
            st.session_state['df_clean'] = df_clean
        association_rules_mining(df_raw.copy())
        
    elif page == "🩺 KNN Classification":
        if 'df_clean' not in st.session_state:
            df_clean, _ = preprocess_data(df_raw.copy())
            st.session_state['df_clean'] = df_clean
        knn_classification(st.session_state['df_clean'].copy())
        
    elif page == "⚙️ WEKA Analysis":
        if 'df_clean' not in st.session_state:
            df_clean, _ = preprocess_data(df_raw.copy())
            st.session_state['df_clean'] = df_clean
        weka_analysis(st.session_state['df_clean'].copy())
    
    # Enhanced sidebar
    st.sidebar.write("---")
    st.sidebar.markdown("""
        <div style='background: linear-gradient(135deg, rgba(0, 210, 255, 0.1) 0%, rgba(58, 123, 213, 0.1) 100%); 
                    padding: 15px; border-radius: 10px; border-left: 4px solid #00d2ff;'>
            <h3 style='font-family: Rajdhani; color: #00d2ff; margin-bottom: 10px;'>📋 Project Checklist</h3>
            <p style='font-family: Space Grotesk; color: #b8b8ff; line-height: 1.8;'>
                ✅ Data Preprocessing<br>
                ✅ OLAP Operations<br>
                ✅ K-Means Clustering<br>
                ✅ Association Rules<br>
                ✅ KNN Classification<br>
                ✅ WEKA Analysis
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.write("---")
    st.sidebar.markdown("""
        <div style='background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); 
                    padding: 15px; border-radius: 10px;'>
            <h3 style='font-family: Rajdhani; color: #667eea; margin-bottom: 10px;'>📊 Quick Stats</h3>
        </div>
    """, unsafe_allow_html=True)
    st.sidebar.metric("Dataset Size", f"{len(df_raw)} planets")
    st.sidebar.metric("Habitability Rate", f"{(df_raw['habitable'].sum()/len(df_raw)*100):.1f}%")
    
    st.sidebar.write("---")
    st.sidebar.markdown("""
        <div style='background: linear-gradient(135deg, rgba(17, 153, 142, 0.1) 0%, rgba(56, 239, 125, 0.1) 100%); 
                    padding: 15px; border-radius: 10px; border-left: 4px solid #11998e;'>
            <h3 style='font-family: Rajdhani; color: #11998e;'>💡 Pro Tip</h3>
            <p style='font-family: Space Grotesk; color: #b8b8ff;'>
                Use the WEKA Analysis tab for complete in-browser data mining without installing WEKA!
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Enhanced Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 30px; background: rgba(255, 255, 255, 0.05); border-radius: 15px; margin-top: 40px;'>
        <p style='font-family: Orbitron; color: #00d2ff; font-size: 1.3rem; margin-bottom: 10px;'>
            🪐 EXOPLANET HABITABILITY PREDICTION SYSTEM
        </p>
        <p style='font-family: Space Grotesk; color: #b8b8ff; font-size: 1rem;'>
            Data Mining & Warehousing Project | Built with Streamlit, Scikit-learn, and Plotly
        </p>
        <p style='font-family: Rajdhani; color: #667eea; font-size: 0.9rem; margin-top: 10px;'>
            Exploring the Universe Through Data Science ✨
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()