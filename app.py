import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import os

# Page configuration
st.set_page_config(
    page_title="Evil Twin Attack Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - FIXED WITH BETTER VISIBILITY
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .alert-danger {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: #c62828;
    }
    .alert-danger h2, .alert-danger h3, .alert-danger p {
        color: #c62828 !important;
    }
    .alert-success {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: #2e7d32;
    }
    .alert-success h2, .alert-success h3, .alert-success p {
        color: #2e7d32 !important;
    }
    .alert-warning {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: #e65100;
    }
    .alert-warning h4, .alert-warning p, .alert-warning ol, .alert-warning li {
        color: #e65100 !important;
    }
    .alert-info {
        background-color: #e3f2fd;
        border-left: 5px solid #2196f3;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: #0d47a1;
    }
    .alert-info h4, .alert-info p, .alert-info ol, .alert-info li {
        color: #0d47a1 !important;
        font-weight: 500;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #1f77b4, #2ca02c);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.75rem;
        border-radius: 8px;
        font-size: 1.1rem;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #2ca02c, #1f77b4);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    """Load all trained models and artifacts"""
    try:
        models = {}
        
        # Check directory
        model_paths = ['models/', './']
        model_dir = None
        
        for path in model_paths:
            if os.path.exists(path + 'rf_model.pkl'):
                model_dir = path
                break
        
        if model_dir is None:
            return None
        
        # Load Random Forest
        with open(model_dir + 'rf_model.pkl', 'rb') as f:
            models['rf_model'] = pickle.load(f)
        
        # Load DNN
        models['dnn_model'] = keras.models.load_model(
            model_dir + 'dnn_model.h5',
            compile=False
        )
        models['dnn_model'].compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Load Isolation Forest
        with open(model_dir + 'iso_forest_model.pkl', 'rb') as f:
            models['iso_forest'] = pickle.load(f)
        
        # Load Autoencoder
        models['autoencoder'] = keras.models.load_model(
            model_dir + 'autoencoder_model.h5',
            compile=False
        )
        models['autoencoder'].compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Load artifacts
        with open(model_dir + 'scaler.pkl', 'rb') as f:
            models['scaler'] = pickle.load(f)
        
        with open(model_dir + 'selected_features.pkl', 'rb') as f:
            models['selected_features'] = pickle.load(f)
        
        with open(model_dir + 'label_encoder.pkl', 'rb') as f:
            models['label_encoder'] = pickle.load(f)
        
        with open(model_dir + 'ae_threshold.pkl', 'rb') as f:
            models['ae_threshold'] = pickle.load(f)
        
        with open(model_dir + 'label_mapping.pkl', 'rb') as f:
            models['label_mapping'] = pickle.load(f)
        
        return models
    
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

# Initialize session state
def init_session_state():
    if 'detection_history' not in st.session_state:
        st.session_state.detection_history = []
    if 'total_packets' not in st.session_state:
        st.session_state.total_packets = 0
    if 'attacks_detected' not in st.session_state:
        st.session_state.attacks_detected = 0

# Prediction function
def predict_packet(packet_data, models):
    """Predict if packet is an Evil Twin attack"""
    try:
        selected_features = models['selected_features']
        
        # Extract features
        features = []
        for feature in selected_features:
            if feature in packet_data.index:
                features.append(packet_data[feature])
            else:
                features.append(0)
        
        X = np.array(features).reshape(1, -1)
        X = np.nan_to_num(X, nan=0.0)
        
        # Scale features
        X_scaled = models['scaler'].transform(X)
        
        # Get predictions
        rf_proba = float(models['rf_model'].predict_proba(X_scaled)[0, 1])
        dnn_proba = float(models['dnn_model'].predict(X_scaled, verbose=0)[0, 0])
        
        iso_pred = models['iso_forest'].predict(X_scaled)[0]
        iso_pred = 1 if iso_pred == -1 else 0
        
        X_reconstructed = models['autoencoder'].predict(X_scaled, verbose=0)
        mse = float(np.mean(np.power(X_scaled - X_reconstructed, 2)))
        ae_pred = 1 if mse > models['ae_threshold'] else 0
        
        # Hybrid ensemble
        hybrid_proba = 0.3 * rf_proba + 0.4 * dnn_proba + 0.15 * iso_pred + 0.15 * ae_pred
        
        prediction = 'Evil Twin Attack' if hybrid_proba > 0.5 else 'Normal'
        confidence = hybrid_proba * 100 if prediction == 'Evil Twin Attack' else (1 - hybrid_proba) * 100
        
        return {
            'prediction': prediction,
            'confidence': float(confidence),
            'rf_proba': float(rf_proba),
            'dnn_proba': float(dnn_proba),
            'iso_pred': int(iso_pred),
            'ae_pred': int(ae_pred),
            'hybrid_proba': float(hybrid_proba)
        }
    
    except Exception as e:
        return None

# Main detection function - ANALYZE ALL ROWS
def analyze_csv_file(df, models, show_realtime=True):
    """Analyze all packets in the CSV file"""
    
    total_packets = len(df)
    st.info(f"📊 Found **{total_packets:,}** packets to analyze")
    
    # Ask user confirmation if too many packets
    if total_packets > 1000:
        st.warning(f"⚠️ Large dataset detected ({total_packets:,} packets). This may take some time.")
        max_packets = st.number_input(
            "Limit analysis to (packets):",
            min_value=1,
            max_value=total_packets,
            value=min(1000, total_packets),
            help="Analyze first N packets for faster results"
        )
    else:
        max_packets = total_packets
    
    # Show real-time toggle
    col1, col2 = st.columns(2)
    with col1:
        show_realtime = st.checkbox("Show real-time progress", value=True)
    with col2:
        show_details = st.checkbox("Show detailed packet info", value=False)
    
    if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Real-time visualization placeholders
        if show_realtime:
            live_stats_placeholder = st.empty()
            live_chart_placeholder = st.empty()
        
        results = []
        start_time = time.time()
        
        # Analyze each packet
        for idx in range(min(max_packets, len(df))):
            status_text.text(f"🔍 Analyzing packet {idx+1}/{max_packets}...")
            progress_bar.progress((idx + 1) / max_packets)
            
            result = predict_packet(df.iloc[idx], models)
            
            if result:
                results.append({
                    'Packet_ID': idx + 1,
                    'Timestamp': datetime.now().strftime("%H:%M:%S"),
                    'Prediction': result['prediction'],
                    'Confidence': f"{result['confidence']:.2f}%",
                    'RF_Score': f"{result['rf_proba']*100:.1f}%",
                    'DNN_Score': f"{result['dnn_proba']*100:.1f}%",
                    'IsoForest': '🔴 Anomaly' if result['iso_pred']==1 else '🟢 Normal',
                    'Autoencoder': '🔴 Anomaly' if result['ae_pred']==1 else '🟢 Normal',
                    'Status': '🚨 THREAT' if result['prediction']=='Evil Twin Attack' else '✅ SAFE'
                })
                
                # Update session stats
                st.session_state.total_packets += 1
                if result['prediction'] == 'Evil Twin Attack':
                    st.session_state.attacks_detected += 1
                
                # Real-time visualization (update every 10 packets)
                if show_realtime and (idx + 1) % 10 == 0:
                    results_df = pd.DataFrame(results)
                    
                    # Live stats
                    with live_stats_placeholder.container():
                        col1, col2, col3, col4 = st.columns(4)
                        attacks = len(results_df[results_df['Prediction'] == 'Evil Twin Attack'])
                        normal = len(results_df) - attacks
                        
                        col1.metric("📊 Analyzed", f"{len(results_df):,}")
                        col2.metric("🚨 Attacks", attacks)
                        col3.metric("✅ Normal", normal)
                        col4.metric("📈 Attack Rate", f"{attacks/len(results_df)*100:.1f}%")
                    
                    # Live chart
                    confidences = [float(c.strip('%')) for c in results_df['Confidence']]
                    colors = ['red' if p == 'Evil Twin Attack' else 'green' 
                            for p in results_df['Prediction']]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=results_df['Packet_ID'],
                        y=confidences,
                        mode='lines+markers',
                        name='Confidence',
                        line=dict(color='blue', width=2),
                        marker=dict(size=4, color=colors)
                    ))
                    
                    fig.update_layout(
                        title="Real-Time Detection Confidence",
                        xaxis_title="Packet ID",
                        yaxis_title="Confidence (%)",
                        height=300,
                        yaxis=dict(range=[0, 100])
                    )
                    
                    live_chart_placeholder.plotly_chart(fig, use_container_width=True)
            
            # Small delay for visual effect
            if show_realtime:
                time.sleep(0.02)
        
        elapsed_time = time.time() - start_time
        status_text.text(f"✅ Analysis completed in {elapsed_time:.2f} seconds!")
        progress_bar.progress(1.0)
        
        # Display final results
        display_final_results(results, elapsed_time, show_details)

def display_final_results(results, elapsed_time, show_details=False):
    """Display comprehensive analysis results"""
    
    st.markdown("---")
    st.markdown("# 📊 Analysis Complete!")
    
    results_df = pd.DataFrame(results)
    
    # Summary metrics
    attacks = len(results_df[results_df['Prediction'] == 'Evil Twin Attack'])
    normal = len(results_df) - attacks
    
    st.markdown("### 📈 Summary Statistics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <h2>{len(results_df):,}</h2>
                <p>Total Analyzed</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #f44336 0%, #e91e63 100%);">
                <h2>{attacks:,}</h2>
                <p>Attacks Detected</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #4caf50 0%, #8bc34a 100%);">
                <h2>{normal:,}</h2>
                <p>Normal Traffic</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #ff9800 0%, #ff5722 100%);">
                <h2>{attacks/len(results_df)*100:.1f}%</h2>
                <p>Attack Rate</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #9c27b0 0%, #673ab7 100%);">
                <h2>{elapsed_time:.2f}s</h2>
                <p>Processing Time</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Detailed results table
    st.markdown("### 📋 Detailed Detection Report")
    
    if show_details:
        st.dataframe(results_df, use_container_width=True, height=400)
    else:
        # Show condensed view
        display_cols = ['Packet_ID', 'Prediction', 'Confidence', 'Status']
        st.dataframe(results_df[display_cols], use_container_width=True, height=400)
        
        with st.expander("🔍 Show All Columns"):
            st.dataframe(results_df, use_container_width=True)
    
    # Download button
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Full Report (CSV)",
        data=csv,
        file_name=f"evil_twin_detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    # Visualizations
    st.markdown("### 📊 Visual Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart
        fig = px.pie(
            results_df,
            names='Prediction',
            title="Threat Distribution",
            color='Prediction',
            color_discrete_map={'Normal': '#4caf50', 'Evil Twin Attack': '#f44336'},
            hole=0.4
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Bar chart
        prediction_counts = results_df['Prediction'].value_counts()
        fig = go.Figure(data=[
            go.Bar(
                x=prediction_counts.index,
                y=prediction_counts.values,
                marker_color=['#4caf50' if x=='Normal' else '#f44336' for x in prediction_counts.index],
                text=prediction_counts.values,
                textposition='auto'
            )
        ])
        fig.update_layout(
            title="Detection Counts",
            xaxis_title="Prediction",
            yaxis_title="Count",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Confidence distribution
    st.markdown("### 📈 Confidence Analysis")
    
    confidences = [float(c.strip('%')) for c in results_df['Confidence']]
    colors = ['red' if p == 'Evil Twin Attack' else 'green' for p in results_df['Prediction']]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=results_df['Packet_ID'],
        y=confidences,
        mode='lines+markers',
        name='Confidence',
        line=dict(color='blue', width=2),
        marker=dict(size=6, color=colors, line=dict(color='black', width=1))
    ))
    
    fig.update_layout(
        title="Detection Confidence Over All Packets",
        xaxis_title="Packet ID",
        yaxis_title="Confidence (%)",
        yaxis=dict(range=[0, 100]),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Attack packets list
    if attacks > 0:
        st.markdown("### 🚨 Detected Threats")
        attack_packets = results_df[results_df['Prediction'] == 'Evil Twin Attack']
        st.dataframe(attack_packets, use_container_width=True)

# Main app
def main():
    init_session_state()
    
    # Header
    st.markdown('<p class="main-header">🛡️ Evil Twin Attack Detection System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Real-Time Threat Analysis for Wireless Networks</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load models
    with st.spinner("🔄 Loading AI models..."):
        models = load_models()
    
    if models is None:
        st.error("⚠️ Failed to load models. Please ensure all model files are in the 'models/' directory.")
        st.stop()
        return
    
    st.success("✅ All AI models loaded successfully!")
    
    # Sidebar
    st.sidebar.title("⚙️ Control Panel")
    st.sidebar.markdown("---")
    
    st.sidebar.markdown("### 📊 System Statistics")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Total Packets", st.session_state.total_packets)
    with col2:
        st.metric("Attacks", st.session_state.attacks_detected)
    
    detection_rate = (st.session_state.attacks_detected / st.session_state.total_packets * 100) if st.session_state.total_packets > 0 else 0
    st.sidebar.metric("Detection Rate", f"{detection_rate:.1f}%")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 AI Models")
    st.sidebar.info("""
    **Hybrid Detection System**
    
    - 🌲 Random Forest
    - 🧠 Deep Neural Network
    - 🌳 Isolation Forest
    - 🔄 Autoencoder
    
    **Weights:**
    RF: 30% | DNN: 40%
    ISO: 15% | AE: 15%
    """)
    
    # Main content
    st.subheader("📤 Upload Network Traffic Data")
    
    st.markdown("""
        <div class="alert-info">
            <h4>📋 Instructions:</h4>
            <ol>
                <li>Upload a CSV file containing network packet features</li>
                <li>The system will analyze <strong>ALL packets</strong> in the file</li>
                <li>View real-time detection progress</li>
                <li>Download comprehensive analysis report</li>
            </ol>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose CSV file",
        type=['csv'],
        help="Upload network traffic data in CSV format"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File loaded successfully!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Packets", f"{len(df):,}")
            with col2:
                st.metric("Features", len(df.columns))
            with col3:
                st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
            
            with st.expander("📊 Preview Data (First 10 rows)"):
                st.dataframe(df.head(10), use_container_width=True)
            
            # Analyze the file
            analyze_csv_file(df, models)
        
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
    
    # Detection history
    if st.session_state.detection_history:
        st.markdown("---")
        st.subheader("📜 Session History")
        history_df = pd.DataFrame(st.session_state.detection_history[-20:])
        st.dataframe(history_df, use_container_width=True)

if __name__ == "__main__":
    main()
