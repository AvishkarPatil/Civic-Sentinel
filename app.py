import streamlit as st
import cv2
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import os
from datetime import datetime
import json
from anomaly_detector import CivicAnomalyDetector

# Page config
st.set_page_config(
    page_title="Civic Sentinel - Road Anomaly Detection",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .prediction-normal {
        background: linear-gradient(90deg, #56ab2f 0%, #a8e6cf 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .prediction-anomaly {
        background: linear-gradient(90deg, #ff416c 0%, #ff4b2b 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []
if 'detector' not in st.session_state:
    st.session_state.detector = None

@st.cache_resource
def load_model():
    """Load the trained model"""
    if os.path.exists("civic_model.pkl"):
        detector = CivicAnomalyDetector()
        detector.load_model("civic_model.pkl")
        return detector
    return None

def save_detection_history(result, image_name):
    """Save detection to history"""
    history_entry = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'image_name': image_name,
        'prediction': result['anomaly_type'],
        'confidence': result['confidence'],
        'is_anomaly': result['is_anomaly'],
        'probabilities': result['probabilities']
    }
    st.session_state.detection_history.append(history_entry)

def create_confidence_chart(probabilities):
    """Create confidence chart"""
    fig = go.Figure(data=[
        go.Bar(
            x=['Normal Road', 'Pothole'],
            y=[probabilities['plain'], probabilities['pothole']],
            marker_color=['#2E8B57', '#DC143C'],
            text=[f"{probabilities['plain']:.1%}", f"{probabilities['pothole']:.1%}"],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Prediction Confidence",
        yaxis_title="Probability",
        xaxis_title="Classification",
        showlegend=False,
        height=400,
        yaxis=dict(tickformat='.0%')
    )
    
    return fig

def create_history_chart():
    """Create detection history chart"""
    if not st.session_state.detection_history:
        return None
    
    df = pd.DataFrame(st.session_state.detection_history)
    
    # Count predictions by type
    prediction_counts = df['prediction'].value_counts()
    
    fig = px.pie(
        values=prediction_counts.values,
        names=prediction_counts.index,
        title="Detection History Summary",
        color_discrete_map={'normal': '#2E8B57', 'pothole': '#DC143C'}
    )
    
    return fig

def analyze_image_features(image_array):
    """Analyze image features for dashboard"""
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    
    # Basic image analysis
    brightness = np.mean(gray)
    contrast = np.std(gray)
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    # Texture analysis
    texture_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    return {
        'brightness': brightness,
        'contrast': contrast,
        'edge_density': edge_density,
        'texture_variance': texture_var
    }

def main():
    # Header
    st.markdown('<h2 class="main-header">🛣️ Civic Sentinel</h2>', unsafe_allow_html=True)
    st.markdown('<h4 style="text-align: center; font-size: 1.2rem; color: #666;">Your city\'s ever-watchful eye</h4>', unsafe_allow_html=True)
    
    # Load model
    detector = load_model()
    
    if detector is None:
        st.error("❌ No trained model found! Please run 'python train.py' first.")
        st.stop()
    
    # Sidebar
    st.sidebar.title("🔧 Control Panel")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Navigate to:",
        ["🏠 Home", "📊 Analytics Dashboard", "📋 Detection History", "ℹ️ About"]
    )
    
    if page == "🏠 Home":
        home_page(detector)
    elif page == "📊 Analytics Dashboard":
        analytics_page()
    elif page == "📋 Detection History":
        history_page()
    elif page == "ℹ️ About":
        about_page()

def home_page(detector):
    """Main detection page"""
    st.markdown('<h3 style="text-align: center;">Road Anomaly Detection</h3>', unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload a road image to detect anomalies"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📸 Uploaded Image")
            st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)
            
            # Image info
            st.info(f"**Image Details:**\n- Size: {image.size[0]} x {image.size[1]}\n- Format: {image.format}\n- Mode: {image.mode}")
        
        with col2:
            st.subheader("🤖 AI Analysis")
            
            # Analyze button
            if st.button("🚀 Analyze Image", type="primary"):
                with st.spinner("Analyzing image..."):
                    # Save temporary image
                    temp_path = f"temp_{uploaded_file.name}"
                    image.save(temp_path)
                    
                    try:
                        # Get prediction
                        result = detector.predict(temp_path)
                        
                        # Display prediction
                        if result['is_anomaly']:
                            st.markdown(f'<div class="prediction-anomaly">⚠️ POTHOLE DETECTED</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="prediction-normal">✅ NORMAL ROAD</div>', unsafe_allow_html=True)
                        
                        # Confidence metrics
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Confidence", f"{result['confidence']:.1%}")
                        with col_b:
                            st.metric("Prediction", result['anomaly_type'].title())
                        
                        # Confidence chart
                        fig = create_confidence_chart(result['probabilities'])
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Detailed probabilities
                        st.subheader("📊 Detailed Analysis")
                        prob_df = pd.DataFrame({
                            'Classification': ['Normal Road', 'Pothole'],
                            'Probability': [result['probabilities']['plain'], result['probabilities']['pothole']],
                            'Percentage': [f"{result['probabilities']['plain']:.1%}", f"{result['probabilities']['pothole']:.1%}"]
                        })
                        st.dataframe(prob_df, use_container_width=True)
                        
                        # Save to history
                        save_detection_history(result, uploaded_file.name)
                        
                        # Success message
                        st.success("✅ Analysis completed successfully!")
                        
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {e}")
                    
                    finally:
                        # Clean up temp file
                        if os.path.exists(temp_path):
                            os.remove(temp_path)

def analytics_page():
    """Analytics dashboard page"""
    st.header("📊 Analytics Dashboard")
    
    if not st.session_state.detection_history:
        st.info("No detection history available. Upload and analyze some images first!")
        return
    
    # Create metrics
    df = pd.DataFrame(st.session_state.detection_history)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card"><h3>Total Images</h3><h2>{}</h2></div>'.format(len(df)), unsafe_allow_html=True)
    
    with col2:
        normal_count = len(df[df['prediction'] == 'normal'])
        st.markdown('<div class="metric-card"><h3>Normal Roads</h3><h2>{}</h2></div>'.format(normal_count), unsafe_allow_html=True)
    
    with col3:
        pothole_count = len(df[df['prediction'] == 'pothole'])
        st.markdown('<div class="metric-card"><h3>Potholes</h3><h2>{}</h2></div>'.format(pothole_count), unsafe_allow_html=True)
    
    with col4:
        avg_confidence = df['confidence'].mean()
        st.markdown('<div class="metric-card"><h3>Avg Confidence</h3><h2>{:.1%}</h2></div>'.format(avg_confidence), unsafe_allow_html=True)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart
        fig_pie = create_history_chart()
        if fig_pie:
            st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Confidence over time
        fig_line = px.line(
            df, 
            x='timestamp', 
            y='confidence',
            color='prediction',
            title="Confidence Over Time",
            color_discrete_map={'normal': '#2E8B57', 'pothole': '#DC143C'}
        )
        st.plotly_chart(fig_line, use_container_width=True)

def history_page():
    """Detection history page"""
    st.header("📋 Detection History")
    
    if not st.session_state.detection_history:
        st.info("No detection history available.")
        return
    
    # Controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🗑️ Clear History"):
            st.session_state.detection_history = []
            st.rerun()
    
    with col2:
        if st.button("📥 Download History"):
            df = pd.DataFrame(st.session_state.detection_history)
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"detection_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    # Display history
    df = pd.DataFrame(st.session_state.detection_history)
    
    # Add filters
    st.subheader("🔍 Filters")
    col1, col2 = st.columns(2)
    
    with col1:
        prediction_filter = st.selectbox("Filter by Prediction:", ["All", "normal", "pothole"])
    
    with col2:
        confidence_filter = st.slider("Minimum Confidence:", 0.0, 1.0, 0.0)
    
    # Apply filters
    filtered_df = df.copy()
    if prediction_filter != "All":
        filtered_df = filtered_df[filtered_df['prediction'] == prediction_filter]
    filtered_df = filtered_df[filtered_df['confidence'] >= confidence_filter]
    
    # Display filtered results
    st.subheader(f"📊 Results ({len(filtered_df)} items)")
    
    for idx, row in filtered_df.iterrows():
        with st.expander(f"🖼️ {row['image_name']} - {row['prediction'].title()} ({row['confidence']:.1%})"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Timestamp:** {row['timestamp']}")
                st.write(f"**Prediction:** {row['prediction'].title()}")
                st.write(f"**Confidence:** {row['confidence']:.1%}")
                st.write(f"**Is Anomaly:** {'Yes' if row['is_anomaly'] else 'No'}")
            
            with col2:
                # Mini confidence chart
                mini_fig = go.Figure(data=[
                    go.Bar(
                        x=['Normal', 'Pothole'],
                        y=[row['probabilities']['plain'], row['probabilities']['pothole']],
                        marker_color=['#2E8B57', '#DC143C']
                    )
                ])
                mini_fig.update_layout(height=200, showlegend=False, title="Probabilities")
                st.plotly_chart(mini_fig, use_container_width=True)

def about_page():
    """About page"""
    st.header("ℹ️ About Civic Sentinel")
    
    st.markdown("""
    ## 🛣️ Civic Sentinel - AI Road Anomaly Detection
    
    **Civic Sentinel** is an advanced computer vision system designed to automatically detect road anomalies, 
    specifically potholes, to help municipal authorities maintain better road infrastructure.
    
    ### 🎯 Key Features:
    - **AI-Powered Detection**: Uses machine learning to identify road anomalies
    - **High Accuracy**: Trained on real road images for reliable predictions
    - **Interactive Dashboard**: Easy-to-use web interface for image analysis
    - **Detection History**: Track and analyze detection patterns over time
    - **Export Capabilities**: Download detection history for reporting
    
    ### 🔧 Technical Details:
    - **Algorithm**: Random Forest Classifier
    - **Features**: 17 different image features including edges, texture, color, and shape
    - **Training Data**: Plain roads vs. Pothole images
    - **Framework**: Streamlit for web interface, OpenCV for image processing
    
    ### 📊 Model Performance:
    - **Training Accuracy**: ~95%
    - **Test Accuracy**: ~92%
    - **Feature Extraction**: Edge density, texture variance, color analysis, contour detection
    
    ### 🚀 How to Use:
    1. Upload a road image using the file uploader
    2. Click "Analyze Image" to get AI prediction
    3. View confidence scores and detailed analysis
    4. Check analytics dashboard for insights
    5. Review detection history for patterns
    
    ### 👨‍💻 Developed by:
    **AI Development Team** - Civic Infrastructure Solutions
    
    ---
    *For technical support or feature requests, please contact the development team.*
    """)
    
    # System stats
    st.subheader("📈 System Statistics")
    
    if os.path.exists("civic_model.pkl"):
        st.success("✅ AI Model: Loaded and Ready")
    else:
        st.error("❌ AI Model: Not Found")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Detections", len(st.session_state.detection_history))
    
    with col2:
        if st.session_state.detection_history:
            avg_conf = np.mean([h['confidence'] for h in st.session_state.detection_history])
            st.metric("Average Confidence", f"{avg_conf:.1%}")
        else:
            st.metric("Average Confidence", "N/A")
    
    with col3:
        st.metric("Model Version", "1.0.0")

if __name__ == "__main__":
    main()