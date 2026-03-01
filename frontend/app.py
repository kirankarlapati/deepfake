# frontend/app.py

import streamlit as st
import requests
import time
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go

############################################################
#  CONFIGURATION
############################################################
API_BASE_URL = "https://aeruginous-vizierial-teofila.ngrok-free.dev"
BACKEND_TIMEOUT = 300  # 5 minutes for analysis


############################################################
#  PAGE CONFIGURATION
############################################################
st.set_page_config(
    page_title="DeepTrace - Deepfake Forensics",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


############################################################
#  MATERIAL DESIGN 3 (MATERIAL YOU) STYLING
############################################################
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
    
    /* Material Design 3 Color Tokens - Dark Theme */
    :root {
        --md-sys-color-primary: #A8C7FA;
        --md-sys-color-on-primary: #062E6F;
        --md-sys-color-primary-container: #1E4489;
        --md-sys-color-on-primary-container: #D6E3FF;
        
        --md-sys-color-secondary: #BDC7DC;
        --md-sys-color-on-secondary: #283141;
        --md-sys-color-secondary-container: #3E4759;
        --md-sys-color-on-secondary-container: #D9E3F8;
        
        --md-sys-color-error: #FFB4AB;
        --md-sys-color-on-error: #690005;
        --md-sys-color-error-container: #93000A;
        --md-sys-color-on-error-container: #FFDAD6;
        
        --md-sys-color-success: #7FDB8C;
        --md-sys-color-on-success: #003910;
        --md-sys-color-success-container: #005319;
        --md-sys-color-on-success-container: #9CF7A6;
        
        --md-sys-color-surface: #1A1C1E;
        --md-sys-color-surface-dim: #111316;
        --md-sys-color-surface-bright: #37393C;
        --md-sys-color-surface-container-lowest: #0C0E10;
        --md-sys-color-surface-container-low: #1A1C1E;
        --md-sys-color-surface-container: #1E2022;
        --md-sys-color-surface-container-high: #282A2D;
        --md-sys-color-surface-container-highest: #333538;
        
        --md-sys-color-on-surface: #E3E2E6;
        --md-sys-color-on-surface-variant: #C4C6D0;
        --md-sys-color-outline: #8E9099;
        --md-sys-color-outline-variant: #44464F;
    }
    
    /* Base App Styling */
    .main {
        background-color: var(--md-sys-color-surface);
        font-family: 'Roboto', sans-serif;
    }
    
    .stApp {
        background-color: var(--md-sys-color-surface);
        color: var(--md-sys-color-on-surface);
    }
    
    /* Material Design 3 Typography Scale */
    h1, h2, h3 {
        font-family: 'Roboto', sans-serif;
        color: var(--md-sys-color-on-surface);
    }
    
    /* Title - Display Large */
    .big-title {
        font-size: 3.5rem;
        font-weight: 400;
        line-height: 4rem;
        letter-spacing: -0.25px;
        color: var(--md-sys-color-primary);
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    /* Subtitle - Title Large */
    .subtitle {
        font-size: 1.375rem;
        font-weight: 400;
        line-height: 1.75rem;
        color: var(--md-sys-color-on-surface-variant);
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Material Design 3 Buttons - Filled Buttons */
    .stButton > button {
        background-color: var(--md-sys-color-primary);
        color: var(--md-sys-color-on-primary);
        border: none;
        border-radius: 20px;
        padding: 0.625rem 1.5rem;
        font-family: 'Roboto', sans-serif;
        font-weight: 500;
        font-size: 0.875rem;
        letter-spacing: 0.1px;
        text-transform: none;
        box-shadow: 0 1px 3px 1px rgba(0, 0, 0, 0.15);
        transition: all 0.2s cubic-bezier(0.2, 0, 0, 1);
    }
    
    .stButton > button:hover {
        background-color: #B8D4FF;
        box-shadow: 0 2px 6px 2px rgba(0, 0, 0, 0.2);
    }
    
    .stButton > button:active {
        box-shadow: 0 1px 3px 1px rgba(0, 0, 0, 0.15);
    }
    
    /* Download Button - Filled Tonal Button */
    .stDownloadButton > button {
        background-color: var(--md-sys-color-secondary-container);
        color: var(--md-sys-color-on-secondary-container);
        border: none;
        border-radius: 20px;
        padding: 0.625rem 1.5rem;
        font-family: 'Roboto', sans-serif;
        font-weight: 500;
        font-size: 0.875rem;
        letter-spacing: 0.1px;
        box-shadow: 0 1px 3px 1px rgba(0, 0, 0, 0.15);
        transition: all 0.2s cubic-bezier(0.2, 0, 0, 1);
    }
    
    .stDownloadButton > button:hover {
        background-color: #4A5468;
        box-shadow: 0 2px 6px 2px rgba(0, 0, 0, 0.2);
    }
    
    /* Material Design 3 Cards - Elevated Surface */
    .metric-card {
        background-color: var(--md-sys-color-surface-container-high);
        padding: 1.5rem;
        border-radius: 16px;
        margin: 1rem 0;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.3), 0 2px 6px 2px rgba(0, 0, 0, 0.15);
    }
    
    /* Error State Badge - Material 3 */
    .fake-badge {
        background-color: var(--md-sys-color-error-container);
        color: var(--md-sys-color-on-error-container);
        padding: 1.5rem 2rem;
        border-radius: 28px;
        font-size: 2rem;
        font-weight: 500;
        text-align: center;
        box-shadow: 0 4px 8px 3px rgba(0, 0, 0, 0.15), 0 1px 3px 0 rgba(0, 0, 0, 0.3);
    }
    
    /* Success State Badge - Material 3 */
    .real-badge {
        background-color: var(--md-sys-color-success-container);
        color: var(--md-sys-color-on-success-container);
        padding: 1.5rem 2rem;
        border-radius: 28px;
        font-size: 2rem;
        font-weight: 500;
        text-align: center;
        box-shadow: 0 4px 8px 3px rgba(0, 0, 0, 0.15), 0 1px 3px 0 rgba(0, 0, 0, 0.3);
    }
    
    /* Gauge Container - Filled Card */
    .gauge-container {
        background-color: var(--md-sys-color-surface-container-highest);
        padding: 2rem;
        border-radius: 24px;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.3), 0 2px 6px 2px rgba(0, 0, 0, 0.15);
        text-align: center;
    }
    
    /* Explanation Box - Outlined Card */
    .explanation-box {
        background-color: var(--md-sys-color-surface-container);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid var(--md-sys-color-outline-variant);
        color: var(--md-sys-color-on-surface);
        font-size: 1rem;
        line-height: 1.5rem;
        font-weight: 400;
    }
    
    /* Info Box */
    .info-box {
        background-color: var(--md-sys-color-surface-container-low);
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid var(--md-sys-color-outline-variant);
        margin: 1rem 0;
    }
    
    /* Video Preview Container - Elevated */
    .video-preview {
        background-color: var(--md-sys-color-surface-container);
        padding: 1.5rem;
        border-radius: 16px;
        margin: 1rem 0;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.3), 0 1px 3px 1px rgba(0, 0, 0, 0.15);
    }
    
    /* Status Indicators */
    .status-online {
        color: var(--md-sys-color-success);
        font-weight: 500;
    }
    
    .status-offline {
        color: var(--md-sys-color-error);
        font-weight: 500;
    }
    
    /* Progress Bar - Material 3 */
    .stProgress > div > div > div {
        background-color: var(--md-sys-color-primary);
    }
    
    .stProgress > div > div {
        background-color: var(--md-sys-color-surface-container-highest);
    }
    
    /* Metric Styling */
    [data-testid="stMetricValue"] {
        color: var(--md-sys-color-on-surface);
        font-size: 2rem;
        font-weight: 400;
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--md-sys-color-on-surface-variant);
        font-size: 0.875rem;
        font-weight: 500;
    }
    
    /* File Uploader - Material 3 */
    [data-testid="stFileUploader"] {
        background-color: var(--md-sys-color-surface-container-low);
        border: 2px dashed var(--md-sys-color-outline);
        border-radius: 16px;
        padding: 2rem;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: var(--md-sys-color-primary);
        background-color: var(--md-sys-color-surface-container);
    }
    
    /* Divider - Material 3 */
    hr {
        border: none;
        border-top: 1px solid var(--md-sys-color-outline-variant);
        margin: 2rem 0;
    }
    
    /* Expander - Material 3 */
    .streamlit-expander {
        background-color: var(--md-sys-color-surface-container);
        border: 1px solid var(--md-sys-color-outline-variant);
        border-radius: 12px;
        margin: 0.5rem 0;
    }
    
    /* Sidebar - Material 3 Navigation Drawer */
    [data-testid="stSidebar"] {
        background-color: var(--md-sys-color-surface-container-low);
        border-right: 1px solid var(--md-sys-color-outline-variant);
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: var(--md-sys-color-on-surface);
    }
    
    /* Code Block - Material 3 */
    code {
        background-color: var(--md-sys-color-surface-container-highest);
        color: var(--md-sys-color-primary);
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-family: 'Roboto Mono', monospace;
    }
    
    .stCodeBlock {
        background-color: var(--md-sys-color-surface-container-highest);
        border-radius: 12px;
        border: 1px solid var(--md-sys-color-outline-variant);
    }
</style>
""", unsafe_allow_html=True)


############################################################
#  HELPER FUNCTIONS
############################################################
def check_backend_health():
    """Check if the FastAPI backend is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=3)
        return response.status_code == 200
    except:
        return False


def create_confidence_gauge(confidence, verdict):
    """Create a color-coded confidence gauge visualization with Material Design 3 colors."""
    # Determine color based on verdict and confidence using MD3 tokens
    if verdict == "FAKE":
        color = "#FFB4AB"  # md-sys-color-error
        label_color = "#FFB4AB"
    elif verdict == "REAL":
        color = "#7FDB8C"  # md-sys-color-success
        label_color = "#7FDB8C"
    else:
        color = "#BDC7DC"  # md-sys-color-secondary
        label_color = "#BDC7DC"
    
    # Create gauge chart with Material Design 3 styling
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = confidence,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence Score", 'font': {'size': 20, 'color': '#E3E2E6', 'family': 'Roboto'}},
        number = {'suffix': "%", 'font': {'size': 44, 'color': label_color, 'family': 'Roboto'}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#C4C6D0"},
            'bar': {'color': label_color, 'thickness': 0.7},
            'bgcolor': "#1E2022",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 30], 'color': 'rgba(127, 219, 140, 0.15)'},
                {'range': [30, 70], 'color': 'rgba(189, 199, 220, 0.15)'},
                {'range': [70, 100], 'color': 'rgba(255, 180, 171, 0.15)'}
            ],
            'threshold': {
                'line': {'color': "#E3E2E6", 'width': 3},
                'thickness': 0.8,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor = "#282A2D",
        plot_bgcolor = "#282A2D",
        font = {'color': "#E3E2E6", 'family': "Roboto"},
        height = 300,
        margin = dict(l=20, r=20, t=60, b=20)
    )
    
    return fig


def analyze_video(video_file):
    """Send video to backend for analysis."""
    try:
        files = {"file": (video_file.name, video_file, video_file.type)}
        response = requests.post(
            f"{API_BASE_URL}/analyze",
            files=files,
            timeout=BACKEND_TIMEOUT
        )
        
        if response.status_code == 200:
            return response.json(), None
        else:
            error_detail = response.json().get("detail", "Unknown error")
            return None, f"Analysis failed: {error_detail}"
    
    except requests.exceptions.Timeout:
        return None, "Analysis timed out. Please try with a shorter video."
    
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to backend. Make sure the API server is running at http://localhost:8000"
    
    except Exception as e:
        return None, f"Error: {str(e)}"


def download_pdf(pdf_filename):
    """Download PDF report from backend."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/download/{pdf_filename}",
            timeout=10
        )
        
        if response.status_code == 200:
            return response.content
        else:
            return None
    
    except Exception as e:
        st.error(f"Failed to download PDF: {str(e)}")
        return None


############################################################
#  MAIN APP
############################################################
def main():
    # Material Design 3 Header
    st.markdown('<h1 class="big-title">🔍 DeepTrace</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-Powered Deepfake Forensics Analysis</p>', unsafe_allow_html=True)
    
    # Material Design 3 Example Section - Elevated Card
    st.markdown("")
    example_col1, example_col2, example_col3 = st.columns([1, 2, 1])
    with example_col2:
        st.markdown("""
        <div style='background-color: var(--md-sys-color-primary-container); 
                    color: var(--md-sys-color-on-primary-container);
                    padding: 1.5rem; border-radius: 16px; text-align: center; 
                    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.3), 0 2px 6px 2px rgba(0, 0, 0, 0.15);'>
            <h3 style='color: var(--md-sys-color-on-primary-container); margin: 0; font-weight: 500;'>📊 Example Forensics Report</h3>
            <p style='color: var(--md-sys-color-on-primary-container); margin-top: 0.5rem; opacity: 0.9;'>See what DeepTrace can do for judges and evaluators</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("")
        
        # Load and display prototype PDF
        try:
            with open("../r14.pdf", "rb") as pdf_file:
                prototype_pdf = pdf_file.read()
            
            col_a, col_b, col_c = st.columns([1, 2, 1])
            with col_b:
                st.download_button(
                    label="📥 Download Example Report (r14.pdf)",
                    data=prototype_pdf,
                    file_name="DeepTrace_Example_Report.pdf",
                    mime="application/pdf",
                    type="secondary",
                    width="stretch",
                    help="Download this example to see the detailed forensics analysis DeepTrace generates"
                )
        except FileNotFoundError:
            st.info("ℹ️ Example report not found. Upload a video below to generate your own analysis!")
    
    st.markdown("---")
    
    # Backend status check
    backend_status = check_backend_health()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if backend_status:
            st.markdown('✅ <span class="status-online">Backend Server: Online</span>', unsafe_allow_html=True)
        else:
            st.markdown('❌ <span class="status-offline">Backend Server: Offline</span>', unsafe_allow_html=True)
            st.warning("⚠️ Backend server is not running. Start it with: `python backend/main.py`")
    
    st.markdown("---")
    
    # File uploader
    st.markdown("### 📹 Upload Video for Analysis")
    
    uploaded_file = st.file_uploader(
        "Drag and drop your video file here",
        type=["mp4", "avi", "mov", "MP4", "AVI", "MOV"],
        help="Supported formats: MP4, AVI, MOV (max 200MB recommended)"
    )
    
    if uploaded_file is not None:
        # Display video info
        file_size_mb = uploaded_file.size / (1024 * 1024)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📁 Filename", uploaded_file.name)
        with col2:
            st.metric("📊 Size", f"{file_size_mb:.2f} MB")
        with col3:
            st.metric("🎬 Format", uploaded_file.name.split('.')[-1].upper())
        
        st.markdown("")
        
        # Video Preview Section
        st.markdown("### 🎥 Video Preview")
        st.markdown('<div class="video-preview">', unsafe_allow_html=True)
        
        col_preview1, col_preview2, col_preview3 = st.columns([1, 2, 1])
        with col_preview2:
            st.video(uploaded_file)
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("")
        
        # Analyze button
        st.markdown("")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            analyze_button = st.button("🔍 Analyze Video", type="primary", width="stretch")
        
        if analyze_button:
            if not backend_status:
                st.error("❌ Cannot analyze: Backend server is offline")
                return
            
            # Progress section with enhanced styling
            st.markdown("---")
            st.markdown("### ⚙️ Analysis in Progress")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Analysis stages with emojis
            stages = [
                ("📤", "Uploading video to server..."),
                ("🎞️", "Extracting frames from video..."),
                ("👁️", "Running visual deepfake detection..."),
                ("🎵", "Analyzing audio-visual synchronization..."),
                ("🤖", "Generating AI forensics explanation..."),
                ("📄", "Creating detailed PDF report..."),
            ]
            
            # Simulate progress with styled messages
            for i, (emoji, stage) in enumerate(stages):
                progress = int((i / len(stages)) * 100)
                progress_bar.progress(progress)
                status_text.markdown(
                    f'<div style="text-align: center; font-size: 1.1rem; color: var(--md-sys-color-primary); padding: 1rem; font-family: Roboto;">'
                    f'{emoji} <b style="font-weight: 500;">{stage}</b></div>',
                    unsafe_allow_html=True
                )
                time.sleep(0.5)
            
            # Actual analysis
            status_text.markdown(
                f'<div style="text-align: center; font-size: 1.1rem; color: var(--md-sys-color-on-surface); padding: 1rem; font-family: Roboto;">'
                f'🔬 <b style="font-weight: 500;">Deep analysis in progress... This may take a few minutes.</b></div>',
                unsafe_allow_html=True
            )
            result, error = analyze_video(uploaded_file)
            
            # Complete progress with success animation
            progress_bar.progress(100)
            status_text.markdown(
                f'<div style="text-align: center; font-size: 1.2rem; color: var(--md-sys-color-success); padding: 1rem; font-family: Roboto;">'
                f'✅ <b style="font-weight: 500;">Analysis Complete!</b></div>',
                unsafe_allow_html=True
            )
            time.sleep(1.5)
            progress_bar.empty()
            status_text.empty()
            
            if error:
                st.error(f"❌ {error}")
                return
            
            # Store results in session state
            st.session_state['analysis_results'] = result
    
    # Display results if available
    if 'analysis_results' in st.session_state:
        results = st.session_state['analysis_results']
        
        st.markdown("---")
        st.markdown("## 📊 Analysis Results")
        st.markdown("")
        
        # Extract data
        visual_analysis = results.get('visual_analysis', {})
        audio_analysis = results.get('audio_sync_analysis', {})
        explanation = results.get('explanation', {})
        report = results.get('report', {})
        
        verdict = visual_analysis.get('overall_verdict', 'UNKNOWN')
        confidence = visual_analysis.get('overall_confidence', 0)
        
        # ============================================================
        # SECTION 1: VISUAL ANALYSIS
        # ============================================================
        st.markdown("### 🎭 Visual Analysis")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Big colored verdict badge
            if verdict == "FAKE":
                st.markdown(
                    f'<div class="fake-badge">⚠️ DEEPFAKE DETECTED</div>',
                    unsafe_allow_html=True
                )
            elif verdict == "REAL":
                st.markdown(
                    f'<div class="real-badge">✓ AUTHENTIC VIDEO</div>',
                    unsafe_allow_html=True
                )
            else:
                st.warning("⚠️ INCONCLUSIVE")
            
            st.markdown("")
            
            # Additional metrics
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Max Fake Score", f"{visual_analysis.get('max_fake_score', 0):.2f}%",
                         help="Highest fake score detected across all frames")
            with col_b:
                st.metric("Faces Detected", f"{visual_analysis.get('faces_detected', 0)} / {visual_analysis.get('total_frames_analyzed', 0)} frames")
        
        with col2:
            # Color-coded confidence gauge
            st.markdown('<div class="gauge-container">', unsafe_allow_html=True)
            gauge_fig = create_confidence_gauge(confidence, verdict)
            st.plotly_chart(gauge_fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("")
        
        # Noise level vs fake confidence visualization
        if "noise_scores" in visual_analysis and len(visual_analysis["noise_scores"]) > 0:
            st.markdown("#### 🔬 Fake Confidence vs Noise Level Per Frame")
            
            frames = visual_analysis["frame_numbers"]
            noise = visual_analysis["noise_scores"]
            confidence = [r["fake_score"] for r in visual_analysis["frame_results"] if r["face_detected"]]
            
            # Normalize noise scores to 0-100 scale for comparison
            max_noise = max(noise) if max(noise) > 0 else 1
            noise_normalized = [n / max_noise * 100 for n in noise]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=frames, y=confidence, name="Fake Confidence %", line=dict(color="#FFB4AB", width=2)))
            fig.add_trace(go.Scatter(x=frames, y=noise_normalized, name="Noise Level (normalized)", line=dict(color="#A8C7FA", width=2, dash="dot")))
            fig.update_layout(
                title="Fake Confidence vs Noise Level Per Frame",
                xaxis_title="Frame Number",
                yaxis_title="Score (0-100)",
                height=350,
                paper_bgcolor="#1A1C1E",
                plot_bgcolor="#1A1C1E",
                font=dict(family="Roboto", color="#E3E2E6"),
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, width="stretch")
            
            st.markdown("")
        
        # ============================================================
        # SECTION 2: AUDIO-VISUAL SYNC
        # ============================================================
        st.markdown("### 🎵 Audio-Visual Synchronization")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            sync_verdict = audio_analysis.get('verdict', 'UNKNOWN')
            sync_score = audio_analysis.get('sync_score', 0)
            
            if sync_verdict == "SYNCED":
                st.success(f"✅ **{sync_verdict}**")
            elif sync_verdict == "QUESTIONABLE":
                st.warning(f"⚠️ **{sync_verdict}**")
            elif sync_verdict == "SUSPICIOUS":
                st.error(f"🚨 **{sync_verdict}**")
            else:
                st.info(f"ℹ️ **{sync_verdict}**")
            
            st.metric("Sync Score", f"{sync_score:.2f}%",
                     help="Higher score = better synchronization")
            st.metric("Correlation", f"{audio_analysis.get('correlation', 0):.3f}",
                     help="Audio-visual correlation coefficient (-1 to 1)")
        
        with col2:
            st.metric("Total Mismatches", audio_analysis.get('total_mismatches', 0))
            st.metric("Mismatch Percentage", f"{audio_analysis.get('mismatch_percentage', 0):.2f}%")
            st.metric("Total Frames", audio_analysis.get('total_frames', 0))
        
        # Full audio energy vs lip movement chart with mismatch highlights
        if audio_analysis.get("audio_signal") and audio_analysis.get("lip_signal"):
            st.markdown("#### 📊 Audio Energy vs Lip Movement Over Time")
            
            audio_sig = audio_analysis["audio_signal"]
            lip_sig = audio_analysis["lip_signal"]
            timestamps = [i / 30 for i in range(len(audio_sig))]
            
            mismatch_times = [m["timestamp"] for m in audio_analysis.get("mismatch_timestamps", [])]
            mismatch_audio = [audio_sig[min(int(m["frame"]), len(audio_sig)-1)] for m in audio_analysis.get("mismatch_timestamps", [])]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=timestamps, y=audio_sig, name="Audio Energy", line=dict(color="#A8C7FA", width=2)))
            fig.add_trace(go.Scatter(x=timestamps, y=lip_sig, name="Lip Movement", line=dict(color="#BDC7DC", width=2)))
            fig.add_trace(go.Scatter(x=mismatch_times, y=mismatch_audio, mode="markers", name="Mismatch", marker=dict(color="#FFB4AB", size=8, symbol="x")))
            fig.update_layout(
                title="Audio Energy vs Lip Movement Over Time",
                xaxis_title="Time (seconds)",
                yaxis_title="Normalized Value (0-1)",
                legend=dict(x=0, y=1),
                height=350,
                paper_bgcolor="#1A1C1E",
                plot_bgcolor="#1A1C1E",
                font=dict(family="Roboto", color="#E3E2E6"),
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, width="stretch")
        
        # Line chart of mismatches
        mismatch_timestamps = audio_analysis.get('mismatch_timestamps', [])
        
        if len(mismatch_timestamps) > 0:
            st.markdown("#### 📈 Mismatch Timeline")
            
            # Prepare data for chart
            timestamps = [m['timestamp'] for m in mismatch_timestamps]
            audio_energy = [m['audio_energy'] for m in mismatch_timestamps]
            lip_movement = [m['lip_movement'] for m in mismatch_timestamps]
            
            # Create plotly chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=audio_energy,
                mode='lines+markers',
                name='Audio Energy',
                line=dict(color='#FFB4AB', width=2),
                marker=dict(size=6)
            ))
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=lip_movement,
                mode='lines+markers',
                name='Lip Movement',
                line=dict(color='#A8C7FA', width=2),
                marker=dict(size=6)
            ))
            
            fig.update_layout(
                title="Audio Energy vs Lip Movement at Suspicious Frames",
                xaxis_title="Timestamp (seconds)",
                yaxis_title="Normalized Value (0-1)",
                paper_bgcolor="#1A1C1E",
                plot_bgcolor="#1A1C1E",
                font=dict(family="Roboto", color="#E3E2E6"),
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig, width="stretch")
            
            # Table of suspicious frames
            with st.expander("🔍 View Suspicious Frame Details"):
                df = pd.DataFrame(mismatch_timestamps)
                df = df[['timestamp', 'frame', 'audio_energy', 'lip_movement', 'type']]
                df.columns = ['Timestamp (s)', 'Frame #', 'Audio Energy', 'Lip Movement', 'Anomaly Type']
                st.dataframe(df, width="stretch")
        else:
            st.info("✅ No significant audio-visual synchronization anomalies detected.")
        
        st.markdown("")
        
        # ============================================================
        # SECTION 3: FORENSICS EXPLANATION
        # ============================================================
        st.markdown("### 🤖 AI Forensics Explanation")
        
        explanation_text = explanation.get('text', 'No explanation available.')
        explanation_model = explanation.get('model', 'Unknown')
        
        st.markdown(
            f'<div class="explanation-box">{explanation_text}</div>',
            unsafe_allow_html=True
        )
        
        st.caption(f"Generated by: {explanation_model}")
        
        st.markdown("")
        
        # ============================================================
        # PDF DOWNLOAD BUTTON
        # ============================================================
        st.markdown("### 📄 Download Report")
        
        pdf_filename = report.get('pdf_filename', '')
        
        if pdf_filename:
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col2:
                # Download PDF
                pdf_content = download_pdf(pdf_filename)
                
                if pdf_content:
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=pdf_content,
                        file_name=pdf_filename,
                        mime="application/pdf",
                        type="primary",
                        width="stretch"
                    )
                else:
                    st.warning("⚠️ PDF report not available for download")
        
        st.markdown("---")
        
        # Additional info
        with st.expander("ℹ️ Technical Details"):
            st.json(results)


############################################################
#  SIDEBAR
############################################################
def sidebar():
    with st.sidebar:
        st.markdown("# 🔍 DeepTrace")
        st.markdown("*AI-Powered Deepfake Forensics*")
        st.markdown("---")
        
        st.markdown("## 📖 About")
        st.markdown("""
        DeepTrace is an advanced AI forensics tool that detects deepfake videos 
        using multi-modal analysis techniques.
        """)
        
        st.markdown("---")
        st.markdown("## ⚙️ Technologies")
        
        with st.expander("🎭 Visual Analysis"):
            st.markdown("""
            - **Deep Learning Model**: Frame-by-frame detection
            - **Face Detection**: Automatic face cropping
            - **Threshold**: >70% confidence = deepfake
            - **Noise Analysis**: Compression artifact detection
            """)
        
        with st.expander("🎵 Audio-Visual Sync"):
            st.markdown("""
            - **MediaPipe**: Lip movement tracking
            - **Librosa**: Audio energy analysis
            - **Correlation**: Mismatch detection algorithm
            - **Timeline**: Frame-by-frame synchronization
            """)
        
        with st.expander("🤖 AI Explanation"):
            st.markdown("""
            - **LLM Model**: Ollama (llama3.2)
            - **Natural Language**: Plain English forensics
            - **Context-Aware**: Analyzes all metrics
            - **Professional**: Court-ready explanations
            """)
        
        with st.expander("📄 PDF Report"):
            st.markdown("""
            - **Comprehensive**: All metrics & timestamps
            - **Professional**: Court-ready documentation
            - **Visualizations**: Charts & graphs
            - **Downloadable**: PDF format
            """)
        
        st.markdown("---")
        st.markdown("## 🚀 Quick Start")
        
        st.markdown("**1. Start Backend Server:**")
        st.code("python backend/main.py", language="bash")
        
        st.markdown("**2. Launch Frontend:**")
        st.code("streamlit run frontend/app.py", language="bash")
        
        st.markdown("**3. Upload & Analyze:**")
        st.markdown("Upload a video and click **Analyze**!")
        
        st.markdown("---")
        st.markdown("## 📊 Supported Formats")
        st.markdown("✅ MP4, AVI, MOV")
        st.markdown("💾 Max 200MB recommended")
        
        st.markdown("---")
        st.markdown("## 💡 Tips")
        st.info("""
        **For Best Results:**
        - Use high-quality videos
        - Ensure good lighting
        - Clear facial visibility
        - Minimal background noise
        """)
        
        st.markdown("---")
        st.markdown("### 🛡️ Powered by AI")
        st.caption("Version 1.0 | © 2026 DeepTrace")


############################################################
#  RUN APP
############################################################
if __name__ == "__main__":
    sidebar()
    main()
