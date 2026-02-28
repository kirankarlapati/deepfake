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
API_BASE_URL = "http://localhost:8000"
BACKEND_TIMEOUT = 300  # 5 minutes for analysis


############################################################
#  PAGE CONFIGURATION
############################################################
st.set_page_config(
    page_title="DeepTrace - Deepfake Forensics",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)


############################################################
#  CUSTOM CSS FOR DARK THEME
############################################################
st.markdown("""
<style>
    /* Main app styling */
    .main {
        background-color: #0e1117;
    }
    
    /* Title styling */
    .big-title {
        font-size: 4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: 1.3rem;
        color: #b0b0b0;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e1e1e 0%, #2d2d2d 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #3d3d3d;
        margin: 1rem 0;
    }
    
    .fake-badge {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 10px;
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 4px 15px rgba(231, 76, 60, 0.4);
    }
    
    .real-badge {
        background: linear-gradient(135deg, #27ae60 0%, #229954 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 10px;
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 4px 15px rgba(39, 174, 96, 0.4);
    }
    
    /* Explanation box */
    .explanation-box {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        color: #ecf0f1;
        font-size: 1.1rem;
        line-height: 1.8;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
    }
    
    /* Info box */
    .info-box {
        background: #1a1d23;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #3d3d3d;
        margin: 1rem 0;
    }
    
    /* Status indicators */
    .status-online {
        color: #27ae60;
        font-weight: bold;
    }
    
    .status-offline {
        color: #e74c3c;
        font-weight: bold;
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
    # Header
    st.markdown('<h1 class="big-title">🔍 DeepTrace</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-Powered Deepfake Forensics Analysis</p>', unsafe_allow_html=True)
    
    # Prototype Example Section
    st.markdown("")
    example_col1, example_col2, example_col3 = st.columns([1, 2, 1])
    with example_col2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.5rem; border-radius: 10px; text-align: center; 
                    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);'>
            <h3 style='color: white; margin: 0;'>📊 Example Forensics Report</h3>
            <p style='color: #e0e0e0; margin-top: 0.5rem;'>See what DeepTrace can do for judges and evaluators</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("")
        
        # Load and display prototype PDF
        try:
            with open("r8.pdf", "rb") as pdf_file:
                prototype_pdf = pdf_file.read()
            
            col_a, col_b, col_c = st.columns([1, 2, 1])
            with col_b:
                st.download_button(
                    label="📥 Download Example Report (r8.pdf)",
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
        
        # Analyze button
        st.markdown("")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            analyze_button = st.button("🔍 Analyze Video", type="primary", width="stretch")
        
        if analyze_button:
            if not backend_status:
                st.error("❌ Cannot analyze: Backend server is offline")
                return
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Analysis stages
            stages = [
                "Uploading video...",
                "Extracting frames...",
                "Running visual analysis...",
                "Analyzing audio-visual sync...",
                "Generating AI explanation...",
                "Creating PDF report...",
                "Complete!"
            ]
            
            # Simulate progress (since we can't get real-time updates from backend)
            for i, stage in enumerate(stages[:-1]):
                progress = int((i / len(stages)) * 100)
                progress_bar.progress(progress)
                status_text.markdown(f"**{stage}**")
                time.sleep(0.5)
            
            # Actual analysis
            status_text.markdown("**Processing video... This may take a few minutes.**")
            result, error = analyze_video(uploaded_file)
            
            # Complete progress
            progress_bar.progress(100)
            status_text.markdown("**✅ Analysis Complete!**")
            time.sleep(1)
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
            st.metric("Overall Confidence", f"{confidence:.2f}%", 
                     delta=None if verdict == "UNKNOWN" else None)
        
        with col2:
            st.metric("Max Fake Score", f"{visual_analysis.get('max_fake_score', 0):.2f}%",
                     help="Highest fake score detected across all frames")
            st.metric("Faces Detected", f"{visual_analysis.get('faces_detected', 0)} / {visual_analysis.get('total_frames_analyzed', 0)} frames")
        
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
            fig.add_trace(go.Scatter(x=frames, y=confidence, name="Fake Confidence %", line=dict(color="red", width=2)))
            fig.add_trace(go.Scatter(x=frames, y=noise_normalized, name="Noise Level (normalized)", line=dict(color="purple", width=2, dash="dot")))
            fig.update_layout(title="Fake Confidence vs Noise Level Per Frame", xaxis_title="Frame Number", yaxis_title="Score (0-100)", height=350, template="plotly_dark", hovermode='x unified')
            
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
            fig.add_trace(go.Scatter(x=timestamps, y=audio_sig, name="Audio Energy", line=dict(color="royalblue", width=2)))
            fig.add_trace(go.Scatter(x=timestamps, y=lip_sig, name="Lip Movement", line=dict(color="orange", width=2)))
            fig.add_trace(go.Scatter(x=mismatch_times, y=mismatch_audio, mode="markers", name="Mismatch", marker=dict(color="red", size=8, symbol="x")))
            fig.update_layout(title="Audio Energy vs Lip Movement Over Time", xaxis_title="Time (seconds)", yaxis_title="Normalized Value (0-1)", legend=dict(x=0, y=1), height=350, template="plotly_dark", hovermode='x unified')
            
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
                line=dict(color='#e74c3c', width=2),
                marker=dict(size=6)
            ))
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=lip_movement,
                mode='lines+markers',
                name='Lip Movement',
                line=dict(color='#3498db', width=2),
                marker=dict(size=6)
            ))
            
            fig.update_layout(
                title="Audio Energy vs Lip Movement at Suspicious Frames",
                xaxis_title="Timestamp (seconds)",
                yaxis_title="Normalized Value (0-1)",
                template="plotly_dark",
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
        st.markdown("## About DeepTrace")
        st.markdown("""
        DeepTrace is an AI-powered deepfake forensics tool that analyzes videos using:
        
        **🎭 Visual Analysis**
        - Deep learning model for frame-by-frame detection
        - Face detection and cropping
        - Threshold-based verdict (>70% = fake)
        
        **🎵 Audio-Visual Sync**
        - MediaPipe for lip movement tracking
        - Librosa for audio energy analysis
        - Correlation-based mismatch detection
        
        **🤖 AI Explanation**
        - Ollama LLM (llama3.2) for forensics explanation
        - Plain English interpretation
        
        **📄 PDF Report**
        - Professional forensics documentation
        - Includes all metrics and timestamps
        """)
        
        st.markdown("---")
        st.markdown("### Quick Start")
        st.code("python backend/main.py", language="bash")
        st.code("streamlit run frontend/app.py", language="bash")


############################################################
#  RUN APP
############################################################
if __name__ == "__main__":
    sidebar()
    main()
