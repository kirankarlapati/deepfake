# backend/audio_sync.py

import cv2
import numpy as np
import librosa
import mediapipe as mp
import subprocess
from typing import List, Dict, Tuple
import tempfile
import os


############################################################
#  MEDIAPIPE SETUP
############################################################
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Lip landmark indices for MediaPipe FaceMesh
# Outer lips: 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185
# Inner lips: 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191
UPPER_LIP_INDICES = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
LOWER_LIP_INDICES = [146, 91, 181, 84, 17, 314, 405, 321, 375]


############################################################
#  EXTRACT AUDIO FROM VIDEO
############################################################
def extract_audio(video_path: str) -> Tuple[np.ndarray, int]:
    """
    Extracts audio from video file using ffmpeg.
    
    Returns:
        audio_data: numpy array of audio samples
        sample_rate: sampling rate (Hz)
    """
    print(f"🎵 Extracting audio from video...")
    tmp_wav = tempfile.mktemp(suffix=".wav")
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-ac", "1", "-ar", "22050", "-f", "wav", tmp_wav],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        if result.returncode != 0 or not os.path.exists(tmp_wav):
            raise RuntimeError("ffmpeg failed to extract audio")
        audio_data, sample_rate = librosa.load(tmp_wav, sr=22050, mono=True)
        print(f"✅ Audio extracted: {len(audio_data)} samples at {sample_rate} Hz")
        return audio_data, sample_rate
    except Exception as e:
        raise RuntimeError(f"Failed to extract audio: {str(e)}")
    finally:
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)


############################################################
#  MEASURE AUDIO ENERGY
############################################################
def measure_audio_energy(audio_data: np.ndarray, sample_rate: int, fps: float) -> np.ndarray:
    """
    Measures audio energy (RMS) over time using librosa.
    Returns energy values aligned with video frames.
    
    Args:
        audio_data: Audio samples
        sample_rate: Audio sampling rate
        fps: Video frames per second
        
    Returns:
        Frame-aligned audio energy values
    """
    print(f"📊 Measuring audio energy...")
    
    # Calculate frame length in samples
    hop_length = int(sample_rate / fps)
    
    # Calculate RMS energy
    rms = librosa.feature.rms(y=audio_data, frame_length=hop_length*2, hop_length=hop_length)[0]
    
    # Normalize to 0-1 range
    if np.max(rms) > 0:
        rms_normalized = rms / np.max(rms)
    else:
        rms_normalized = rms
    
    print(f"✅ Audio energy measured: {len(rms_normalized)} frames")
    
    return rms_normalized


############################################################
#  MEASURE LIP MOVEMENT
############################################################
def measure_lip_movement(video_path: str) -> Tuple[np.ndarray, float]:
    """
    Measures lip movement over time using MediaPipe FaceMesh.
    Returns lip opening values for each frame.
    
    Args:
        video_path: Path to video file
        
    Returns:
        lip_movement: Array of lip opening values (0-1)
        fps: Video frames per second
    """
    print(f"👄 Measuring lip movement...")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    lip_movements = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            # Get lip landmarks
            h, w, _ = frame.shape
            
            # Calculate vertical lip opening (distance between upper and lower lip centers)
            upper_lip_y = np.mean([face_landmarks.landmark[i].y for i in UPPER_LIP_INDICES])
            lower_lip_y = np.mean([face_landmarks.landmark[i].y for i in LOWER_LIP_INDICES])
            
            lip_opening = abs(lower_lip_y - upper_lip_y) * h  # Convert to pixels
            lip_movements.append(lip_opening)
        else:
            # No face detected, use 0
            lip_movements.append(0.0)
        
        frame_count += 1
    
    cap.release()
    
    # Convert to numpy array and normalize
    lip_movements = np.array(lip_movements)
    
    if np.max(lip_movements) > 0:
        lip_movements = lip_movements / np.max(lip_movements)
    
    print(f"✅ Lip movement measured: {len(lip_movements)} frames at {fps:.2f} FPS")
    
    return lip_movements, fps


############################################################
#  COMPARE AUDIO AND LIP SYNC
############################################################
def compare_sync(audio_energy: np.ndarray, lip_movement: np.ndarray, 
                 fps: float, threshold: float = 0.3) -> Dict:
    """
    Compares audio energy with lip movement to detect sync issues.
    
    Args:
        audio_energy: Frame-aligned audio energy values
        lip_movement: Lip opening values per frame
        fps: Video frames per second
        threshold: Mismatch threshold for flagging issues
        
    Returns:
        Dictionary with sync analysis results
    """
    print(f"🔍 Comparing audio-visual synchronization...")
    
    # Ensure both arrays are same length
    min_len = min(len(audio_energy), len(lip_movement))
    audio_energy = audio_energy[:min_len]
    lip_movement = lip_movement[:min_len]
    
    # Smooth signals to reduce noise
    from scipy.ndimage import gaussian_filter1d
    audio_smooth = gaussian_filter1d(audio_energy, sigma=2)
    lip_smooth = gaussian_filter1d(lip_movement, sigma=2)
    
    # Calculate correlation
    correlation = np.corrcoef(audio_smooth, lip_smooth)[0, 1]
    
    # Find mismatches (where audio is high but lips aren't moving, or vice versa)
    mismatches = []
    mismatch_frames = []
    
    for i in range(len(audio_smooth)):
        audio_val = audio_smooth[i]
        lip_val = lip_smooth[i]
        
        # High audio but low lip movement
        if audio_val > 0.3 and lip_val < 0.2:
            diff = abs(audio_val - lip_val)
            if diff > threshold:
                timestamp = i / fps
                mismatches.append({
                    "frame": i,
                    "timestamp": round(timestamp, 2),
                    "audio_energy": round(float(audio_val), 3),
                    "lip_movement": round(float(lip_val), 3),
                    "difference": round(float(diff), 3),
                    "type": "audio_without_lips"
                })
                mismatch_frames.append(i)
        
        # High lip movement but low audio
        elif lip_val > 0.3 and audio_val < 0.2:
            diff = abs(audio_val - lip_val)
            if diff > threshold:
                timestamp = i / fps
                mismatches.append({
                    "frame": i,
                    "timestamp": round(timestamp, 2),
                    "audio_energy": round(float(audio_val), 3),
                    "lip_movement": round(float(lip_val), 3),
                    "difference": round(float(diff), 3),
                    "type": "lips_without_audio"
                })
                mismatch_frames.append(i)
    
    # Calculate sync score (0-100, higher is better)
    if np.isnan(correlation):
        correlation = 0.0
    
    sync_score = max(0, min(100, correlation * 100))
    
    # Overall verdict
    if len(mismatches) > min_len * 0.2:  # More than 20% frames have mismatches
        verdict = "SUSPICIOUS"
    elif len(mismatches) > min_len * 0.1:  # More than 10% frames
        verdict = "QUESTIONABLE"
    else:
        verdict = "SYNCED"
    
    print(f"✅ Sync analysis complete!")
    print(f"   Verdict: {verdict}")
    print(f"   Sync Score: {sync_score:.2f}%")
    print(f"   Correlation: {correlation:.3f}")
    print(f"   Mismatches Found: {len(mismatches)}")
    
    return {
        "verdict": verdict,
        "sync_score": round(sync_score, 2),
        "correlation": round(float(correlation), 3),
        "total_mismatches": len(mismatches),
        "mismatch_timestamps": mismatches[:20],  # Return first 20 for brevity
        "total_frames": min_len,
        "mismatch_percentage": round((len(mismatches) / min_len * 100), 2),
        "audio_signal": audio_smooth.tolist(),
        "lip_signal": lip_smooth.tolist()
    }


############################################################
#  MAIN ANALYSIS FUNCTION
############################################################
def analyze_audio_sync(video_path: str, threshold: float = 0.3) -> Dict:
    """
    Full audio-visual synchronization analysis pipeline.
    
    Args:
        video_path: Path to video file
        threshold: Mismatch threshold for flagging issues
        
    Returns:
        Complete sync analysis results with mismatch timestamps
    """
    print(f"\n{'='*60}")
    print(f"AUDIO-VISUAL SYNC ANALYSIS")
    print(f"{'='*60}")
    
    try:
        # Extract audio
        audio_data, sample_rate = extract_audio(video_path)
        
        # Measure lip movement
        lip_movement, fps = measure_lip_movement(video_path)
        
        # Measure audio energy
        audio_energy = measure_audio_energy(audio_data, sample_rate, fps)
        
        # Compare sync
        sync_results = compare_sync(audio_energy, lip_movement, fps, threshold)
        
        return {
            "success": True,
            **sync_results
        }
        
    except Exception as e:
        print(f"❌ Error during audio sync analysis: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "verdict": "ERROR",
            "sync_score": 0.0
        }


############################################################
#  QUICK TEST
############################################################
if __name__ == "__main__":
    from pathlib import Path
    
    print("\n" + "="*60)
    print("AUDIO SYNC ANALYSIS TEST")
    print("="*60)
    
    # Look for test video files
    test_video_dir = Path("test_videos")
    if test_video_dir.exists():
        video_files = list(test_video_dir.glob("*.mp4")) + \
                     list(test_video_dir.glob("*.avi")) + \
                     list(test_video_dir.glob("*.mov"))
        
        if video_files:
            test_video = video_files[0]
            print(f"\nTesting with video: {test_video}")
            
            result = analyze_audio_sync(str(test_video), threshold=0.3)
            
            if result["success"]:
                print("\n📊 RESULTS:")
                print(f"Verdict: {result['verdict']}")
                print(f"Sync Score: {result['sync_score']}%")
                print(f"Correlation: {result['correlation']}")
                print(f"Total Mismatches: {result['total_mismatches']}")
                print(f"Mismatch Percentage: {result['mismatch_percentage']}%")
                
                if result['total_mismatches'] > 0:
                    print(f"\nFirst 5 mismatch timestamps:")
                    for mismatch in result['mismatch_timestamps'][:5]:
                        print(f"  {mismatch['timestamp']}s (Frame {mismatch['frame']}): "
                              f"{mismatch['type']} - diff: {mismatch['difference']}")
            else:
                print(f"\n❌ Analysis failed: {result['error']}")
        else:
            print("⚠ No video files found in test_videos/")
            print("   Supported formats: .mp4, .avi, .mov")
    else:
        print(f"⚠ Test video directory not found: {test_video_dir}")
