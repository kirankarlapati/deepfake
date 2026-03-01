# backend/detector.py

from pathlib import Path
from PIL import Image
import torch
import numpy as np
import cv2
from typing import List, Dict, Tuple
import tempfile
import os

############################################################
#  MODEL PATH
############################################################
MODEL_PATH = Path("models/weights/deepfake_model")

print("Loading model from:", MODEL_PATH)

############################################################
#  TRY TRANSFORMERS LOADER (Version A)
############################################################
USE_TRANSFORMERS = True    # <--- CHANGE THIS TO False IF IT FAILS

if USE_TRANSFORMERS:
    try:
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        print("Trying Transformers loader...")
        processor = AutoImageProcessor.from_pretrained(str(MODEL_PATH))
        model = AutoModelForImageClassification.from_pretrained(str(MODEL_PATH))
        model.eval()

        def preprocess(image):
            return processor(images=image, return_tensors="pt")

        print("Transformers model loaded successfully ✅")

    except Exception as e:
        print("\n❌ Transformers loader failed:", e)
        print("Switching to PyTorch fallback loader...\n")
        USE_TRANSFORMERS = False


############################################################
#  PYTORCH FALLBACK LOADER (Version B)
############################################################
if not USE_TRANSFORMERS:
    # Detect .pth / .pt / .bin file automatically
    weight_files = list(MODEL_PATH.glob("*.pth")) + \
                   list(MODEL_PATH.glob("*.pt")) + \
                   list(MODEL_PATH.glob("*.bin"))

    if len(weight_files) == 0:
        raise FileNotFoundError(
            f"No model weights found in {MODEL_PATH}. "
            "Supported formats: .pth, .pt, .bin"
        )

    weight_file = weight_files[0]
    print("Using PyTorch weights:", weight_file)

    # Load model state_dict
    model = torch.load(weight_file, map_location="cpu")
    model.eval()

    # SIMPLE DEFAULT PREPROCESS
    import torchvision.transforms as T
    preprocess_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])

    def preprocess(image):
        return {"pixel_values": preprocess_transform(image).unsqueeze(0)}

    # Fallback label names if missing
    id2label = {0: "REAL", 1: "FAKE"}

    print("PyTorch model loaded successfully ✅")


############################################################
#  PREDICT ON A SINGLE FRAME
############################################################
def predict_frame(image_path: str) -> dict:
    image = Image.open(image_path).convert("RGB")

    inputs = preprocess(image)

    with torch.no_grad():
        if USE_TRANSFORMERS:
            outputs = model(**inputs)
            logits = outputs.logits
        else:
            logits = model(inputs["pixel_values"])

    probs = torch.softmax(logits, dim=1)[0]

    # If transformers model, use its config labels
    if USE_TRANSFORMERS:
        id2label = model.config.id2label
    else:
        id2label = {0: "REAL", 1: "FAKE"}

    scores = {id2label[i]: round(float(probs[i]) * 100, 2) for i in range(len(probs))}
    predicted_label = max(scores, key=scores.get)
    confidence = scores[predicted_label]

    return {
        "label": predicted_label,
        "confidence": confidence,
        "all_scores": scores
    }


############################################################
#  FACE DETECTION USING OPENCV
############################################################
def detect_faces(frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Detects faces in a frame using OpenCV's Haar Cascade.
    Returns list of (x, y, w, h) tuples.
    """
    # Load pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    return faces


############################################################
#  EXTRACT FRAMES FROM VIDEO
############################################################
def extract_frames(video_path: str, frame_interval: int = 5) -> List[Tuple[int, np.ndarray]]:
    """
    Extracts every Nth frame from a video.
    Returns list of (frame_number, frame_image) tuples.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract every Nth frame
        if frame_count % frame_interval == 0:
            frames.append((frame_count, frame))
        
        frame_count += 1
    
    cap.release()
    return frames


############################################################
#  SAVE TOP 3 KEY EVIDENCE FRAMES
############################################################
def save_key_frames(
    frames_data: list,          # list of (frame_number, frame_image, fake_score)
    output_dir: str = None
) -> Dict[str, List[Dict]]:
    """
    Saves top 6 key frames split into two categories:
    - most_fake: 3 frames with HIGHEST fake score (most suspicious)
    - most_real: 3 frames with LOWEST fake score (most authentic)

    Returns dict with two lists: 
    {
        "most_fake": [{"path": str, "frame_number": int, "fake_score": float, "real_score": float}, ...],
        "most_real": [{"path": str, "frame_number": int, "fake_score": float, "real_score": float}, ...]
    }
    """
    if not frames_data:
        return {"most_fake": [], "most_real": []}

    # Save to temp directory
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="deeptrace_frames_")
    else:
        os.makedirs(output_dir, exist_ok=True)

    # Sort by fake_score - highest first
    sorted_by_fake = sorted(frames_data, key=lambda x: x[2], reverse=True)
    top3_fake = sorted_by_fake[:3]
    
    # Sort by fake_score - lowest first
    sorted_by_real = sorted(frames_data, key=lambda x: x[2], reverse=False)
    top3_real = sorted_by_real[:3]

    result = {"most_fake": [], "most_real": []}

    # Save most fake frames (red annotations)
    for idx, (frame_num, frame_img, fake_score) in enumerate(top3_fake):
        filename = f"most_fake_{idx+1}_frame{frame_num}.jpg"
        path = os.path.join(output_dir, filename)

        frame_copy = frame_img.copy()
        real_score = round(100 - fake_score, 1)
        score_text = f"Fake: {fake_score:.1f}%  Real: {real_score:.1f}%"
        cv2.putText(
            frame_copy, score_text,
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 0, 255),  # Red for suspicious
            2, cv2.LINE_AA
        )

        cv2.imwrite(path, frame_copy)
        result["most_fake"].append({
            "path": path,
            "frame_number": frame_num,
            "fake_score": round(fake_score, 2),
            "real_score": round(real_score, 2)
        })

    # Save most real frames (green annotations)
    for idx, (frame_num, frame_img, fake_score) in enumerate(top3_real):
        filename = f"most_real_{idx+1}_frame{frame_num}.jpg"
        path = os.path.join(output_dir, filename)

        frame_copy = frame_img.copy()
        real_score = round(100 - fake_score, 1)
        score_text = f"Fake: {fake_score:.1f}%  Real: {real_score:.1f}%"
        cv2.putText(
            frame_copy, score_text,
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 200, 0),  # Green for authentic
            2, cv2.LINE_AA
        )

        cv2.imwrite(path, frame_copy)
        result["most_real"].append({
            "path": path,
            "frame_number": frame_num,
            "fake_score": round(fake_score, 2),
            "real_score": round(real_score, 2)
        })

    return result


############################################################
#  ANALYZE VIDEO FOR DEEPFAKES
############################################################
def analyze_video(video_path: str, frame_interval: int = 5) -> Dict:
    """
    Analyzes a video for deepfake detection.
    Now returns key_frames as a dict with two lists: most_fake and most_real.
    
    Logic: Counts frames with fake score > 85%. If more than 60% of frames are highly fake, 
    the entire video is marked as FAKE. This approach is robust against occasional bad frames.
    Overall confidence is based on the ratio of high-fake frames.
    
    Returns:
        {
            "overall_verdict": "FAKE" or "REAL",
            "overall_confidence": float (based on max fake score),
            "max_fake_score": float (highest fake score across all frames),
            "average_fake_score": float (mean fake score across all frames),
            "frame_results": [...],
            "total_frames_analyzed": int,
            "faces_detected": int,
            "key_frames": {
                "most_fake": [{"path": str, "frame_number": int, "fake_score": float, "real_score": float}, ...],
                "most_real": [{"path": str, "frame_number": int, "fake_score": float, "real_score": float}, ...]
            }
        }
    """
    print(f"📹 Analyzing video: {video_path}")
    
    # Extract frames
    print(f"🎞️  Extracting every {frame_interval}th frame...")
    frames = extract_frames(video_path, frame_interval)
    print(f"✅ Extracted {len(frames)} frames")
    
    frame_results = []
    fake_scores = []
    noise_scores = []
    faces_detected_count = 0
    
    # Store raw frame data for key frame saving: (frame_number, frame_image, fake_score)
    frames_with_faces = []
    
    for frame_num, frame in frames:
        # Detect faces in frame
        faces = detect_faces(frame)
        
        if len(faces) == 0:
            # No face detected, skip or use whole frame
            frame_results.append({
                "frame_number": frame_num,
                "face_detected": False,
                "prediction": "N/A",
                "confidence": 0.0,
                "fake_score": 0.0
            })
            continue
        
        # Use the first (largest) face detected
        faces_detected_count += 1
        x, y, w, h = faces[0]
        
        # Crop face region
        face_crop = frame[y:y+h, x:x+w]
        
        # Convert BGR (OpenCV) to RGB (PIL)
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb)
        
        # Run model prediction
        inputs = preprocess(face_pil)
        
        with torch.no_grad():
            if USE_TRANSFORMERS:
                outputs = model(**inputs)
                logits = outputs.logits
            else:
                logits = model(inputs["pixel_values"])
        
        probs = torch.softmax(logits, dim=1)[0]
        
        # Get labels
        if USE_TRANSFORMERS:
            id2label = model.config.id2label
        else:
            id2label = {0: "REAL", 1: "FAKE"}
        
        # Calculate scores
        scores = {id2label[i]: float(probs[i]) for i in range(len(probs))}
        
        # Normalize labels to uppercase
        fake_score = scores.get("Fake", scores.get("FAKE", 0.0))
        real_score = scores.get("Real", scores.get("REAL", 0.0))
        
        predicted_label = "FAKE" if fake_score > real_score else "REAL"
        confidence = max(fake_score, real_score) * 100
        
        frame_results.append({
            "frame_number": frame_num,
            "face_detected": True,
            "prediction": predicted_label,
            "confidence": round(confidence, 2),
            "fake_score": round(fake_score * 100, 2)
        })
        
        # Store full frame (not just face crop) for key frame saving
        frames_with_faces.append((frame_num, frame, fake_score * 100))
        
        # Calculate noise level using Laplacian variance
        gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        noise_score = cv2.Laplacian(gray_face, cv2.CV_64F).var()
        noise_scores.append(round(float(noise_score), 2))
        
        fake_scores.append(fake_score * 100)
    
    # Calculate overall verdict
    if len(fake_scores) == 0:
        return {
            "overall_verdict": "UNKNOWN",
            "overall_confidence": 0.0,
            "max_fake_score": 0.0,
            "average_fake_score": 0.0,
            "frame_results": frame_results,
            "total_frames_analyzed": len(frames),
            "faces_detected": 0,
            "key_frames": {"most_fake": [], "most_real": []},
            "error": "No faces detected in any frame"
        }
    
    # Calculate both max and average fake scores for analysis
    max_fake_score = np.max(fake_scores)
    average_fake_score = np.mean(fake_scores)
    
    # Count frames with high fake scores (>85%) and calculate ratio
    high_fake_frames = sum(1 for s in fake_scores if s > 85)
    fake_frame_ratio = high_fake_frames / len(fake_scores)
    overall_verdict = "FAKE" if fake_frame_ratio > 0.6 else "REAL"
    overall_confidence = fake_frame_ratio * 100 if overall_verdict == "FAKE" else (1 - fake_frame_ratio) * 100
    
    # ✅ Save top 3 key frames based on verdict
    key_frames = save_key_frames(frames_with_faces)
    
    print(f"✅ Analysis complete!")
    print(f"   Overall Verdict: {overall_verdict}")
    print(f"   Confidence: {overall_confidence:.2f}%")
    print(f"   Max Fake Score: {max_fake_score:.2f}%")
    print(f"   Average Fake Score: {average_fake_score:.2f}%")
    print(f"   Frames Analyzed: {len(frames)}")
    print(f"   Faces Detected: {faces_detected_count}")
    print(f"   Key Frames Saved: {len(key_frames['most_fake'])} fake + {len(key_frames['most_real'])} real")
    
    return {
        "overall_verdict": overall_verdict,
        "overall_confidence": round(overall_confidence, 2),
        "max_fake_score": round(max_fake_score, 2),
        "average_fake_score": round(average_fake_score, 2),
        "frame_results": frame_results,
        "total_frames_analyzed": len(frames),
        "faces_detected": faces_detected_count,
        "noise_scores": noise_scores,
        "frame_numbers": [r["frame_number"] for r in frame_results if r["face_detected"]],
        "key_frames": key_frames   # ← NEW: dict with "most_fake" and "most_real" lists
    }


############################################################
#  QUICK TEST
############################################################
if __name__ == "__main__":
    # Test single frame
    test_image = Path("test_videos/deepfake_images_1.png")

    if test_image.exists():
        print("\n" + "="*60)
        print("SINGLE FRAME TEST")
        print("="*60)
        result = predict_frame(str(test_image))
        print("Label:", result["label"])
        print("Confidence:", result["confidence"], "%")
        print("All Scores:", result["all_scores"])
    else:
        print(f"⚠ Test image not found at {test_image}")
    
    # Test video analysis
    print("\n" + "="*60)
    print("VIDEO ANALYSIS TEST")
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
            
            result = analyze_video(str(test_video), frame_interval=5)
            
            print("\n📊 RESULTS:")
            print(f"Overall Verdict: {result['overall_verdict']}")
            print(f"Confidence: {result['overall_confidence']}%")
            print(f"Most Fake Frames: {[kf['path'] for kf in result['key_frames']['most_fake']]}")
            print(f"Most Real Frames: {[kf['path'] for kf in result['key_frames']['most_real']]}")
        else:
            print("⚠ No video files found in test_videos/")
            print("   Supported formats: .mp4, .avi, .mov")
    else:
        print(f"⚠ Test video directory not found: {test_video_dir}")
