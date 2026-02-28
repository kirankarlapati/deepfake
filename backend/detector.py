# backend/detector.py

from pathlib import Path
from PIL import Image
import torch
import numpy as np
import cv2
from typing import List, Dict, Tuple

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
#  ANALYZE VIDEO FOR DEEPFAKES
############################################################
def analyze_video(video_path: str, frame_interval: int = 5) -> Dict:
    """
    Analyzes a video for deepfake detection.
    
    Logic: Counts frames with fake score > 85%. If more than 60% of frames are highly fake, 
    the entire video is marked as FAKE. This approach is robust against occasional bad frames.
    Overall confidence is based on the ratio of high-fake frames.
    
    Returns:
        {
            "overall_verdict": "FAKE" or "REAL",
            "overall_confidence": float (based on max fake score),
            "max_fake_score": float (highest fake score across all frames),
            "average_fake_score": float (mean fake score across all frames),
            "frame_results": [
                {
                    "frame_number": int,
                    "face_detected": bool,
                    "prediction": str,
                    "confidence": float,
                    "fake_score": float
                },
                ...
            ],
            "total_frames_analyzed": int,
            "faces_detected": int
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
            "error": "No faces detected in any frame"
        }
    
    # Calculate both max and average fake scores for analysis
    max_fake_score = np.max(fake_scores)
    average_fake_score = np.mean(fake_scores)
    
    # Count frames with high fake scores (>85%) and calculate ratio
    high_fake_frames = sum(1 for s in fake_scores if s > 85)
    fake_frame_ratio = high_fake_frames / len(fake_scores)
    overall_verdict = "FAKE" if fake_frame_ratio > 0.6 else "REAL"
    overall_confidence = fake_frame_ratio * 100 if overall_verdict == "FAKE" else (1 - fake_frame_ratio) * 10
    
    print(f"✅ Analysis complete!")
    print(f"   Overall Verdict: {overall_verdict}")
    print(f"   Confidence: {overall_confidence:.2f}%")
    print(f"   Max Fake Score: {max_fake_score:.2f}%")
    print(f"   Average Fake Score: {average_fake_score:.2f}%")
    print(f"   Frames Analyzed: {len(frames)}")
    print(f"   Faces Detected: {faces_detected_count}")
    
    return {
        "overall_verdict": overall_verdict,
        "overall_confidence": round(overall_confidence, 2),
        "max_fake_score": round(max_fake_score, 2),
        "average_fake_score": round(average_fake_score, 2),
        "frame_results": frame_results,
        "total_frames_analyzed": len(frames),
        "faces_detected": faces_detected_count,
        "noise_scores": noise_scores,
        "frame_numbers": [r["frame_number"] for r in frame_results if r["face_detected"]]
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
            print(f"Max Fake Score: {result['max_fake_score']}%")
            print(f"Average Fake Score: {result['average_fake_score']}%")
            print(f"Frames Analyzed: {result['total_frames_analyzed']}")
            print(f"Faces Detected: {result['faces_detected']}")
            print(f"\nFirst 3 frame results:")
            for fr in result['frame_results'][:3]:
                print(f"  Frame {fr['frame_number']}: {fr['prediction']} ({fr['confidence']}%)")
        else:
            print("⚠ No video files found in test_videos/")
            print("   Supported formats: .mp4, .avi, .mov")
    else:
        print(f"⚠ Test video directory not found: {test_video_dir}")
