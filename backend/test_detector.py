# backend/test_detector.py

from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from pathlib import Path
from PIL import Image
import torch
import numpy as np
import cv2

# ── Load Model ───────────────────────────────────────────────
MODEL_PATH = str(Path("models/weights/deepfake_model"))
print("Loading model... please wait")
extractor = AutoFeatureExtractor.from_pretrained(MODEL_PATH)
model = AutoModelForImageClassification.from_pretrained(MODEL_PATH)
model.eval()
print("Model ready ✅\n")


# ── Single Image Prediction ──────────────────────────────────
def predict_image(image_path: str) -> dict:
    image = Image.open(image_path).convert("RGB")
    inputs = extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]
    id2label = model.config.id2label
    scores = {id2label[i]: round(probs[i].item() * 100, 2) for i in range(len(probs))}
    predicted_label = max(scores, key=scores.get)
    return {"label": predicted_label, "confidence": scores[predicted_label], "all_scores": scores}


# ── Video Prediction ─────────────────────────────────────────
def predict_video(video_path: str, sample_every=10) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": f"Cannot open video: {video_path}"}

    frame_count = 0
    analyzed = 0
    fake_scores = []
    results = []

    print(f"  Analyzing video: {Path(video_path).name}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Only analyze every Nth frame
        if frame_count % sample_every != 0:
            continue

        # Convert OpenCV frame (BGR) to PIL (RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        inputs = extractor(images=pil_image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0]

        id2label = model.config.id2label
        scores = {id2label[i]: round(probs[i].item() * 100, 2) for i in range(len(probs))}
        predicted = max(scores, key=scores.get)

        for key, val in scores.items():
            if "fake" in key.lower():
                fake_scores.append(val)

        results.append({"frame": frame_count, "label": predicted, "scores": scores})
        analyzed += 1
        print(f"    Frame {frame_count}: {predicted} ({scores})")

    cap.release()

    if not fake_scores:
        return {"error": "No frames analyzed"}

    overall = round(np.mean(fake_scores), 2)
    verdict = "🔴 MANIPULATED" if overall >= 50 else "🟢 AUTHENTIC"

    return {
        "video": Path(video_path).name,
        "verdict": verdict,
        "overall_fake_confidence": overall,
        "frames_analyzed": analyzed,
        "total_frames": frame_count
    }


# ══════════════════════════════════════════════════════════════
# RUN ALL TESTS
# ══════════════════════════════════════════════════════════════

print("=" * 55)
print("TEST 1 — Single Frame (example_frame.jpg)")
print("=" * 55)
r = predict_image("test_videos/example_frame.jpg")
print(f"Label:      {r['label']}")
print(f"Confidence: {r['confidence']}%")
print(f"All Scores: {r['all_scores']}\n")


print("=" * 55)
print("TEST 2 — Deepfake Image 1 (deepfake_images_1.png)")
print("=" * 55)
r = predict_image("test_videos/deepfake_images_1.png")
print(f"Label:      {r['label']}")
print(f"Confidence: {r['confidence']}%")
print(f"All Scores: {r['all_scores']}\n")


print("=" * 55)
print("TEST 3 — Deepfake Image 2 (deepfake_images_2.png)")
print("=" * 55)
r = predict_image("test_videos/deepfake_images_2.png")
print(f"Label:      {r['label']}")
print(f"Confidence: {r['confidence']}%")
print(f"All Scores: {r['all_scores']}\n")


print("=" * 55)
print("TEST 4 — Original Video (fadg0-original.mov)")
print("=" * 55)
r = predict_video("test_videos/fadg0-original.mov")
print(f"\nVerdict:          {r.get('verdict')}")
print(f"Fake Confidence:  {r.get('overall_fake_confidence')}%")
print(f"Frames Analyzed:  {r.get('frames_analyzed')} / {r.get('total_frames')}\n")


print("=" * 55)
print("TEST 5 — Manipulated Video (fadg0-fram1-roi93.mov)")
print("=" * 55)
r = predict_video("test_videos/fadg0-fram1-roi93.mov")
print(f"\nVerdict:          {r.get('verdict')}")
print(f"Fake Confidence:  {r.get('overall_fake_confidence')}%")
print(f"Frames Analyzed:  {r.get('frames_analyzed')} / {r.get('total_frames')}\n")

print("=" * 55)
print("ALL TESTS COMPLETE ✅")
print("=" * 55)