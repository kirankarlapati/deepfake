# backend/detector.py

from pathlib import Path
from PIL import Image
import torch
import numpy as np

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
#  QUICK TEST
############################################################
if __name__ == "__main__":
    test_image = Path("test_videos/deepfake_images_1.png")

    if test_image.exists():
        print("\nRunning single frame test...")
        result = predict_frame(str(test_image))
        print("Label:", result["label"])
        print("Confidence:", result["confidence"], "%")
        print("All Scores:", result["all_scores"])
    else:
        print(f"⚠ Test image not found at {test_image}")