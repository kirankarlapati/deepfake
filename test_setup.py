print("🟩 DeepTrace Environment Check Running...\n")

modules = [
    "torch", "transformers", "cv2", "mediapipe", "librosa",
    "moviepy", "streamlit", "fastapi", "uvicorn",
    "reportlab", "PIL", "numpy", "scipy", "matplotlib",
    "requests", "huggingface_hub", "pytorch_grad_cam", "dlib"
]

for m in modules:
    try:
        __import__(m)
        print(f"✅ {m} imported successfully")
    except Exception as e:
        print(f"❌ {m} failed -> {e}")

print("\n🟦 Testing Ollama CLI...")
try:
    import subprocess
    result = subprocess.run(
        ["ollama", "--version"],
        capture_output=True,
        text=True
    )
    if result.stdout:
        print(f"✅ Ollama detected: {result.stdout.strip()}")
    else:
        print("⚠️ Ollama installed but no response")
except Exception as e:
    print(f"❌ Ollama error -> {e}")

print("\n🟢 All checks finished!")