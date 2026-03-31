# DeepTrace - Deepfake Forensics Tool

DeepTrace is a multi-modal deepfake forensics tool with:
- Visual deepfake detection on video frames
- Audio-visual lip sync analysis
- Grad-CAM heatmaps for suspicious frames
- PDF report generation
- Streamlit frontend and FastAPI backend

## Project Structure
- backend/ - API and analysis pipeline
- frontend/ - Streamlit UI
- models/ - Model weights and configs
- outputs/ - Generated reports, uploads, and heatmaps
- download_fakeavceleb_subset.py - Helper to download a small FakeAVCeleb subset

## Requirements
Python 3.10+ recommended.

Install dependencies:
```
pip install -r requirements.txt
```

## Run Backend
From the project root:
```
uvicorn backend.main:app --reload
```

## Run Frontend
From the project root:
```
cd frontend
streamlit run app.py
```

## Model Weights
Ensure model weights exist under:
```
models/weights/deepfake_model
```

## Outputs
Generated files are stored under:
```
outputs/
```

## FakeAVCeleb Subset Downloader
To download ~30 sample videos from FakeAVCeleb (Google Drive links):
```
python download_fakeavceleb_subset.py --count 30 --output dataset_videos
```

## Notes
- test_videos/ and venv/ are ignored by git.
- The audio sync stage requires ffmpeg in PATH.
