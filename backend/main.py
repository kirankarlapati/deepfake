# backend/main.py

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
import os
from datetime import datetime
from typing import Dict

# Import our backend modules
from backend.detector import analyze_video
from backend.audio_sync import analyze_audio_sync
from backend.explainer import explain_analysis
from backend.report import generate_report


############################################################
#  FASTAPI APP INITIALIZATION
############################################################
app = FastAPI(
    title="DeepTrace API",
    description="AI-Powered Deepfake Forensics Tool",
    version="1.0.0"
)

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (restrict in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


############################################################
#  DIRECTORIES
############################################################
UPLOAD_DIR = Path("outputs/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


############################################################
#  HEALTH CHECK ENDPOINT
############################################################
@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify server is running.
    
    Returns:
        Status message and server info
    """
    return {
        "status": "healthy",
        "service": "DeepTrace API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }


############################################################
#  VIDEO ANALYSIS ENDPOINT
############################################################
@app.post("/analyze")
async def analyze_video_endpoint(file: UploadFile = File(...)):
    """
    Main analysis endpoint that accepts video upload and performs full deepfake analysis.
    
    Process:
    1. Save uploaded video to outputs/uploads/
    2. Run visual analysis (detector.py)
    3. Run audio-visual sync analysis (audio_sync.py)
    4. Generate AI explanation (explainer.py)
    5. Generate PDF report (report.py)
    6. Return comprehensive results
    
    Args:
        file: Video file upload (mp4, avi, mov formats)
    
    Returns:
        JSON with all analysis results and PDF path
    """
    print("\n" + "="*60)
    print(f"🎬 NEW ANALYSIS REQUEST")
    print("="*60)
    
    # Validate file type
    allowed_extensions = [".mp4", ".avi", ".mov", ".MP4", ".AVI", ".MOV"]
    file_extension = Path(file.filename).suffix
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Supported formats: {', '.join(allowed_extensions)}"
        )
    
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    original_name = Path(file.filename).stem
    video_filename = f"{original_name}_{timestamp}{file_extension}"
    video_path = UPLOAD_DIR / video_filename
    
    try:
        # Save uploaded video
        print(f"💾 Saving uploaded video: {video_filename}")
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"✅ Video saved to: {video_path}")
        
        # Step 1: Visual Analysis (detector.py)
        print("\n" + "-"*60)
        print("STEP 1: VISUAL DEEPFAKE DETECTION")
        print("-"*60)
        
        visual_results = analyze_video(str(video_path), frame_interval=5)
        
        if "error" in visual_results:
            raise HTTPException(
                status_code=500,
                detail=f"Visual analysis failed: {visual_results['error']}"
            )
        
        # Step 2: Audio-Visual Sync Analysis (audio_sync.py)
        print("\n" + "-"*60)
        print("STEP 2: AUDIO-VISUAL SYNCHRONIZATION ANALYSIS")
        print("-"*60)
        
        audio_results = analyze_audio_sync(str(video_path), threshold=0.3)
        
        if not audio_results.get("success", False):
            # Audio analysis failed, use fallback
            print("⚠ Audio analysis failed, using fallback data")
            audio_results = {
                "success": False,
                "verdict": "ERROR",
                "sync_score": 0.0,
                "correlation": 0.0,
                "total_mismatches": 0,
                "mismatch_percentage": 0.0,
                "mismatch_timestamps": [],
                "total_frames": visual_results.get("total_frames_analyzed", 0),
                "error": audio_results.get("error", "Audio analysis unavailable")
            }
        
        # Combined verdict using both visual and audio signals
        visual_verdict = visual_results.get("overall_verdict", "UNKNOWN")
        audio_mismatches = audio_results.get("total_mismatches", 0)
        audio_sync_score = audio_results.get("sync_score", 100)
        average_fake_score = visual_results.get("average_fake_score", 0)

        # Audio is a strong signal: >20 mismatches or sync_score < 20 = suspicious
        audio_says_fake = audio_mismatches > 20 or audio_sync_score < 20

        # Visual is reliable only when average is very high (>88%)
        visual_says_fake = average_fake_score > 88

        # Final decision: both must agree OR audio alone is very strong (>35 mismatches)
        if visual_says_fake and audio_says_fake:
            final_verdict = "FAKE"
        elif audio_mismatches > 35:
            final_verdict = "FAKE"
        else:
            final_verdict = "REAL"
        
        # Step 3: Generate AI Explanation (explainer.py)
        print("\n" + "-"*60)
        print("STEP 3: AI FORENSICS EXPLANATION")
        print("-"*60)
        
        explanation_result = explain_analysis(visual_results, audio_results)
        
        explanation_text = explanation_result.get("explanation", "Analysis complete.")
        
        # Step 4: Generate PDF Report (report.py)
        print("\n" + "-"*60)
        print("STEP 4: PDF REPORT GENERATION")
        print("-"*60)
        
        pdf_path = generate_report(
            video_path=str(video_path),
            visual_results=visual_results,
            audio_results=audio_results,
            explanation=explanation_text,
            final_verdict=final_verdict
        )
        
        # Prepare response
        print("\n" + "="*60)
        print("✅ ANALYSIS COMPLETE")
        print("="*60)
        
        response = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "video_filename": video_filename,
            
            # Visual Analysis Results
            "visual_analysis": {
                "overall_verdict": final_verdict,
                "overall_confidence": visual_results.get("overall_confidence"),
                "max_fake_score": visual_results.get("max_fake_score"),
                "average_fake_score": visual_results.get("average_fake_score"),
                "total_frames_analyzed": visual_results.get("total_frames_analyzed"),
                "faces_detected": visual_results.get("faces_detected"),
                "frame_results": visual_results.get("frame_results", [])[:10]  # Return first 10 frames
            },
            
            # Audio Sync Results
            "audio_sync_analysis": {
                "verdict": audio_results.get("verdict"),
                "sync_score": audio_results.get("sync_score"),
                "correlation": audio_results.get("correlation"),
                "total_mismatches": audio_results.get("total_mismatches"),
                "mismatch_percentage": audio_results.get("mismatch_percentage"),
                "total_frames": audio_results.get("total_frames"),
                "mismatch_timestamps": audio_results.get("mismatch_timestamps", [])[:10]  # Return first 10
            },
            
            # AI Explanation
            "explanation": {
                "text": explanation_text,
                "model": explanation_result.get("model", "fallback"),
                "success": explanation_result.get("success", True)
            },
            
            # PDF Report
            "report": {
                "pdf_path": pdf_path,
                "pdf_filename": Path(pdf_path).name,
                "download_url": f"/download/{Path(pdf_path).name}"
            }
        }
        
        print(f"\n📊 Overall Verdict: {final_verdict}")
        print(f"📊 Visual Confidence: {visual_results.get('overall_confidence')}%")
        print(f"📊 Audio Sync Score: {audio_results.get('sync_score')}%")
        print(f"📄 PDF Report: {pdf_path}")
        
        return JSONResponse(content=response)
    
    except HTTPException as he:
        # Re-raise HTTP exceptions
        raise he
    
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )
    
    finally:
        # Clean up if needed (optional - comment out to keep videos)
        # if video_path.exists():
        #     video_path.unlink()
        pass


############################################################
#  DOWNLOAD PDF ENDPOINT
############################################################
@app.get("/download/{filename}")
async def download_pdf(filename: str):
    """
    Download generated PDF report.
    
    Args:
        filename: Name of the PDF file to download
    
    Returns:
        PDF file as downloadable response
    """
    pdf_path = OUTPUT_DIR / filename
    
    if not pdf_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"PDF report not found: {filename}"
        )
    
    if not pdf_path.suffix == ".pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files can be downloaded"
        )
    
    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=filename,
        headers={
            "Content-Disposition": f"attachment; filename={filename}"
        }
    )


############################################################
#  LIST REPORTS ENDPOINT (BONUS)
############################################################
@app.get("/reports")
async def list_reports():
    """
    List all generated PDF reports.
    
    Returns:
        List of available PDF reports with metadata
    """
    reports = []
    
    for pdf_file in OUTPUT_DIR.glob("*.pdf"):
        file_stats = pdf_file.stat()
        reports.append({
            "filename": pdf_file.name,
            "size_bytes": file_stats.st_size,
            "size_mb": round(file_stats.st_size / (1024 * 1024), 2),
            "created": datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
            "download_url": f"/download/{pdf_file.name}"
        })
    
    # Sort by creation time (newest first)
    reports.sort(key=lambda x: x["created"], reverse=True)
    
    return {
        "total_reports": len(reports),
        "reports": reports
    }


############################################################
#  DELETE REPORT ENDPOINT (BONUS)
############################################################
@app.delete("/reports/{filename}")
async def delete_report(filename: str):
    """
    Delete a specific PDF report.
    
    Args:
        filename: Name of the PDF file to delete
    
    Returns:
        Success message
    """
    pdf_path = OUTPUT_DIR / filename
    
    if not pdf_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"PDF report not found: {filename}"
        )
    
    if not pdf_path.suffix == ".pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files can be deleted"
        )
    
    try:
        pdf_path.unlink()
        return {
            "success": True,
            "message": f"Report deleted: {filename}"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete report: {str(e)}"
        )


############################################################
#  ROOT ENDPOINT
############################################################
@app.get("/")
async def root():
    """
    Root endpoint with API information.
    
    Returns:
        Welcome message and available endpoints
    """
    return {
        "service": "DeepTrace API",
        "description": "AI-Powered Deepfake Forensics Tool",
        "version": "1.0.0",
        "endpoints": {
            "POST /analyze": "Upload and analyze video for deepfakes",
            "GET /download/{filename}": "Download generated PDF report",
            "GET /reports": "List all generated reports",
            "DELETE /reports/{filename}": "Delete a specific report",
            "GET /health": "Health check endpoint"
        },
        "documentation": "/docs"
    }


############################################################
#  RUN SERVER
############################################################
if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("🚀 STARTING DEEPTRACE API SERVER")
    print("="*60)
    print("\n📍 Server will be available at:")
    print("   http://localhost:8000")
    print("\n📚 API Documentation:")
    print("   http://localhost:8000/docs")
    print("\n⚡ Press CTRL+C to stop the server\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
