# backend/explainer.py

import requests
import json
from typing import Dict, Optional


############################################################
#  OLLAMA CONFIGURATION
############################################################
OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "llama3.2"


############################################################
#  CHECK OLLAMA STATUS
############################################################
def check_ollama_status() -> bool:
    """
    Checks if Ollama server is running and accessible.
    
    Returns:
        True if Ollama is running, False otherwise
    """
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        return response.status_code == 200
    except Exception:
        return False


############################################################
#  GENERATE FORENSICS EXPLANATION
############################################################
def generate_explanation(
    visual_score: float,
    visual_verdict: str,
    audio_sync_score: float,
    audio_sync_verdict: str,
    max_fake_score: float = None,
    average_fake_score: float = None,
    sync_correlation: float = None,
    total_mismatches: int = None
) -> Dict:
    """
    Generates a plain English forensics explanation using Ollama llama3.2.
    
    Args:
        visual_score: Overall visual deepfake confidence (0-100)
        visual_verdict: Visual analysis verdict ("FAKE" or "REAL")
        audio_sync_score: Audio-visual sync score (0-100)
        audio_sync_verdict: Audio sync verdict ("SUSPICIOUS", "QUESTIONABLE", "SYNCED")
        max_fake_score: Maximum fake score across frames (optional)
        average_fake_score: Average fake score (optional)
        sync_correlation: Audio-visual correlation coefficient (optional)
        total_mismatches: Number of sync mismatches found (optional)
    
    Returns:
        {
            "success": bool,
            "explanation": str (3 sentences),
            "error": str (if failed)
        }
    """
    print(f"🤖 Generating AI explanation using Ollama {MODEL_NAME}...")
    
    # Check if Ollama is running
    if not check_ollama_status():
        return {
            "success": False,
            "explanation": "AI explanation service is unavailable. Please ensure Ollama is running locally.",
            "error": "Ollama server not reachable at localhost:11434"
        }
    
    # Build the prompt with forensics context
    prompt = f"""You are a digital forensics expert analyzing a video for deepfake detection. 

Analysis Results:
- Visual Analysis: {visual_verdict} with {visual_score:.1f}% confidence
- Maximum fake score detected: {max_fake_score:.1f}% (threshold: 70%)
- Average fake score: {average_fake_score:.1f}%
- Audio-Visual Sync: {audio_sync_verdict} with {audio_sync_score:.1f}% sync score
- Sync correlation: {sync_correlation:.3f}
- Audio-visual mismatches found: {total_mismatches} frames

Task: Provide a forensics explanation in EXACTLY 3 sentences that:
1. First sentence: Summarize the overall finding (fake or real) based on the evidence
2. Second sentence: Explain the key visual and audio indicators found
3. Third sentence: State the final conclusion with confidence level

Be technical but clear. Use forensics terminology. Do not use bullet points or numbering. Write in a professional, authoritative tone."""

    try:
        # Call Ollama API
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_predict": 200  # Limit response length
                }
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            explanation = result.get("response", "").strip()
            
            # Clean up the explanation
            explanation = explanation.replace("\n\n", " ").replace("\n", " ")
            
            # Ensure it's roughly 3 sentences (split and rejoin if needed)
            sentences = [s.strip() for s in explanation.split('.') if s.strip()]
            
            # Take first 3 sentences if more than 3
            if len(sentences) > 3:
                explanation = '. '.join(sentences[:3]) + '.'
            elif len(sentences) < 3:
                # If less than 3, use as is
                explanation = '. '.join(sentences) + '.'
            else:
                explanation = '. '.join(sentences) + '.'
            
            print(f"✅ Explanation generated successfully")
            
            return {
                "success": True,
                "explanation": explanation,
                "model": MODEL_NAME
            }
        else:
            error_msg = f"Ollama API returned status {response.status_code}"
            print(f"❌ {error_msg}")
            return {
                "success": False,
                "explanation": generate_fallback_explanation(
                    visual_score, visual_verdict, audio_sync_score, audio_sync_verdict
                ),
                "error": error_msg
            }
            
    except requests.exceptions.Timeout:
        print("❌ Ollama request timed out")
        return {
            "success": False,
            "explanation": generate_fallback_explanation(
                visual_score, visual_verdict, audio_sync_score, audio_sync_verdict
            ),
            "error": "Request timed out after 30 seconds"
        }
    except Exception as e:
        print(f"❌ Error calling Ollama: {str(e)}")
        return {
            "success": False,
            "explanation": generate_fallback_explanation(
                visual_score, visual_verdict, audio_sync_score, audio_sync_verdict
            ),
            "error": str(e)
        }


############################################################
#  FALLBACK EXPLANATION (IF OLLAMA UNAVAILABLE)
############################################################
def generate_fallback_explanation(
    visual_score: float,
    visual_verdict: str,
    audio_sync_score: float,
    audio_sync_verdict: str
) -> str:
    """
    Generates a rule-based explanation if Ollama is unavailable.
    
    Returns:
        A 3-sentence forensics explanation
    """
    # Sentence 1: Overall finding
    if visual_verdict == "FAKE":
        sentence1 = f"The video has been identified as a deepfake with {visual_score:.1f}% confidence based on visual forensics analysis."
    else:
        sentence1 = f"The video appears authentic with {visual_score:.1f}% confidence based on visual analysis."
    
    # Sentence 2: Technical details
    if audio_sync_verdict == "SUSPICIOUS":
        sentence2 = f"Significant audio-visual desynchronization was detected (sync score: {audio_sync_score:.1f}%), indicating potential manipulation of the audio or video track."
    elif audio_sync_verdict == "QUESTIONABLE":
        sentence2 = f"Moderate audio-visual inconsistencies were observed (sync score: {audio_sync_score:.1f}%), suggesting possible editing artifacts."
    else:
        sentence2 = f"Audio-visual synchronization is consistent (sync score: {audio_sync_score:.1f}%), with lip movements properly aligned to speech patterns."
    
    # Sentence 3: Conclusion
    if visual_verdict == "FAKE" or audio_sync_verdict == "SUSPICIOUS":
        sentence3 = "The evidence strongly suggests this video has been synthetically generated or manipulated using deepfake technology."
    elif audio_sync_verdict == "QUESTIONABLE":
        sentence3 = "While some anomalies were detected, additional forensic examination may be required for conclusive determination."
    else:
        sentence3 = "Based on the comprehensive multi-modal analysis, this video shows no significant indicators of deepfake manipulation."
    
    return f"{sentence1} {sentence2} {sentence3}"


############################################################
#  COMBINED ANALYSIS EXPLANATION
############################################################
def explain_analysis(
    visual_results: Dict,
    audio_sync_results: Dict
) -> Dict:
    """
    Takes results from detector.py and audio_sync.py and generates explanation.
    
    Args:
        visual_results: Output from detector.analyze_video()
        audio_sync_results: Output from audio_sync.analyze_audio_sync()
    
    Returns:
        {
            "success": bool,
            "explanation": str,
            "model": str,
            "error": str (if applicable)
        }
    """
    # Extract visual data
    visual_score = visual_results.get("overall_confidence", 0.0)
    visual_verdict = visual_results.get("overall_verdict", "UNKNOWN")
    max_fake_score = visual_results.get("max_fake_score", 0.0)
    average_fake_score = visual_results.get("average_fake_score", 0.0)
    
    # Extract audio sync data
    audio_sync_score = audio_sync_results.get("sync_score", 0.0)
    audio_sync_verdict = audio_sync_results.get("verdict", "UNKNOWN")
    sync_correlation = audio_sync_results.get("correlation", 0.0)
    total_mismatches = audio_sync_results.get("total_mismatches", 0)
    
    # Generate explanation
    result = generate_explanation(
        visual_score=visual_score,
        visual_verdict=visual_verdict,
        audio_sync_score=audio_sync_score,
        audio_sync_verdict=audio_sync_verdict,
        max_fake_score=max_fake_score,
        average_fake_score=average_fake_score,
        sync_correlation=sync_correlation,
        total_mismatches=total_mismatches
    )
    
    return result


############################################################
#  QUICK TEST
############################################################
if __name__ == "__main__":
    print("\n" + "="*60)
    print("OLLAMA EXPLAINER TEST")
    print("="*60)
    
    # Check Ollama status
    print(f"\n🔍 Checking Ollama status at {OLLAMA_BASE_URL}...")
    if check_ollama_status():
        print(f"✅ Ollama is running and accessible")
    else:
        print(f"❌ Ollama is not running. Start it with: ollama serve")
        print(f"   Then run: ollama pull {MODEL_NAME}")
    
    # Test with sample data
    print(f"\n📝 Testing explanation generation...")
    
    # Sample visual results (FAKE video)
    sample_visual_results = {
        "overall_verdict": "FAKE",
        "overall_confidence": 87.5,
        "max_fake_score": 92.3,
        "average_fake_score": 78.6,
        "total_frames_analyzed": 30,
        "faces_detected": 28
    }
    
    # Sample audio sync results (SUSPICIOUS)
    sample_audio_results = {
        "verdict": "SUSPICIOUS",
        "sync_score": 45.2,
        "correlation": 0.452,
        "total_mismatches": 15,
        "mismatch_percentage": 25.5
    }
    
    result = explain_analysis(sample_visual_results, sample_audio_results)
    
    print("\n📊 RESULTS:")
    print(f"Success: {result['success']}")
    if result['success']:
        print(f"Model: {result.get('model', 'fallback')}")
    print(f"\nExplanation:\n{result['explanation']}")
    
    if not result['success'] and 'error' in result:
        print(f"\n⚠ Error: {result['error']}")
    
    # Test with REAL video
    print("\n" + "="*60)
    print("Testing with REAL video scenario...")
    print("="*60)
    
    sample_visual_results_real = {
        "overall_verdict": "REAL",
        "overall_confidence": 82.3,
        "max_fake_score": 17.7,
        "average_fake_score": 12.4,
        "total_frames_analyzed": 40,
        "faces_detected": 40
    }
    
    sample_audio_results_synced = {
        "verdict": "SYNCED",
        "sync_score": 88.5,
        "correlation": 0.885,
        "total_mismatches": 2,
        "mismatch_percentage": 5.0
    }
    
    result2 = explain_analysis(sample_visual_results_real, sample_audio_results_synced)
    
    print(f"\nSuccess: {result2['success']}")
    print(f"\nExplanation:\n{result2['explanation']}")
