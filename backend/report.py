# backend/report.py

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from datetime import datetime
from pathlib import Path
import os
from typing import Dict, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tempfile


############################################################
#  OUTPUT DIRECTORY
############################################################
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


############################################################
#  GENERATE SYNC GRAPH
############################################################
def generate_sync_graph(audio_signal, lip_signal, mismatch_timestamps):
    """Generate sync graph and return as temp image path."""
    timestamps = [i / 30 for i in range(len(audio_signal))]
    
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(timestamps, audio_signal, color="royalblue", label="Audio Energy", linewidth=1.5)
    ax.plot(timestamps, lip_signal, color="orange", label="Lip Movement", linewidth=1.5)
    
    for m in mismatch_timestamps:
        ax.axvline(x=m["timestamp"], color="red", alpha=0.3, linewidth=1)
    
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Normalized Value")
    ax.set_title("Audio Energy vs Lip Movement — Mismatch Analysis")
    ax.legend()
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    
    tmp = tempfile.mktemp(suffix=".png")
    plt.savefig(tmp, dpi=150)
    plt.close()
    return tmp


############################################################
#  GENERATE KEY EVIDENCE FRAMES SECTION
############################################################
def build_key_frames_section(story, key_frames: Dict[str, List[Dict]], styles, final_verdict: str = "UNKNOWN"):
    """
    Adds TWO 'Key Evidence Frames' sections to the PDF story:
    - Order depends on final_verdict:
      * FAKE: Most Suspicious first, then Most Authentic
      * REAL: Most Authentic first, then Most Suspicious
    
    Always shows both sections regardless of verdict.
    """
    heading_style = ParagraphStyle(
        'KeyFrameHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=8,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )

    caption_style = ParagraphStyle(
        'FrameCaption',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#555555'),
        alignment=TA_CENTER,
        fontName='Helvetica'
    )

    note_style = ParagraphStyle(
        'FrameNote',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#666666'),
        alignment=TA_LEFT,
        fontName='Helvetica-Oblique',
        spaceAfter=10
    )

    most_fake_frames = key_frames.get("most_fake", [])
    most_real_frames = key_frames.get("most_real", [])

    # Helper function to build a section
    def build_section(section_num: int, title: str, description: str, frames: List[Dict], border_color: str, score_type: str):
        story.append(Paragraph(f"Section {section_num}: {title}", heading_style))
        story.append(Paragraph(description, note_style))

        if not frames:
            story.append(Paragraph(f"No {title.lower()} available.", note_style))
        else:
            img_cells = []
            caption_cells = []

            for kf in frames:
                img_path = kf.get("path", "")
                frame_num = kf.get("frame_number", "?")
                score = kf.get("fake_score" if score_type == "fake" else "real_score", 0)

                if os.path.exists(img_path):
                    img = Image(img_path, width=1.8*inch, height=1.35*inch)
                    img_cells.append(img)
                else:
                    img_cells.append(Paragraph("Frame not found", caption_style))

                score_label = "Fake Score" if score_type == "fake" else "Real Score"
                caption_cells.append(
                    Paragraph(f"Frame #{frame_num}<br/>{score_label}: {score:.1f}%", caption_style)
                )

            # Pad to 3 if fewer frames
            while len(img_cells) < 3:
                img_cells.append(Paragraph("", caption_style))
                caption_cells.append(Paragraph("", caption_style))

            frame_table = Table(
                [img_cells, caption_cells],
                colWidths=[2.1*inch, 2.1*inch, 2.1*inch]
            )
            frame_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f8f9fa')),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dee2e6')),
                ('BOX', (0, 0), (-1, -1), 1, colors.HexColor(border_color)),
            ]))

            story.append(frame_table)

        story.append(Spacer(1, 0.3 * inch))

    # ============================================================
    # ORDER SECTIONS BASED ON VERDICT
    # ============================================================
    if final_verdict == "FAKE":
        # Show Most Suspicious FIRST, then Most Authentic
        build_section(
            1, "Most Suspicious Moments",
            "The frames below show the moments with the highest fake confidence scores, "
            "representing the strongest evidence of potential deepfake manipulation.",
            most_fake_frames, '#e74c3c', 'fake'
        )
        build_section(
            2, "Most Authentic Moments",
            "The frames below show the moments with the highest authenticity confidence, "
            "representing the strongest evidence that these portions of the video are genuine.",
            most_real_frames, '#27ae60', 'real'
        )
    else:
        # Show Most Authentic FIRST, then Most Suspicious (for REAL or UNKNOWN)
        build_section(
            1, "Most Authentic Moments",
            "The frames below show the moments with the highest authenticity confidence, "
            "representing the strongest evidence that these portions of the video are genuine.",
            most_real_frames, '#27ae60', 'real'
        )
        build_section(
            2, "Most Suspicious Moments",
            "The frames below show the moments with the highest fake confidence scores, "
            "representing the strongest evidence of potential deepfake manipulation.",
            most_fake_frames, '#e74c3c', 'fake'
        )


############################################################
#  GENERATE PDF REPORT
############################################################
def generate_report(
    video_path: str,
    visual_results: Dict,
    audio_results: Dict,
    explanation: str,
    output_filename: str = None,
    final_verdict: str = "UNKNOWN"
) -> str:
    """
    Generates a professional PDF forensics report using ReportLab.
    Now includes Key Evidence Frames section.
    
    Args:
        video_path: Path to analyzed video file
        visual_results: Results from detector.analyze_video()
        audio_results: Results from audio_sync.analyze_audio_sync()
        explanation: 3-sentence explanation from explainer.explain_analysis()
        output_filename: Optional custom output filename
        final_verdict: Final combined verdict (FAKE/REAL/UNKNOWN)
    
    Returns:
        Path to generated PDF file
    """
    print(f"📄 Generating PDF report...")
    
    # Generate output filename
    if output_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = Path(video_path).stem
        output_filename = f"DeepTrace_Report_{video_name}_{timestamp}.pdf"
    
    output_path = OUTPUT_DIR / output_filename
    
    # Create PDF document
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    # Container for flowable objects
    story = []
    graph_path = None  # Track temp graph file for cleanup
    
    # Styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=28,
        textColor=colors.HexColor('#1a1a1a'),
        spaceAfter=6,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Normal'],
        fontSize=12,
        textColor=colors.HexColor('#666666'),
        spaceAfter=20,
        alignment=TA_CENTER,
        fontName='Helvetica'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.HexColor('#333333'),
        spaceAfter=12,
        alignment=TA_JUSTIFY,
        fontName='Helvetica'
    )
    
    # Extract data
    overall_verdict = final_verdict  # Use the combined verdict passed from main.py
    visual_confidence = visual_results.get("overall_confidence", 0.0)
    max_fake_score = visual_results.get("max_fake_score", 0.0)
    average_fake_score = visual_results.get("average_fake_score", 0.0)
    total_frames = visual_results.get("total_frames_analyzed", 0)
    faces_detected = visual_results.get("faces_detected", 0)
    key_frames = visual_results.get("key_frames", {"most_fake": [], "most_real": []})
    
    audio_verdict = audio_results.get("verdict", "UNKNOWN")
    audio_sync_score = audio_results.get("sync_score", 0.0)
    audio_correlation = audio_results.get("correlation", 0.0)
    total_mismatches = audio_results.get("total_mismatches", 0)
    mismatch_timestamps = audio_results.get("mismatch_timestamps", [])
    
    video_filename = Path(video_path).name
    analysis_date = datetime.now().strftime("%B %d, %Y at %H:%M:%S")
    
    # ============================================================
    # HEADER: DeepTrace Logo/Title
    # ============================================================
    story.append(Paragraph("DeepTrace", title_style))
    story.append(Paragraph("AI-Powered Deepfake Forensics Analysis", subtitle_style))
    story.append(Spacer(1, 0.3 * inch))
    
    # ============================================================
    # VIDEO INFORMATION
    # ============================================================
    story.append(Paragraph("Video Information", heading_style))
    
    info_data = [
        ["Video File:", video_filename],
        ["Analysis Date:", analysis_date],
        ["Frames Analyzed:", f"{total_frames}"],
        ["Faces Detected:", f"{faces_detected}"]
    ]
    
    info_table = Table(info_data, colWidths=[2*inch, 4*inch])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ecf0f1')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#2c3e50')),
        ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#bdc3c7'))
    ]))
    
    story.append(info_table)
    story.append(Spacer(1, 0.4 * inch))
    
    # ============================================================
    # OVERALL VERDICT - BIG COLORED TEXT
    # ============================================================
    story.append(Paragraph("Overall Verdict", heading_style))
    
    # Determine color based on verdict
    if overall_verdict == "FAKE":
        verdict_color = colors.HexColor('#e74c3c')  # RED
        verdict_text = "DEEPFAKE DETECTED"
    elif overall_verdict == "REAL":
        verdict_color = colors.HexColor('#27ae60')  # GREEN
        verdict_text = "AUTHENTIC VIDEO"
    else:
        verdict_color = colors.HexColor('#f39c12')  # ORANGE
        verdict_text = "INCONCLUSIVE"
    
    verdict_style = ParagraphStyle(
        'VerdictStyle',
        parent=styles['Normal'],
        fontSize=24,
        textColor=verdict_color,
        spaceAfter=20,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    story.append(Paragraph(verdict_text, verdict_style))
    story.append(Spacer(1, 0.3 * inch))
    
    # ============================================================
    # KEY EVIDENCE FRAMES - Always show both sections
    # ============================================================
    build_key_frames_section(story, key_frames, styles, final_verdict)
    
    # ============================================================
    # ANALYSIS SCORES
    # ============================================================
    story.append(Paragraph("Analysis Scores", heading_style))
    
    scores_data = [
        ["Metric", "Score", "Status"],
        ["Visual Confidence", f"{visual_confidence:.2f}%", overall_verdict],
        ["Maximum Fake Score", f"{max_fake_score:.2f}%", "Threshold: 70%"],
        ["Audio-Visual Sync", f"{audio_sync_score:.2f}%", audio_verdict],
        ["Sync Correlation", f"{audio_correlation:.3f}", "Range: -1 to 1"],
        ["Audio Mismatches", f"{total_mismatches} frames", f"{audio_results.get('mismatch_percentage', 0):.1f}%"]
    ]
    
    scores_table = Table(scores_data, colWidths=[2.2*inch, 1.8*inch, 2*inch])
    scores_table.setStyle(TableStyle([
        # Header row
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('TOPPADDING', (0, 0), (-1, 0), 10),
        
        # Data rows
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor('#2c3e50')),
        ('ALIGN', (0, 1), (0, -1), 'LEFT'),
        ('ALIGN', (1, 1), (1, -1), 'CENTER'),
        ('ALIGN', (2, 1), (2, -1), 'CENTER'),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
        ('TOPPADDING', (0, 1), (-1, -1), 8),
        
        # Grid
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#bdc3c7')),
        
        # Alternating row colors
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')])
    ]))
    
    story.append(scores_table)
    story.append(Spacer(1, 0.4 * inch))
    
    # ============================================================
    # FRAME-BY-FRAME FORENSICS ANALYSIS TABLE
    # ============================================================
    if visual_results.get("frame_results") and visual_results.get("noise_scores"):
        story.append(Paragraph("Frame-by-Frame Forensics Analysis", heading_style))
        story.append(Spacer(1, 0.1 * inch))
        
        # Build table data — group every 5 frames
        frame_results = [r for r in visual_results["frame_results"] if r["face_detected"]]
        noise_scores = visual_results.get("noise_scores", [])
        fps = 30  # default
        
        # Dynamic column header based on verdict
        score_column_header = "Fake Score %" if final_verdict == "FAKE" else "Truth Score %"
        table_data = [["Time (s)", "Frame", score_column_header, "Noise Level"]]
        
        for idx, (frame, noise) in enumerate(zip(frame_results, noise_scores)):
            timestamp = round(frame["frame_number"] / fps, 2)
            confidence = round(frame["fake_score"], 1)
            noise_normalized = round(min(noise / 1000 * 100, 100), 1)  # normalize to 0-100
            
            table_data.append([
                f"{timestamp}s",
                str(frame["frame_number"]),
                f"{confidence}%",
                f"{noise_normalized}%"
            ])
        
        table = Table(table_data, colWidths=[1.5*inch, 1.2*inch, 2*inch, 2*inch])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3a6b")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4ff")]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        story.append(table)
        story.append(Spacer(1, 0.4 * inch))
    
    # ============================================================
    # FORENSICS EXPLANATION (LLM Generated)
    # ============================================================
    story.append(Paragraph("Forensics Analysis", heading_style))
    
    explanation_style = ParagraphStyle(
        'ExplanationStyle',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12,
        alignment=TA_JUSTIFY,
        fontName='Helvetica',
        leading=16
    )
    
    story.append(Paragraph(explanation, explanation_style))
    story.append(Spacer(1, 0.4 * inch))
    
    # ============================================================
    # AUDIO-VISUAL SYNC GRAPH
    # ============================================================
    if audio_results.get("success") and "audio_signal" in audio_results:
        story.append(Paragraph("Audio-Visual Synchronization Analysis", heading_style))
        
        graph_path = generate_sync_graph(
            audio_results["audio_signal"],
            audio_results["lip_signal"],
            audio_results.get("mismatch_timestamps", [])
        )
        
        # Add image to PDF
        img = Image(graph_path, width=6.5*inch, height=2*inch)
        story.append(img)
        story.append(Spacer(1, 0.3 * inch))
    
    # ============================================================
    # SUSPICIOUS FRAME TIMESTAMPS TABLE
    # ============================================================
    if len(mismatch_timestamps) > 0:
        story.append(Paragraph("Suspicious Frame Timestamps", heading_style))
        
        # Prepare timestamp data
        timestamp_data = [["Frame #", "Timestamp (s)", "Audio Energy", "Lip Movement", "Anomaly Type"]]
        
        # Add up to 15 most significant mismatches
        for mismatch in mismatch_timestamps[:15]:
            timestamp_data.append([
                str(mismatch.get('frame', 'N/A')),
                f"{mismatch.get('timestamp', 0):.2f}",
                f"{mismatch.get('audio_energy', 0):.3f}",
                f"{mismatch.get('lip_movement', 0):.3f}",
                mismatch.get('type', 'unknown').replace('_', ' ').title()
            ])
        
        timestamp_table = Table(timestamp_data, colWidths=[0.8*inch, 1.2*inch, 1.2*inch, 1.2*inch, 1.8*inch])
        timestamp_table.setStyle(TableStyle([
            # Header
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e74c3c')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('TOPPADDING', (0, 0), (-1, 0), 8),
            
            # Data
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor('#2c3e50')),
            ('ALIGN', (0, 1), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            ('TOPPADDING', (0, 1), (-1, -1), 6),
            
            # Grid
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#bdc3c7')),
            
            # Alternating rows
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#fff5f5'), colors.HexColor('#ffe5e5')])
        ]))
        
        story.append(timestamp_table)
        
        if len(mismatch_timestamps) > 15:
            story.append(Spacer(1, 0.1 * inch))
            note_style = ParagraphStyle(
                'NoteStyle',
                parent=styles['Normal'],
                fontSize=9,
                textColor=colors.HexColor('#7f8c8d'),
                alignment=TA_CENTER,
                fontName='Helvetica-Oblique'
            )
            story.append(Paragraph(f"Note: Showing first 15 of {len(mismatch_timestamps)} suspicious frames", note_style))
    else:
        story.append(Paragraph("Suspicious Frame Timestamps", heading_style))
        story.append(Paragraph("No significant audio-visual synchronization anomalies detected.", body_style))
    
    story.append(Spacer(1, 0.4 * inch))
    
    # ============================================================
    # FOOTER
    # ============================================================
    footer_style = ParagraphStyle(
        'FooterStyle',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#95a5a6'),
        alignment=TA_CENTER,
        fontName='Helvetica-Oblique'
    )
    
    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph("—" * 50, footer_style))
    story.append(Paragraph(
        f"This report was generated by DeepTrace AI Forensics System on {analysis_date}",
        footer_style
    ))
    story.append(Paragraph(
        "For research and educational purposes only. Results should be verified by qualified forensics experts.",
        footer_style
    ))
    
    # Build PDF
    try:
        doc.build(story)
        print(f"✅ PDF report generated: {output_path}")
        return str(output_path)
    finally:
        # Clean up temp graph file after PDF is built
        if graph_path and os.path.exists(graph_path):
            os.remove(graph_path)


############################################################
#  QUICK TEST
############################################################
if __name__ == "__main__":
    print("\n" + "="*60)
    print("PDF REPORT GENERATOR TEST")
    print("="*60)
    
    # Sample data for testing
    sample_visual_results = {
        "overall_verdict": "FAKE",
        "overall_confidence": 87.5,
        "max_fake_score": 92.3,
        "average_fake_score": 78.6,
        "total_frames_analyzed": 30,
        "faces_detected": 28
    }
    
    sample_audio_results = {
        "verdict": "SUSPICIOUS",
        "sync_score": 45.2,
        "correlation": 0.452,
        "total_mismatches": 15,
        "mismatch_percentage": 25.5,
        "total_frames": 30,
        "mismatch_timestamps": [
            {
                "frame": 50,
                "timestamp": 1.67,
                "audio_energy": 0.856,
                "lip_movement": 0.123,
                "difference": 0.733,
                "type": "audio_without_lips"
            },
            {
                "frame": 80,
                "timestamp": 2.67,
                "audio_energy": 0.234,
                "lip_movement": 0.789,
                "difference": 0.555,
                "type": "lips_without_audio"
            },
            {
                "frame": 120,
                "timestamp": 4.00,
                "audio_energy": 0.912,
                "lip_movement": 0.156,
                "difference": 0.756,
                "type": "audio_without_lips"
            }
        ]
    }
    
    sample_explanation = (
        "The video has been identified as a deepfake with 87.5% confidence based on comprehensive forensics analysis. "
        "Significant visual artifacts were detected with a maximum fake score of 92.3%, and substantial audio-visual "
        "desynchronization indicates the audio track may have been manipulated independently. The evidence strongly "
        "suggests this video has been synthetically generated using deepfake technology."
    )
    
    # Generate test report
    print("\n📝 Generating test report...")
    
    test_video_path = "test_videos/sample_video.mp4"
    
    report_path = generate_report(
        video_path=test_video_path,
        visual_results=sample_visual_results,
        audio_results=sample_audio_results,
        explanation=sample_explanation
    )
    
    print(f"\n✅ Test report saved to: {report_path}")
    print(f"📂 Check the outputs/ folder to view the PDF")
