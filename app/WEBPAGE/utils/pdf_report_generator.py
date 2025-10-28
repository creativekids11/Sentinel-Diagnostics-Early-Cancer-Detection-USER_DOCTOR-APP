"""
Professional Medical Report PDF Generator
Generates PDF reports following medical prescription format with doctor details,
patient information, diagnosis, procedures, medications, and signature.
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfgen import canvas
from datetime import datetime
import io
import os
import re
import base64
from PIL import Image as PILImage
from io import BytesIO


def generate_medical_report_pdf(report_data):
    """
    Generate a professional medical report PDF
    
    Args:
        report_data: Dictionary containing report information including:
            - doctor_name, doctor_qualifications, doctor_email
            - patient_name, patient_email, patient_phone
            - report_title, diagnosis, symptoms, procedures, medications
            - suggested_solution, additional_notes, doctor_signature
            - created_at
    
    Returns:
        bytes: PDF file content as bytes
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                           rightMargin=72, leftMargin=72,
                           topMargin=72, bottomMargin=72)
    
    # Container for the 'Flowable' objects
    elements = []
    
    # Define styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#1e3a5f'),
        spaceAfter=12,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=13,
        textColor=colors.HexColor('#2c5f7d'),
        spaceAfter=6,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor=colors.black
    )
    
    small_style = ParagraphStyle(
        'CustomSmall',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#666666')
    )
    
    # Add logo if exists
    logo_path = os.path.join('static', 'assets', 'logo.png')
    if os.path.exists(logo_path):
        try:
            logo = Image(logo_path, width=2.5*inch, height=0.8*inch)
            logo.hAlign = 'CENTER'
            elements.append(logo)
            elements.append(Spacer(1, 12))
        except:
            pass  # Skip logo if there's an error
    
    # Doctor Information Header
    doctor_name = report_data.get('doctor_name', 'Dr. [Name]')
    doctor_qualifications = report_data.get('doctor_qualifications', '')
    
    doctor_header = Paragraph(
        f"<b>{doctor_name}</b>",
        title_style
    )
    elements.append(doctor_header)
    
    if doctor_qualifications:
        qualifications_para = Paragraph(
            doctor_qualifications,
            ParagraphStyle('Qualifications', parent=small_style, alignment=TA_CENTER)
        )
        elements.append(qualifications_para)
        elements.append(Spacer(1, 6))
    
    # Horizontal line
    elements.append(Spacer(1, 12))
    
    # Patient Information Box
    patient_data = [
        ['Patient Name:', report_data.get('case_patient_name') or report_data.get('patient_name', 'N/A'), 'Date:', 
         datetime.fromisoformat(report_data.get('created_at', datetime.now().isoformat())).strftime('%B %d, %Y')],
        ['Age:', report_data.get('patient_age', 'N/A'), 'Gender:', report_data.get('patient_gender', 'N/A')],
        ['Weight:', f"{report_data.get('patient_weight', 'N/A')} kg" if report_data.get('patient_weight') else 'N/A', 'Diagnosis:', report_data.get('diagnosis', 'General Consultation')]
    ]
    
    patient_table = Table(patient_data, colWidths=[1.2*inch, 2.3*inch, 0.8*inch, 2.2*inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f0f8ff')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#1e90ff')),
    ]))
    elements.append(patient_table)
    elements.append(Spacer(1, 20))
    
    # Patient Description
    if report_data.get('patient_description'):
        elements.append(Paragraph("<b>Patient History & Description:</b>", heading_style))
        elements.append(Paragraph(report_data['patient_description'], normal_style))
        elements.append(Spacer(1, 12))
    
    # Symptoms & Findings
    if report_data.get('symptoms'):
        elements.append(Paragraph("<b>Symptoms & Clinical Findings:</b>", heading_style))
        elements.append(Paragraph(report_data['symptoms'], normal_style))
        elements.append(Spacer(1, 12))
    
    # Procedures Performed
    if report_data.get('procedures'):
        elements.append(Paragraph("<b>Procedures Performed:</b>", heading_style))
        procedures_para = Paragraph(report_data['procedures'], normal_style)
        elements.append(procedures_para)
        elements.append(Spacer(1, 12))
    
    # Medications Prescribed
    if report_data.get('medications'):
        elements.append(Paragraph("<b>Medications Prescribed:</b>", heading_style))
        
        # Try to format medications as a list if they contain newlines or bullets
        medications_text = report_data['medications']
        if '\n' in medications_text:
            # Format each line as a bullet point
            med_lines = [line.strip() for line in medications_text.split('\n') if line.strip()]
            formatted_meds = '<br/>'.join([f"• {line}" for line in med_lines])
            medications_para = Paragraph(formatted_meds, normal_style)
        else:
            medications_para = Paragraph(medications_text, normal_style)
        
        elements.append(medications_para)
        elements.append(Spacer(1, 12))
    
    # Treatment Plan / Suggested Solution
    if report_data.get('suggested_solution'):
        elements.append(Paragraph("<b>Treatment Plan & Recommendations:</b>", heading_style))
        elements.append(Paragraph(report_data['suggested_solution'], normal_style))
        elements.append(Spacer(1, 12))
    
    # Additional Notes
    if report_data.get('additional_notes'):
        elements.append(Paragraph("<b>Additional Notes:</b>", heading_style))
        elements.append(Paragraph(report_data['additional_notes'], normal_style))
        elements.append(Spacer(1, 12))
    
    # AI Scanner Analysis Section
    if report_data.get('ai_scan_type') and (report_data.get('ai_original_image') or report_data.get('ai_segmented_image')):
        elements.append(Paragraph("<b>AI Scanner Analysis:</b>", heading_style))
        
        # Add scan type information
        scan_type = report_data.get('ai_scan_type', 'Unknown').title()
        elements.append(Paragraph(f"<b>Scan Type:</b> {scan_type}", normal_style))
        elements.append(Spacer(1, 6))
        
        # Process original image
        if report_data.get('ai_original_image'):
            try:
                # Handle data URLs (remove prefix if present)
                image_data_str = report_data['ai_original_image']
                if image_data_str.startswith('data:image/'):
                    # Extract base64 data from data URL
                    base64_data = image_data_str.split(',')[1]
                else:
                    base64_data = image_data_str
                
                # Decode base64 image
                image_data = base64.b64decode(base64_data)
                image_buffer = BytesIO(image_data)
                pil_image = PILImage.open(image_buffer)
                
                # Convert to RGB if necessary
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                
                # Create temporary file for ReportLab Image
                temp_buffer = BytesIO()
                pil_image.save(temp_buffer, format='PNG')
                temp_buffer.seek(0)
                
                # Add image to PDF (resize to fit page width)
                img_width = min(4*inch, pil_image.width * 0.75)  # Max 4 inches wide
                aspect_ratio = pil_image.height / pil_image.width
                img_height = img_width * aspect_ratio
                
                # Ensure height doesn't exceed reasonable limits
                if img_height > 3*inch:
                    img_height = 3*inch
                    img_width = img_height / aspect_ratio
                
                ai_image = Image(temp_buffer, width=img_width, height=img_height)
                ai_image.hAlign = 'CENTER'
                
                elements.append(Paragraph("<i>Original Scan:</i>", small_style))
                elements.append(ai_image)
                elements.append(Spacer(1, 12))
                
            except Exception as e:
                # If image processing fails, add text note
                elements.append(Paragraph("<i>Original Scan: [Image could not be displayed]</i>", small_style))
                elements.append(Spacer(1, 6))
        
        # Process segmented image
        if report_data.get('ai_segmented_image'):
            try:
                # Handle data URLs (remove prefix if present)
                image_data_str = report_data['ai_segmented_image']
                if image_data_str.startswith('data:image/'):
                    # Extract base64 data from data URL
                    base64_data = image_data_str.split(',')[1]
                else:
                    base64_data = image_data_str
                
                # Decode base64 image
                image_data = base64.b64decode(base64_data)
                image_buffer = BytesIO(image_data)
                pil_image = PILImage.open(image_buffer)
                
                # Convert to RGB if necessary
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                
                # Create temporary file for ReportLab Image
                temp_buffer = BytesIO()
                pil_image.save(temp_buffer, format='PNG')
                temp_buffer.seek(0)
                
                # Add image to PDF (resize to fit page width)
                img_width = min(4*inch, pil_image.width * 0.75)  # Max 4 inches wide
                aspect_ratio = pil_image.height / pil_image.width
                img_height = img_width * aspect_ratio
                
                # Ensure height doesn't exceed reasonable limits
                if img_height > 3*inch:
                    img_height = 3*inch
                    img_width = img_height / aspect_ratio
                
                ai_image = Image(temp_buffer, width=img_width, height=img_height)
                ai_image.hAlign = 'CENTER'
                
                elements.append(Paragraph("<i>AI Analysis Result:</i>", small_style))
                elements.append(ai_image)
                elements.append(Spacer(1, 12))
                
            except Exception as e:
                # If image processing fails, add text note
                elements.append(Paragraph("<i>AI Analysis Result: [Image could not be displayed]</i>", small_style))
                elements.append(Spacer(1, 6))
    
    # Medical Symbol (Caduceus) - using text representation
    elements.append(Spacer(1, 20))
    
    # Signature Section
    # Strip HTML tags from doctor signature
    raw_signature = report_data.get('doctor_signature', doctor_name)
    clean_signature = re.sub(r'<[^>]+>', '', raw_signature)  # Remove HTML tags
    
    signature_data = [
        ['', ''],
        ['', f"{clean_signature}"],
    ]
    
    signature_table = Table(signature_data, colWidths=[4*inch, 2.5*inch])
    signature_table.setStyle(TableStyle([
        ('ALIGN', (1, 0), (1, -1), 'CENTER'),
        ('LINEABOVE', (1, 1), (1, 1), 1, colors.black),
        ('FONTSIZE', (1, 1), (1, 1), 10),
        ('FONTSIZE', (1, 2), (1, 2), 8),
        ('TEXTCOLOR', (1, 2), (1, 2), colors.HexColor('#666666')),
    ]))
    elements.append(signature_table)
    
    # Footer with contact information
    elements.append(Spacer(1, 15))
    
    doctor_phone = report_data.get('doctor_phone', '+9920753662')
    footer_data = [
        [f'📞 {doctor_phone}']
    ]
    
    footer_table = Table(footer_data, colWidths=[6.5*inch])
    footer_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (0, 0), 'RIGHT'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#2c5f7d')),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
    ]))
    elements.append(footer_table)
    
    # Medical symbol at the bottom left
    caduceus_style = ParagraphStyle(
        'Caduceus',
        parent=small_style,
        fontSize=30,
        textColor=colors.HexColor('#1e90ff'),
        alignment=TA_LEFT
    )
    
    # Build PDF
    doc.build(elements)
    
    # Get the value of the BytesIO buffer and return it
    pdf_bytes = buffer.getvalue()
    buffer.close()
    
    return pdf_bytes


def add_geometric_shapes(canvas_obj, doc):
    """Add geometric shapes decoration similar to the reference image"""
    # This is a callback function for more advanced PDF customization
    canvas_obj.saveState()
    
    # Add geometric shapes in top-right corner
    canvas_obj.setFillColor(colors.HexColor('#1e90ff'))
    canvas_obj.setStrokeColor(colors.HexColor('#1e3a5f'))
    
    # Draw parallelogram shapes
    page_width, page_height = letter
    
    # Top right decorative shapes
    canvas_obj.rotate(45)
    canvas_obj.rect(page_width - 100, page_height - 150, 60, 60, fill=1, stroke=0)
    canvas_obj.setFillColor(colors.HexColor('#2c5f7d'))
    canvas_obj.rect(page_width - 40, page_height - 150, 60, 60, fill=1, stroke=0)
    
    canvas_obj.restoreState()
