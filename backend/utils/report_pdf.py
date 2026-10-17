import os
import xml.sax.saxutils
from datetime import datetime, timezone, timedelta

from PIL import Image, ImageOps
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image as RLImage
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# Set decompression bomb limit
Image.MAX_IMAGE_PIXELS = 100000000

PHT = timezone(timedelta(hours=8))

def format_pht_timestamp(utc_val=None, include_time: bool = True) -> str:
    if utc_val is None:
        return "--"
    try:
        if isinstance(utc_val, datetime):
            dt = utc_val if utc_val.tzinfo else utc_val.replace(tzinfo=timezone.utc)
        else:
            s = str(utc_val).strip()
            if not s:
                return "--"
            dt = datetime.fromisoformat(s.replace(" ", "T")).replace(tzinfo=timezone.utc)
        pht_dt = dt.astimezone(PHT)
        if include_time:
            return pht_dt.strftime("%Y-%m-%d %H:%M:%S PHT")
        return pht_dt.strftime("%Y-%m-%d")
    except Exception:
        return str(utc_val)

def safe_str(s):
    if s is None:
        return ""
    # XML escape for Paragraphs
    # Replace chars that might crash standard fonts. We use Helvetica which is standard but limited.
    # A simple approach is to encode to ascii with 'ignore' or 'replace', but let's be more forgiving.
    # reportlab will raise an error for characters not in its encoding.
    try:
        s = str(s).encode('latin-1', 'replace').decode('latin-1')
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Error escaping string '{s}': {e}")
        s = str(s)
    return xml.sax.saxutils.escape(s)

def process_image(img_path, max_dim=1600):
    """Open image, apply EXIF, downscale, save to a temp JPEG and return the path."""
    if not img_path or not os.path.exists(img_path):
        return None
    try:
        with Image.open(img_path) as img:
            img = ImageOps.exif_transpose(img)
            
            # Downscale
            w, h = img.size
            if max(w, h) > max_dim:
                ratio = max_dim / float(max(w, h))
                new_size = (int(w * ratio), int(h * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            temp_path = img_path + "_temp_pdf.jpg"
            img.save(temp_path, "JPEG", quality=85)
            return temp_path
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Error getting aspect ratio: {e}")
        return None

def draw_watermark(canvas, doc, is_mock, page_info):
    canvas.saveState()
    # Page numbers
    canvas.setFont('Helvetica', 9)
    canvas.drawRightString(doc.pagesize[0] - 0.5 * inch, 0.5 * inch, f"Page {doc.page} of {page_info['total_pages']}")
    
    # Watermark
    if is_mock:
        canvas.setFont('Helvetica-Bold', 60)
        canvas.setFillColorRGB(1, 0, 0, alpha=0.15)
        canvas.translate(doc.pagesize[0]/2, doc.pagesize[1]/2)
        canvas.rotate(45)
        canvas.drawCentredString(0, 0, "DEMO / MOCK RESULTS")
    canvas.restoreState()

def build_pdf(data: dict, output_path: str):
    """
    Builds the PDF report using a plain dict of data.
    """
    # The first pass to determine total pages might be needed, or we just write "Page X".
    # ReportLab doesn't have an easy "Page X of Y" out of the box without multi-pass,
    # but we can implement a custom canvas or just use a placeholder.
    # For now, we'll just implement a simple page counter using a canvas maker.
    
    class NumberedCanvas(SimpleDocTemplate):
        def __init__(self, *args, **kwargs):
            self.is_mock = kwargs.pop('is_mock', False)
            super().__init__(*args, **kwargs)
            
        def build(self, flowables, canvasmaker=None):
            self.page_info = {'total_pages': 0}
            
            # A canvas class that keeps track of the page
            from reportlab.pdfgen import canvas
            class PageNumCanvas(canvas.Canvas):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.pages = []
                    
                def showPage(self):
                    self.pages.append(dict(self.__dict__))
                    self._startPage()
                    
                def save(self):
                    num_pages = len(self.pages)
                    for page in self.pages:
                        self.__dict__.update(page)
                        self.draw_page_number(num_pages)
                        super().showPage()
                    super().save()
                    
                def draw_page_number(self, page_count):
                    draw_watermark(self, doc, is_mock, {'total_pages': page_count})
            
            doc = self
            is_mock = self.is_mock
            
            if canvasmaker is None:
                canvasmaker = PageNumCanvas
            super().build(flowables, canvasmaker=canvasmaker)

    # Instead of custom canvas, a standard approach:
    # Actually reportlab canvas maker approach works.
    
    is_mock = data.get('is_mock', False)
    # We will just use standard simple doc template and a page template.
    # For simplicity in this mock, let's use a simpler page template approach.

    doc = SimpleDocTemplate(output_path, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    
    styles = getSampleStyleSheet()
    title_style = styles['Heading1']
    h2_style = styles['Heading2']
    h3_style = styles['Heading3']
    normal_style = styles['Normal']
    
    elements = []
    
    # Header
    title_text = f"WIREFRAME Report: {safe_str(data.get('id', 'Unknown'))}"
    if is_mock:
        title_text += " <font color='red'>[DEMO / MOCK RESULTS]</font>"
    elements.append(Paragraph(title_text, title_style))
    elements.append(Paragraph(f"Generated at: {format_pht_timestamp(datetime.now(timezone.utc))}", normal_style))
    elements.append(Spacer(1, 12))
    
    # Metadata
    elements.append(Paragraph(f"<b>Company:</b> {safe_str(data.get('company', '-'))}", normal_style))
    elements.append(Paragraph(f"<b>Barangay:</b> {safe_str(data.get('barangay', '-'))}", normal_style))
    elements.append(Paragraph(f"<b>Street:</b> {safe_str(data.get('street', '-'))}", normal_style))
    if data.get('sector'):
        elements.append(Paragraph(f"<b>Sector:</b> {safe_str(data.get('sector'))}", normal_style))
    
    gps_str = f"<b>GPS:</b> {safe_str(data.get('gps_lat'))}, {safe_str(data.get('gps_lng'))}"
    if data.get('gps_source'):
        gps_str += f" <i>(Source: {safe_str(data.get('gps_source'))})</i>"
    elements.append(Paragraph(gps_str, normal_style))
    elements.append(Spacer(1, 12))
    
    # Safety Verdict
    verdict = safe_str(data.get('safety_level', 'Unknown'))
    elements.append(Paragraph(f"<b>Overall Safety Level:</b> {verdict}", h2_style))
    if data.get('action_text'):
        elements.append(Paragraph(f"<b>Action:</b> {safe_str(data.get('action_text'))}", normal_style))
    
    if data.get('reasons'):
        elements.append(Paragraph("<b>Reasons:</b>", normal_style))
        for r in data.get('reasons', []):
            elements.append(Paragraph(f"- {safe_str(r)}", normal_style))
            
    elements.append(Spacer(1, 12))
    
    # Config
    conf = data.get('config', {})
    m_val = conf.get('margin')
    if m_val is None or m_val == "Not available":
        margin_str = "Not available"
    else:
        margin_str = f"{safe_str(m_val)}m"
    elements.append(Paragraph(f"<b>Config Snapshot:</b> Pole height: {safe_str(conf.get('pole_height_m'))}m | PEC Min: {safe_str(conf.get('pec_min'))}m | Margin: {margin_str}", normal_style))
    elements.append(Paragraph(f"<b>Scale Source:</b> {safe_str(data.get('scale_source', 'Assumed'))}", normal_style))
    elements.append(Paragraph(f"<b>Model:</b> {safe_str(data.get('model_file', 'mock'))}", normal_style))
    elements.append(Spacer(1, 12))
    
    def get_img_flowable(path, max_width, max_height):
        if not path:
            return Paragraph("<i>[Image Missing/Corrupt]</i>", normal_style)
        try:
            img = RLImage(path)
            # scale keeping aspect ratio
            aspect = img.imageWidth / float(img.imageHeight)
            if aspect > 1:
                img.drawWidth = max_width
                img.drawHeight = max_width / aspect
            else:
                img.drawHeight = max_height
                img.drawWidth = max_height * aspect
                
            # If still too wide/tall
            if img.drawWidth > max_width:
                img.drawWidth = max_width
                img.drawHeight = max_width / aspect
            if img.drawHeight > max_height:
                img.drawHeight = max_height
                img.drawWidth = max_height * aspect
                
            return img
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to embed image {path}: {e}")
            return Paragraph("<i>[Image Error]</i>", normal_style)

    orig_temp = None
    ann_temp = None
    spec_temp = None
    try:
        # Images
        elements.append(Paragraph("<b>Images</b>", h2_style))
        orig_temp = process_image(data.get('original_path'))
        ann_temp = process_image(data.get('annotated_path'))
        
        # We want them side by side
        img1 = get_img_flowable(orig_temp, 3 * inch, 4 * inch)
        img2 = get_img_flowable(ann_temp, 3 * inch, 4 * inch)
        
        img_table = Table([[img1, img2]], colWidths=[3.2*inch, 3.2*inch])
        img_table.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ]))
        elements.append(img_table)
        elements.append(Spacer(1, 12))
        
        # VNI
        elements.append(Paragraph(f"<b>Vegetation Index (VNI):</b> {safe_str(data.get('vni'))}", h3_style))
        spec_temp = process_image(data.get('spectrum_path'))
        if spec_temp:
            elements.append(get_img_flowable(spec_temp, 6*inch, 2*inch))
        elements.append(Spacer(1, 12))
        
        # Wires Table
        elements.append(Paragraph("<b>Detected Wires</b>", h2_style))
        wires = data.get('wires', [])
        if not wires:
            elements.append(Paragraph("No wires detected.", normal_style))
        else:
            table_data = [[
                Paragraph("<b>Class</b>", normal_style),
                Paragraph("<b>Conf</b>", normal_style),
                Paragraph("<b>CRS</b>", normal_style),
                Paragraph("<b>Band</b>", normal_style),
                Paragraph("<b>Sag (m)</b>", normal_style),
                Paragraph("<b>Span (m)</b>", normal_style),
                Paragraph("<b>Clearance</b>", normal_style),
                Paragraph("<b>Safety</b>", normal_style)
            ]]
            
            for w in wires:
                table_data.append([
                    Paragraph(safe_str(w.get('class_name')), normal_style),
                    Paragraph(safe_str(f"{w.get('confidence', 0):.2f}"), normal_style),
                    Paragraph(safe_str(w.get('crs')), normal_style),
                    Paragraph(safe_str(w.get('crs_band')), normal_style),
                    Paragraph(safe_str(w.get('sag_m')), normal_style),
                    Paragraph(safe_str(w.get('span_m')), normal_style),
                    Paragraph(safe_str(f"{w.get('clearance_min_m')} - {w.get('clearance_max_m')}"), normal_style),
                    Paragraph(safe_str(w.get('safety_level')), normal_style),
                ])
                
            t = Table(table_data, repeatRows=1)
            t.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
                ('INNERGRID', (0,0), (-1,-1), 0.25, colors.black),
                ('BOX', (0,0), (-1,-1), 0.25, colors.black),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ]))
            elements.append(t)
            
        elements.append(Spacer(1, 24))
        elements.append(Paragraph("<i>Advisory estimate, not an official determination.</i>", normal_style))

        # Build PDF with PageNumCanvas equivalent inline
        from reportlab.pdfgen import canvas
        class PageNumCanvas(canvas.Canvas):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.pages = []
                
            def showPage(self):
                self.pages.append(dict(self.__dict__))
                self._startPage()
                
            def save(self):
                num_pages = len(self.pages)
                for page in self.pages:
                    self.__dict__.update(page)
                    draw_watermark(self, doc, is_mock, {'total_pages': num_pages})
                    super().showPage()
                super().save()
                
        doc.build(elements, canvasmaker=PageNumCanvas)
    finally:
        # Cleanup temps
        for p in [orig_temp, ann_temp, spec_temp]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).warning(f"Failed to remove temp PDF image {p}: {e}")
                    pass

def build_summary_pdf(data: dict, output_path: str):
    """
    Builds the Summary PDF.
    """
    is_mock = data.get('has_mock', False)

    doc = SimpleDocTemplate(output_path, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    title_style = styles['Heading1']
    h2_style = styles['Heading2']
    normal_style = styles['Normal']
    
    elements = []
    
    title_text = "WIREFRAME Summary Report"
    if is_mock:
        title_text += " <font color='red'>[DEMO / MOCK RESULTS]</font>"
    elements.append(Paragraph(title_text, title_style))
    elements.append(Paragraph(f"Generated at: {format_pht_timestamp(datetime.now(timezone.utc))}", normal_style))
    elements.append(Spacer(1, 12))
    
    # Summary Table
    elements.append(Paragraph("<b>Ranking Table</b>", h2_style))
    
    table_data = [[
        Paragraph("<b>Rank</b>", normal_style),
        Paragraph("<b>Company</b>", normal_style),
        Paragraph("<b>Location</b>", normal_style),
        Paragraph("<b>Date</b>", normal_style),
        Paragraph("<b>VNI</b>", normal_style),
        Paragraph("<b>Max CRS</b>", normal_style),
        Paragraph("<b>Safety</b>", normal_style),
        Paragraph("<b>Action</b>", normal_style)
    ]]
    
    for i, item in enumerate(data.get('items', [])):
        table_data.append([
            Paragraph(str(i+1), normal_style),
            Paragraph(safe_str(item.get('company', '-')), normal_style),
            Paragraph(safe_str(f"{item.get('barangay', '-')} / {item.get('street', '-')}"), normal_style),
            Paragraph(safe_str(item.get('date')), normal_style),
            Paragraph(safe_str(item.get('vni')), normal_style),
            Paragraph(safe_str(item.get('max_crs')), normal_style),
            Paragraph(safe_str(item.get('safety_level')), normal_style),
            Paragraph(safe_str(item.get('action')), normal_style),
        ])
        
    if len(table_data) > 1:
        col_widths = [0.35 * inch, 0.85 * inch, 1.3 * inch, 0.8 * inch, 0.55 * inch, 0.65 * inch, 0.8 * inch, 2.2 * inch]
        t = Table(table_data, colWidths=col_widths, repeatRows=1)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
            ('INNERGRID', (0,0), (-1,-1), 0.25, colors.black),
            ('BOX', (0,0), (-1,-1), 0.25, colors.black),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ]))
        elements.append(t)
    else:
        elements.append(Paragraph("No items to summarize.", normal_style))
        
    elements.append(Spacer(1, 24))
    elements.append(Paragraph("<i>Advisory estimate, not an official determination.</i>", normal_style))

    from reportlab.pdfgen import canvas
    class PageNumCanvas(canvas.Canvas):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.pages = []
            
        def showPage(self):
            self.pages.append(dict(self.__dict__))
            self._startPage()
            
        def save(self):
            num_pages = len(self.pages)
            for page in self.pages:
                self.__dict__.update(page)
                draw_watermark(self, doc, is_mock, {'total_pages': num_pages})
                super().showPage()
            super().save()

    doc.build(elements, canvasmaker=PageNumCanvas)
