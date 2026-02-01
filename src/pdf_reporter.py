"""
PDF Report Generator
مولد تقارير PDF للشكاوى
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class PDFReporter:
    """مولد تقارير PDF للشكاوى بالعربية"""
    
    # ترجمات
    CRIME_TYPES_AR = {
        "account_hack": "اختراق حسابات",
        "financial_fraud": "احتيال مالي إلكتروني",
        "phishing": "رسائل تصيد",
        "identity_theft": "انتحال الهوية",
        "blackmail": "ابتزاز وتهديد إلكتروني",
        "data_breach": "تسريب بيانات"
    }
    
    SEVERITY_AR = {
        "low": "منخفض",
        "medium": "متوسط",
        "high": "عالٍ",
        "critical": "حرج"
    }
    
    def __init__(self, output_dir: str = "reports"):
        """
        تهيئة المولد
        
        Args:
            output_dir: مجلد حفظ التقارير
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def _reshape_arabic(self, text: str) -> str:
        """إعادة تشكيل النص العربي للـ PDF"""
        try:
            import arabic_reshaper
            from bidi.algorithm import get_display
            
            reshaped = arabic_reshaper.reshape(text)
            return get_display(reshaped)
        except ImportError:
            logger.warning("arabic-reshaper أو python-bidi غير متوفر")
            return text
    
    def generate_report(self, complaint_data: Dict, 
                        filename: Optional[str] = None) -> str:
        """
        توليد تقرير PDF للشكوى
        
        Args:
            complaint_data: بيانات الشكوى
            filename: اسم الملف (اختياري)
            
        Returns:
            مسار الملف المولّد
        """
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import cm
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont
        except ImportError:
            raise ImportError("يرجى تثبيت reportlab: pip install reportlab")
        
        # اسم الملف
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            complaint_id = complaint_data.get("id", "unknown")
            filename = f"complaint_{complaint_id}_{timestamp}.pdf"
        
        filepath = self.output_dir / filename
        
        # محاولة تسجيل خط عربي
        try:
            # محاولة استخدام خط Arial أو خط عربي آخر
            pdfmetrics.registerFont(TTFont('Arabic', 'C:/Windows/Fonts/arial.ttf'))
            arabic_font = 'Arabic'
        except:
            arabic_font = 'Helvetica'
            logger.warning("تعذر تحميل خط عربي، استخدام الخط الافتراضي")
        
        # إنشاء المستند
        doc = SimpleDocTemplate(
            str(filepath),
            pagesize=A4,
            rightMargin=2*cm,
            leftMargin=2*cm,
            topMargin=2*cm,
            bottomMargin=2*cm
        )
        
        # الأنماط
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'ArabicTitle',
            parent=styles['Heading1'],
            fontName=arabic_font,
            fontSize=18,
            alignment=1,  # وسط
            spaceAfter=30
        )
        
        heading_style = ParagraphStyle(
            'ArabicHeading',
            parent=styles['Heading2'],
            fontName=arabic_font,
            fontSize=14,
            spaceAfter=12
        )
        
        body_style = ParagraphStyle(
            'ArabicBody',
            parent=styles['Normal'],
            fontName=arabic_font,
            fontSize=11,
            spaceAfter=6
        )
        
        # العناصر
        elements = []
        
        # العنوان
        title = self._reshape_arabic("تقرير بلاغ جريمة إلكترونية")
        elements.append(Paragraph(title, title_style))
        elements.append(Spacer(1, 20))
        
        # معلومات أساسية
        info_data = [
            [self._reshape_arabic("رقم البلاغ:"), complaint_data.get("id", "-")],
            [self._reshape_arabic("تاريخ الإنشاء:"), complaint_data.get("created_at", "-")[:10]],
            [self._reshape_arabic("نوع الجريمة:"), 
             self._reshape_arabic(self.CRIME_TYPES_AR.get(
                 complaint_data.get("crime_type", ""), "غير محدد"))],
            [self._reshape_arabic("مستوى الخطورة:"), 
             self._reshape_arabic(self.SEVERITY_AR.get(
                 complaint_data.get("severity", ""), "غير محدد"))],
            [self._reshape_arabic("المنصة:"), 
             self._reshape_arabic(complaint_data.get("platform", "غير محددة") or "غير محددة")]
        ]
        
        info_table = Table(info_data, colWidths=[4*cm, 10*cm])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('PADDING', (0, 0), (-1, -1), 8),
            ('FONTNAME', (0, 0), (-1, -1), arabic_font),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
        ]))
        elements.append(info_table)
        elements.append(Spacer(1, 20))
        
        # وصف الحادثة
        elements.append(Paragraph(
            self._reshape_arabic("وصف الحادثة:"), heading_style))
        
        description = complaint_data.get("description", "لا يوجد وصف")
        # تقسيم الوصف لفقرات
        for para in description.split('\n'):
            if para.strip():
                elements.append(Paragraph(
                    self._reshape_arabic(para.strip()), body_style))
        
        elements.append(Spacer(1, 20))
        
        # المحادثة
        conversation = complaint_data.get("conversation", [])
        if conversation:
            elements.append(Paragraph(
                self._reshape_arabic("سجل المحادثة:"), heading_style))
            
            for msg in conversation[-10:]:  # آخر 10 رسائل
                role = self._reshape_arabic("المستخدم:" if msg["role"] == "user" else "المساعد:")
                content = msg["content"][:300]  # أول 300 حرف
                elements.append(Paragraph(
                    f"<b>{role}</b> {self._reshape_arabic(content)}", body_style))
        
        # بناء PDF
        doc.build(elements)
        
        logger.info(f"تم إنشاء التقرير: {filepath}")
        return str(filepath)
    
    def list_reports(self) -> list:
        """قائمة التقارير المتوفرة"""
        return list(self.output_dir.glob("*.pdf"))
