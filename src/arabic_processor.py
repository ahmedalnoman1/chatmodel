"""
Arabic Text Processor
وحدة معالجة وتطبيع النصوص العربية
"""

import re
import unicodedata
from typing import List, Optional


class ArabicProcessor:
    """معالج النصوص العربية للتطبيع والتنظيف"""
    
    # الأحرف العربية للتطبيع
    ALEF_VARIANTS = 'أإآٱٲٳ'
    ALEF_NORMALIZED = 'ا'
    
    YEH_VARIANTS = 'ىۍێيئے'
    YEH_NORMALIZED = 'ي'
    
    # التشكيل العربي
    TASHKEEL = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    
    # الرموز غير الضرورية
    UNNECESSARY_CHARS = re.compile(r'[ـ]+')  # التطويل
    
    # أرقام عربية
    ARABIC_NUMERALS = '٠١٢٣٤٥٦٧٨٩'
    WESTERN_NUMERALS = '0123456789'
    
    def __init__(self, remove_tashkeel: bool = True, 
                 normalize_numbers: bool = True):
        """
        تهيئة المعالج
        
        Args:
            remove_tashkeel: إزالة التشكيل
            normalize_numbers: تحويل الأرقام العربية إلى غربية
        """
        self.remove_tashkeel = remove_tashkeel
        self.normalize_numbers = normalize_numbers
    
    def normalize_alef(self, text: str) -> str:
        """توحيد أشكال الألف (أ/إ/آ → ا)"""
        for char in self.ALEF_VARIANTS:
            text = text.replace(char, self.ALEF_NORMALIZED)
        return text
    
    def normalize_yeh(self, text: str) -> str:
        """توحيد أشكال الياء (ى → ي)"""
        for char in self.YEH_VARIANTS:
            text = text.replace(char, self.YEH_NORMALIZED)
        return text
    
    def remove_diacritics(self, text: str) -> str:
        """إزالة التشكيل من النص"""
        return self.TASHKEEL.sub('', text)
    
    def remove_tatweel(self, text: str) -> str:
        """إزالة حروف التطويل (ـ)"""
        return self.UNNECESSARY_CHARS.sub('', text)
    
    def normalize_numerals(self, text: str) -> str:
        """تحويل الأرقام العربية إلى غربية"""
        trans = str.maketrans(self.ARABIC_NUMERALS, self.WESTERN_NUMERALS)
        return text.translate(trans)
    
    def clean_whitespace(self, text: str) -> str:
        """تنظيف المسافات الزائدة"""
        # استبدال المسافات المتعددة بمسافة واحدة
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def normalize(self, text: str) -> str:
        """
        تطبيق جميع عمليات التطبيع على النص
        
        Args:
            text: النص المراد تطبيعه
            
        Returns:
            النص بعد التطبيع
        """
        if not text:
            return ""
        
        # إزالة Unicode الغريب
        text = unicodedata.normalize('NFKC', text)
        
        # توحيد الألفات
        text = self.normalize_alef(text)
        
        # توحيد الياء
        text = self.normalize_yeh(text)
        
        # إزالة التشكيل
        if self.remove_tashkeel:
            text = self.remove_diacritics(text)
        
        # إزالة التطويل
        text = self.remove_tatweel(text)
        
        # تطبيع الأرقام
        if self.normalize_numbers:
            text = self.normalize_numerals(text)
        
        # تنظيف المسافات
        text = self.clean_whitespace(text)
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        تقسيم النص إلى كلمات
        
        Args:
            text: النص المراد تقسيمه
            
        Returns:
            قائمة الكلمات
        """
        # تطبيع أولاً
        text = self.normalize(text)
        
        # تقسيم على المسافات وعلامات الترقيم
        tokens = re.split(r'[\s\.,،؛:!؟\?\(\)\[\]\{\}\"\']+', text)
        
        # إزالة العناصر الفارغة
        return [t for t in tokens if t]
    
    def chunk_text(self, text: str, chunk_size: int = 512, 
                   overlap: int = 50) -> List[str]:
        """
        تقسيم النص إلى قطع متداخلة
        
        Args:
            text: النص المراد تقسيمه
            chunk_size: حجم القطعة بالأحرف
            overlap: حجم التداخل بين القطع
            
        Returns:
            قائمة القطع النصية
        """
        if not text or chunk_size <= 0:
            return []
        
        text = self.normalize(text)
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # محاولة القطع عند نهاية جملة أو كلمة
            if end < len(text):
                # البحث عن نقطة قطع مناسبة
                last_period = text.rfind('.', start, end)
                last_arabic_period = text.rfind('。', start, end)
                last_space = text.rfind(' ', start, end)
                
                # الأولوية: نهاية جملة، ثم مسافة
                break_point = max(last_period, last_arabic_period)
                if break_point > start:
                    end = break_point + 1
                elif last_space > start:
                    end = last_space
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap if end - overlap > start else end
        
        return chunks


# دالة مساعدة للاستخدام المباشر
def normalize_arabic(text: str) -> str:
    """تطبيع نص عربي - دالة مساعدة"""
    processor = ArabicProcessor()
    return processor.normalize(text)
