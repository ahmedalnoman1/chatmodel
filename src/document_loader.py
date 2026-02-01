"""
Document Loader
وحدة تحميل واستخراج النصوص من المستندات
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Union
import logging

# إعداد التسجيل
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentLoader:
    """محمّل المستندات - يدعم PDF, DOCX, TXT, CSV, XLSX"""
    
    SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.txt', '.csv', '.xlsx'}
    
    def __init__(self, arabic_processor=None):
        """
        تهيئة المحمّل
        
        Args:
            arabic_processor: معالج النصوص العربية (اختياري)
        """
        self.arabic_processor = arabic_processor
    
    def load_pdf(self, file_path: str) -> str:
        """
        استخراج النص من PDF باستخدام pdfplumber مع fallback
        
        Args:
            file_path: مسار ملف PDF
            
        Returns:
            النص المستخرج
        """
        text = ""
        
        # المحاولة الأولى: pdfplumber (أفضل للعربية)
        try:
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            
            if text.strip():
                logger.info(f"تم استخراج النص من {file_path} باستخدام pdfplumber")
                return text
                
        except ImportError:
            logger.warning("pdfplumber غير متوفر، جاري المحاولة بـ PyPDF2")
        except Exception as e:
            logger.warning(f"فشل pdfplumber: {e}")
        
        # Fallback: PyPDF2
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(file_path)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            
            logger.info(f"تم استخراج النص من {file_path} باستخدام PyPDF2")
            return text
            
        except ImportError:
            logger.error("PyPDF2 غير متوفر")
        except Exception as e:
            logger.error(f"فشل استخراج PDF: {e}")
        
        return text
    
    def load_docx(self, file_path: str) -> str:
        """
        استخراج النص من DOCX
        
        Args:
            file_path: مسار ملف DOCX
            
        Returns:
            النص المستخرج
        """
        try:
            from docx import Document
            doc = Document(file_path)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n".join(paragraphs)
        except ImportError:
            logger.error("python-docx غير متوفر")
        except Exception as e:
            logger.error(f"فشل استخراج DOCX: {e}")
        return ""
    
    def load_txt(self, file_path: str) -> str:
        """
        قراءة ملف نصي
        
        Args:
            file_path: مسار الملف النصي
            
        Returns:
            محتوى الملف
        """
        encodings = ['utf-8', 'utf-8-sig', 'cp1256', 'iso-8859-6']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.error(f"فشل قراءة الملف: {e}")
                return ""
        
        logger.error(f"فشل تحديد ترميز الملف: {file_path}")
        return ""
    
    def load_csv(self, file_path: str) -> str:
        """
        قراءة ملف CSV وتحويله لنص
        
        Args:
            file_path: مسار ملف CSV
            
        Returns:
            محتوى CSV كنص
        """
        try:
            import pandas as pd
            df = pd.read_csv(file_path)
            return df.to_string()
        except ImportError:
            logger.error("pandas غير متوفر")
        except Exception as e:
            logger.error(f"فشل قراءة CSV: {e}")
        return ""
    
    def load_xlsx(self, file_path: str) -> str:
        """
        قراءة ملف Excel وتحويله لنص
        
        Args:
            file_path: مسار ملف Excel
            
        Returns:
            محتوى Excel كنص
        """
        try:
            import pandas as pd
            df = pd.read_excel(file_path)
            return df.to_string()
        except ImportError:
            logger.error("pandas أو openpyxl غير متوفر")
        except Exception as e:
            logger.error(f"فشل قراءة Excel: {e}")
        return ""
    
    def load_document(self, file_path: str) -> Dict:
        """
        تحميل مستند بناءً على امتداده
        
        Args:
            file_path: مسار المستند
            
        Returns:
            قاموس يحتوي على النص والبيانات الوصفية
        """
        path = Path(file_path)
        extension = path.suffix.lower()
        
        if extension not in self.SUPPORTED_EXTENSIONS:
            logger.warning(f"نوع الملف غير مدعوم: {extension}")
            return {"text": "", "metadata": {"file": str(path), "error": "unsupported"}}
        
        # اختيار المحمّل المناسب
        loaders = {
            '.pdf': self.load_pdf,
            '.docx': self.load_docx,
            '.txt': self.load_txt,
            '.csv': self.load_csv,
            '.xlsx': self.load_xlsx
        }
        
        text = loaders[extension](file_path)
        
        # تطبيع النص العربي إذا توفر المعالج
        if self.arabic_processor and text:
            text = self.arabic_processor.normalize(text)
        
        return {
            "text": text,
            "metadata": {
                "file": str(path),
                "filename": path.name,
                "extension": extension,
                "size": os.path.getsize(file_path) if os.path.exists(file_path) else 0
            }
        }
    
    def load_directory(self, directory_path: str) -> List[Dict]:
        """
        تحميل جميع المستندات من مجلد
        
        Args:
            directory_path: مسار المجلد
            
        Returns:
            قائمة المستندات المحمّلة
        """
        documents = []
        path = Path(directory_path)
        
        if not path.exists():
            logger.error(f"المجلد غير موجود: {directory_path}")
            return documents
        
        for ext in self.SUPPORTED_EXTENSIONS:
            for file_path in path.glob(f"**/*{ext}"):
                doc = self.load_document(str(file_path))
                if doc["text"]:
                    documents.append(doc)
                    logger.info(f"تم تحميل: {file_path.name}")
        
        logger.info(f"تم تحميل {len(documents)} مستند من {directory_path}")
        return documents
