"""
Arabic Embeddings
وحدة توليد Embeddings للنصوص العربية باستخدام AraBERT
"""

import numpy as np
from typing import List, Union, Optional
import logging

logger = logging.getLogger(__name__)


class ArabicEmbeddings:
    """توليد Embeddings للنصوص العربية"""
    
    def __init__(self, model_name: str = "aubmindlab/bert-base-arabertv2",
                 device: str = "cpu"):
        """
        تهيئة نموذج Embeddings
        
        Args:
            model_name: اسم النموذج من HuggingFace
            device: الجهاز (cpu/cuda)
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """تحميل النموذج"""
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"جاري تحميل نموذج Embeddings: {self.model_name}")
            
            # محاولة التحميل من الملفات المحلية فقط أولاً لتجنب مشاكل الاتصال
            try:
                self.model = SentenceTransformer(self.model_name, device=self.device, local_files_only=True)
                logger.info("تم تحميل النموذج من الملفات المحلية.")
            except Exception as local_err:
                logger.warning(f"لم يتم العثور على النموذج محلياً أو حدث خطأ: {local_err}")
                # في حال فشل التحميل المحلي، نرفع خطأ بدلاً من المحاولة عبر الإنترنت لتجنب التعليق
                # إذا أراد المستخدم التحميل، يجب عليه توفير الإنترنت وتشغيله مرة واحدة، أو تحميله يدوياً
                raise RuntimeError("النموذج غير موجود محلياً ويتم العمل في وضع عدم الاتصال.")

            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"تم تحميل النموذج بنجاح. البعد: {self.dimension}")
            
        except ImportError:
            logger.error("sentence-transformers غير متوفر")
            raise ImportError("يرجى تثبيت sentence-transformers: pip install sentence-transformers")
        except Exception as e:
            logger.error(f"فشل تحميل النموذج: {e}")
            raise
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        توليد Embedding لنص واحد
        
        Args:
            text: النص المراد تضمينه
            
        Returns:
            مصفوفة Embedding
        """
        if not self.model:
            raise RuntimeError("النموذج غير محمّل")
        
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
    
    def embed_texts(self, texts: List[str], batch_size: int = 32,
                    show_progress: bool = True) -> np.ndarray:
        """
        توليد Embeddings لقائمة نصوص
        
        Args:
            texts: قائمة النصوص
            batch_size: حجم الدفعة
            show_progress: عرض شريط التقدم
            
        Returns:
            مصفوفة Embeddings
        """
        if not self.model:
            raise RuntimeError("النموذج غير محمّل")
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        return embeddings
    
    def get_dimension(self) -> int:
        """الحصول على بعد Embedding"""
        return self.dimension if hasattr(self, 'dimension') else 768
