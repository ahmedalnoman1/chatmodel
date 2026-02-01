"""
BM25 Retriever
نظام استرجاع BM25 للبحث المتناثر
"""

from typing import List, Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class BM25Retriever:
    """نظام استرجاع BM25 للبحث النصي"""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75, 
                 arabic_processor=None):
        """
        تهيئة BM25
        
        Args:
            k1: معامل تشبع مصطلح
            b: معامل طول المستند
            arabic_processor: معالج النصوص العربية
        """
        self.k1 = k1
        self.b = b
        self.arabic_processor = arabic_processor
        self.bm25 = None
        self.documents = []
        self.metadata = []
        self.tokenized_docs = []
    
    def _tokenize(self, text: str) -> List[str]:
        """تقسيم النص إلى كلمات"""
        if self.arabic_processor:
            return self.arabic_processor.tokenize(text)
        else:
            # تقسيم بسيط
            import re
            text = re.sub(r'[^\w\s]', ' ', text)
            return text.split()
    
    def add_documents(self, texts: List[str], 
                      metadata: Optional[List[Dict]] = None):
        """
        إضافة مستندات للفهرس
        
        Args:
            texts: قائمة النصوص
            metadata: البيانات الوصفية
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("يرجى تثبيت rank-bm25: pip install rank-bm25")
        
        # تقسيم المستندات
        new_tokenized = [self._tokenize(text) for text in texts]
        
        # تحديث القوائم
        self.documents.extend(texts)
        self.tokenized_docs.extend(new_tokenized)
        
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{} for _ in texts])
        
        # إعادة بناء BM25
        self.bm25 = BM25Okapi(self.tokenized_docs, k1=self.k1, b=self.b)
        
        logger.info(f"تمت إضافة {len(texts)} مستند لـ BM25. الإجمالي: {len(self.documents)}")
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        """
        البحث في المستندات
        
        Args:
            query: نص الاستعلام
            top_k: عدد النتائج
            
        Returns:
            قائمة (نص، درجة، بيانات وصفية)
        """
        if not self.bm25 or not self.documents:
            logger.warning("لا توجد مستندات مفهرسة")
            return []
        
        # تقسيم الاستعلام
        query_tokens = self._tokenize(query)
        
        # الحصول على الدرجات
        scores = self.bm25.get_scores(query_tokens)
        
        # ترتيب النتائج
        top_indices = sorted(range(len(scores)), 
                           key=lambda i: scores[i], 
                           reverse=True)[:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # فقط النتائج ذات الدرجات الإيجابية
                results.append((
                    self.documents[idx],
                    float(scores[idx]),
                    self.metadata[idx]
                ))
        
        return results
    
    def clear(self):
        """مسح الفهرس"""
        self.bm25 = None
        self.documents = []
        self.metadata = []
        self.tokenized_docs = []
        logger.info("تم مسح فهرس BM25")
    
    @property
    def count(self) -> int:
        """عدد المستندات المفهرسة"""
        return len(self.documents)
