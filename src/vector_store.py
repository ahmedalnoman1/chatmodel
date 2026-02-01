"""
FAISS Vector Store
مخزن المتجهات باستخدام FAISS
"""

import os
import pickle
import numpy as np
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """مخزن المتجهات باستخدام FAISS"""
    
    def __init__(self, dimension: int = 768):
        """
        تهيئة المخزن
        
        Args:
            dimension: بعد المتجهات
        """
        self.dimension = dimension
        self.index = None
        self.documents = []  # قائمة المستندات المخزنة
        self.metadata = []   # البيانات الوصفية
        self._init_index()
    
    def _init_index(self):
        """تهيئة فهرس FAISS"""
        try:
            import faiss
            
            # استخدام IndexFlatIP للتشابه الكوسيني
            self.index = faiss.IndexFlatIP(self.dimension)
            logger.info(f"تم إنشاء فهرس FAISS بالبعد: {self.dimension}")
            
        except ImportError:
            logger.error("faiss-cpu غير متوفر")
            raise ImportError("يرجى تثبيت faiss-cpu: pip install faiss-cpu")
    
    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """تطبيع المتجهات للتشابه الكوسيني"""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1  # تجنب القسمة على صفر
        return vectors / norms
    
    def add_documents(self, texts: List[str], embeddings: np.ndarray,
                      metadata: Optional[List[Dict]] = None):
        """
        إضافة مستندات للمخزن
        
        Args:
            texts: قائمة النصوص
            embeddings: مصفوفة Embeddings
            metadata: البيانات الوصفية (اختياري)
        """
        if len(texts) != len(embeddings):
            raise ValueError("عدد النصوص يجب أن يساوي عدد Embeddings")
        
        # تطبيع المتجهات
        normalized = self._normalize_vectors(embeddings.astype('float32'))
        
        # إضافة للفهرس
        self.index.add(normalized)
        
        # حفظ المستندات والبيانات الوصفية
        self.documents.extend(texts)
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{} for _ in texts])
        
        logger.info(f"تمت إضافة {len(texts)} مستند. الإجمالي: {len(self.documents)}")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        """
        البحث عن أقرب المستندات
        
        Args:
            query_embedding: Embedding الاستعلام
            top_k: عدد النتائج
            
        Returns:
            قائمة (نص، درجة، بيانات وصفية)
        """
        if self.index.ntotal == 0:
            logger.warning("المخزن فارغ")
            return []
        
        # تطبيع الاستعلام
        query = self._normalize_vectors(
            query_embedding.reshape(1, -1).astype('float32')
        )
        
        # البحث
        scores, indices = self.index.search(query, min(top_k, self.index.ntotal))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:  # تجاهل النتائج غير الصالحة
                results.append((
                    self.documents[idx],
                    float(score),
                    self.metadata[idx]
                ))
        
        return results
    
    def save(self, path: str):
        """
        حفظ المخزن على القرص
        
        Args:
            path: مسار الحفظ
        """
        import faiss
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # حفظ الفهرس
        faiss.write_index(self.index, str(path / "index.faiss"))
        
        # حفظ المستندات والبيانات الوصفية
        with open(path / "documents.pkl", 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadata': self.metadata,
                'dimension': self.dimension
            }, f)
        
        logger.info(f"تم حفظ المخزن في: {path}")
    
    def load(self, path: str):
        """
        تحميل المخزن من القرص
        
        Args:
            path: مسار التحميل
        """
        import faiss
        
        path = Path(path)
        
        if not (path / "index.faiss").exists():
            logger.warning(f"لا يوجد فهرس محفوظ في: {path}")
            return
        
        # تحميل الفهرس
        self.index = faiss.read_index(str(path / "index.faiss"))
        
        # تحميل المستندات والبيانات الوصفية
        with open(path / "documents.pkl", 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.metadata = data['metadata']
            self.dimension = data['dimension']
        
        logger.info(f"تم تحميل {len(self.documents)} مستند من: {path}")
    
    def clear(self):
        """مسح المخزن"""
        self._init_index()
        self.documents = []
        self.metadata = []
        logger.info("تم مسح المخزن")
    
    @property
    def count(self) -> int:
        """عدد المستندات المخزنة"""
        return len(self.documents)
