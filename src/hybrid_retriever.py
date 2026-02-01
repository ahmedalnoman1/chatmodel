"""
Hybrid Retriever
نظام استرجاع هجين يجمع بين Dense و Sparse
"""

from typing import List, Tuple, Dict, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class HybridRetriever:
    """نظام استرجاع هجين يدمج FAISS و BM25"""
    
    def __init__(self, vector_store, bm25_retriever, embeddings,
                 dense_weight: float = 0.6, sparse_weight: float = 0.4):
        """
        تهيئة المسترجع الهجين
        
        Args:
            vector_store: مخزن FAISS
            bm25_retriever: مسترجع BM25
            embeddings: نموذج Embeddings
            dense_weight: وزن البحث الكثيف
            sparse_weight: وزن البحث المتناثر
        """
        self.vector_store = vector_store
        self.bm25 = bm25_retriever
        self.embeddings = embeddings
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        
        # التحقق من صحة الأوزان
        total = dense_weight + sparse_weight
        self.dense_weight /= total
        self.sparse_weight /= total
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float, Dict]]:
        """
        بحث هجين يجمع نتائج Dense و Sparse
        
        Args:
            query: نص الاستعلام
            top_k: عدد النتائج النهائية
            
        Returns:
            قائمة (نص، درجة مدمجة، بيانات وصفية)
        """
        # البحث الكثيف (FAISS)
        query_embedding = self.embeddings.embed_text(query)
        dense_results = self.vector_store.search(query_embedding, top_k * 2)
        
        # البحث المتناثر (BM25)
        sparse_results = self.bm25.search(query, top_k * 2)
        
        # دمج النتائج
        combined = self._merge_results(dense_results, sparse_results)
        
        # ترتيب النتائج النهائية
        combined_sorted = sorted(combined, key=lambda x: x[1], reverse=True)
        
        return combined_sorted[:top_k]
    
    def _merge_results(self, 
                       dense_results: List[Tuple[str, float, Dict]],
                       sparse_results: List[Tuple[str, float, Dict]]
                      ) -> List[Tuple[str, float, Dict]]:
        """
        دمج نتائج البحث الكثيف والمتناثر
        
        Args:
            dense_results: نتائج FAISS
            sparse_results: نتائج BM25
            
        Returns:
            النتائج المدمجة
        """
        # تحويل لقاموس لسهولة الدمج
        merged = {}
        
        # تطبيع درجات Dense
        if dense_results:
            max_dense = max(r[1] for r in dense_results)
            for text, score, meta in dense_results:
                norm_score = score / max_dense if max_dense > 0 else 0
                merged[text] = {
                    'dense_score': norm_score,
                    'sparse_score': 0,
                    'metadata': meta
                }
        
        # تطبيع درجات Sparse ودمجها
        if sparse_results:
            max_sparse = max(r[1] for r in sparse_results)
            for text, score, meta in sparse_results:
                norm_score = score / max_sparse if max_sparse > 0 else 0
                if text in merged:
                    merged[text]['sparse_score'] = norm_score
                else:
                    merged[text] = {
                        'dense_score': 0,
                        'sparse_score': norm_score,
                        'metadata': meta
                    }
        
        # حساب الدرجة النهائية
        results = []
        for text, scores in merged.items():
            final_score = (
                self.dense_weight * scores['dense_score'] +
                self.sparse_weight * scores['sparse_score']
            )
            results.append((text, final_score, scores['metadata']))
        
        return results
    
    def add_documents(self, texts: List[str], metadata: Optional[List[Dict]] = None):
        """
        إضافة مستندات لكلا النظامين
        
        Args:
            texts: قائمة النصوص
            metadata: البيانات الوصفية
        """
        # توليد Embeddings
        embeddings = self.embeddings.embed_texts(texts)
        
        # إضافة لـ FAISS
        self.vector_store.add_documents(texts, embeddings, metadata)
        
        # إضافة لـ BM25
        self.bm25.add_documents(texts, metadata)
        
        logger.info(f"تمت إضافة {len(texts)} مستند للنظام الهجين")
    
    @property
    def count(self) -> int:
        """عدد المستندات"""
        return self.vector_store.count
