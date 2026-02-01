"""
Arabic Reranker
وحدة إعادة ترتيب النتائج باستخدام BGE Reranker
"""

from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class ArabicReranker:
    """إعادة ترتيب نتائج البحث للنصوص العربية"""
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3",
                 device: str = "cpu"):
        """
        تهيئة Reranker
        
        Args:
            model_name: اسم النموذج
            device: الجهاز (cpu/cuda)
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """تحميل النموذج"""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch
            
            logger.info(f"جاري تحميل نموذج Reranker: {self.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            )
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("تم تحميل Reranker بنجاح")
            
        except ImportError:
            logger.error("transformers غير متوفر")
            raise ImportError("يرجى تثبيت transformers: pip install transformers")
        except Exception as e:
            logger.error(f"فشل تحميل Reranker: {e}")
            raise
    
    def rerank(self, query: str, 
               documents: List[Tuple[str, float, Dict]], 
               top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        """
        إعادة ترتيب المستندات بناءً على الاستعلام
        
        Args:
            query: نص الاستعلام
            documents: قائمة (نص، درجة، بيانات وصفية)
            top_k: عدد النتائج النهائية
            
        Returns:
            المستندات مرتبة حسب الأهمية
        """
        if not documents:
            return []
        
        if not self.model:
            logger.warning("Reranker غير محمّل، إرجاع النتائج كما هي")
            return documents[:top_k]
        
        try:
            import torch
            
            # تحضير الأزواج (query, document)
            pairs = [[query, doc[0]] for doc in documents]
            
            # Tokenize
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)
            
            # الحصول على الدرجات
            with torch.no_grad():
                scores = self.model(**inputs).logits.squeeze(-1)
                if len(scores.shape) == 0:
                    scores = scores.unsqueeze(0)
                scores = scores.cpu().numpy()
            
            # دمج الدرجات مع المستندات
            reranked = []
            for i, (text, _, meta) in enumerate(documents):
                reranked.append((text, float(scores[i]), meta))
            
            # ترتيب تنازلي
            reranked.sort(key=lambda x: x[1], reverse=True)
            
            return reranked[:top_k]
            
        except Exception as e:
            logger.error(f"فشل إعادة الترتيب: {e}")
            return documents[:top_k]
    
    def compute_score(self, query: str, document: str) -> float:
        """
        حساب درجة التطابق بين استعلام ومستند
        
        Args:
            query: نص الاستعلام
            document: نص المستند
            
        Returns:
            درجة التطابق
        """
        result = self.rerank(query, [(document, 0, {})], top_k=1)
        return result[0][1] if result else 0.0
