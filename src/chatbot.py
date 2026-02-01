"""
Cybercrime Assistant Chatbot
المساعد الذكي لبلاغات الجرائم الإلكترونية
"""

import uuid
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import logging
import json
import re

logger = logging.getLogger(__name__)


class CybercrimeAssistant:
    """المساعد الذكي للجرائم الإلكترونية"""
    
    def __init__(self, llm_client, hybrid_retriever=None, reranker=None,
                 system_prompt: str = None):
        """
        تهيئة المساعد
        
        Args:
            llm_client: عميل LLM
            hybrid_retriever: المسترجع الهجين
            reranker: معيد الترتيب
            system_prompt: رسالة النظام
        """
        self.llm = llm_client
        self.retriever = hybrid_retriever
        self.reranker = reranker
        self.system_prompt = system_prompt or self._default_system_prompt()
        
        # حالة المحادثة
        self.conversation_id = str(uuid.uuid4())[:8]
        self.messages: List[Dict[str, str]] = []
        self.complaint_data: Dict = {
            "id": self.conversation_id,
            "created_at": datetime.now().isoformat(),
            "crime_type": None,
            "severity": None,
            "platform": None,
            "incident_date": None,
            "description": "",
            "evidence": [],
            "damages": None,
            "user_actions": [],
            "recommendations": []
        }
    
    def _default_system_prompt(self) -> str:
        """رسالة النظام الافتراضية"""
        return """أنت مساعد ذكي متخصص في استقبال بلاغات الجرائم الإلكترونية. مهمتك هي:

1. **استقبال البلاغات**: استمع بعناية لشكوى المستخدم وأظهر التعاطف والدعم.

2. **جمع المعلومات**: اطلب التفاصيل اللازمة بشكل لطيف:
   - نوع الجريمة (اختراق، احتيال، ابتزاز، تصيد، انتحال هوية، تسريب بيانات)
   - تاريخ ووقت الحادثة
   - المنصة أو الخدمة المتأثرة
   - الأضرار المحتملة
   - أي أدلة متوفرة

3. **تقييم الخطورة**: حدد مستوى خطورة الحالة (منخفض، متوسط، عالٍ، حرج).

4. **تقديم الإرشادات**: قدم نصائح فورية لحماية المستخدم.

**قواعد صارمة:**
- أجب بالعربية الفصحى الحديثة فقط
- لا تخترع معلومات
- لا تساعد في أي نشاط إجرامي
- كن داعماً ومتعاطفاً مع الضحايا
- ارفض بوضوح أي طلبات غير قانونية"""
    
    def _build_context(self, query: str, top_k: int = 5) -> str:
        """بناء السياق من قاعدة المعرفة"""
        if not self.retriever:
            return ""
        
        # استرجاع المستندات ذات الصلة
        results = self.retriever.search(query, top_k=top_k * 2)
        
        # إعادة الترتيب إذا توفر Reranker
        if self.reranker and results:
            results = self.reranker.rerank(query, results, top_k=top_k)
        
        if not results:
            return ""
        
        # بناء نص السياق
        context_parts = []
        for text, score, meta in results:
            source = meta.get('filename', 'مصدر غير معروف')
            context_parts.append(f"[من: {source}]\n{text}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def chat(self, user_message: str) -> str:
        """
        معالجة رسالة المستخدم والرد عليها
        
        Args:
            user_message: رسالة المستخدم
            
        Returns:
            رد المساعد
        """
        # بناء السياق من RAG
        context = self._build_context(user_message)
        
        # إضافة رسالة المستخدم للسجل
        self.messages.append({
            "role": "user",
            "content": user_message
        })
        
        # تحديث وصف الشكوى
        self.complaint_data["description"] += f"\n{user_message}"
        
        # بناء الرسائل للـ LLM
        llm_messages = [{"role": "system", "content": self.system_prompt}]
        
        # إضافة السياق إذا وجد
        if context:
            llm_messages.append({
                "role": "system",
                "content": f"السياق المتاح من قاعدة المعرفة:\n\n{context}\n\nاستخدم هذا السياق للإجابة على أسئلة المستخدم."
            })
        
        # إضافة تاريخ المحادثة (آخر 10 رسائل)
        llm_messages.extend(self.messages[-10:])
        
        # الحصول على الرد
        try:
            response = self.llm.chat(llm_messages)
        except Exception as e:
            logger.error(f"فشل الحصول على رد: {e}")
            response = "عذراً، حدث خطأ تقني. يرجى المحاولة مرة أخرى."
        
        # إضافة الرد للسجل
        self.messages.append({
            "role": "assistant",
            "content": response
        })
        
        # استخراج معلومات الشكوى
        self._extract_complaint_info(user_message, response)
        
        return response
    
    def _extract_complaint_info(self, user_msg: str, assistant_msg: str):
        """استخراج معلومات الشكوى من المحادثة"""
        text = user_msg.lower()
        
        # الكشف عن نوع الجريمة
        crime_keywords = {
            "account_hack": ["اختراق", "هاكر", "سرقة حساب", "اخترق"],
            "financial_fraud": ["احتيال", "نصب", "سرقة مال", "تحويل"],
            "phishing": ["تصيد", "رسالة مزيفة", "رابط مشبوه"],
            "identity_theft": ["انتحال", "هوية مزورة", "تزوير"],
            "blackmail": ["ابتزاز", "تهديد", "صور", "فضيحة"],
            "data_breach": ["تسريب", "بيانات", "معلومات شخصية"]
        }
        
        for crime_type, keywords in crime_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    self.complaint_data["crime_type"] = crime_type
                    break
        
        # الكشف عن المنصة
        platforms = ["فيسبوك", "تويتر", "انستغرام", "واتساب", "سناب", 
                    "تيك توك", "يوتيوب", "جيميل", "بنك", "بطاقة"]
        for platform in platforms:
            if platform in text:
                self.complaint_data["platform"] = platform
                break
        
        # تقييم الخطورة
        high_severity = ["ابتزاز", "تهديد", "سرقة مال", "احتيال مالي"]
        critical_severity = ["تهديد بالقتل", "ابتزاز جنسي", "اختراق بنك"]
        
        for keyword in critical_severity:
            if keyword in text:
                self.complaint_data["severity"] = "critical"
                break
        
        if not self.complaint_data["severity"]:
            for keyword in high_severity:
                if keyword in text:
                    self.complaint_data["severity"] = "high"
                    break
        
        if not self.complaint_data["severity"]:
            self.complaint_data["severity"] = "medium"
    
    def get_complaint_summary(self) -> Dict:
        """الحصول على ملخص الشكوى"""
        return {
            "id": self.complaint_data["id"],
            "created_at": self.complaint_data["created_at"],
            "crime_type": self.complaint_data["crime_type"],
            "severity": self.complaint_data["severity"],
            "platform": self.complaint_data["platform"],
            "messages_count": len(self.messages),
            "description": self.complaint_data["description"][:500]
        }
    
    def get_full_complaint(self) -> Dict:
        """الحصول على بيانات الشكوى الكاملة"""
        return {
            **self.complaint_data,
            "conversation": self.messages
        }
    
    def reset_conversation(self):
        """إعادة تعيين المحادثة"""
        self.conversation_id = str(uuid.uuid4())[:8]
        self.messages = []
        self.complaint_data = {
            "id": self.conversation_id,
            "created_at": datetime.now().isoformat(),
            "crime_type": None,
            "severity": None,
            "platform": None,
            "incident_date": None,
            "description": "",
            "evidence": [],
            "damages": None,
            "user_actions": [],
            "recommendations": []
        }
