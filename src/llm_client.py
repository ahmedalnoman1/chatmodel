"""
LM Studio Client
عميل للاتصال بـ LM Studio API
"""

import requests
import json
from typing import List, Dict, Optional, Generator
import logging

logger = logging.getLogger(__name__)


class LMStudioClient:
    """عميل LM Studio API"""
    
    def __init__(self, base_url: str = "http://localhost:1234/v1",
                 model: str = "gemma-3n-E4B-it-text-GGUF",
                 temperature: float = 0.2,
                 max_tokens: int = 1024):
        """
        تهيئة العميل
        
        Args:
            base_url: عنوان LM Studio API
            model: اسم النموذج
            temperature: درجة الحرارة
            max_tokens: الحد الأقصى للرموز
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = 120  # ثانية
    
    def is_available(self) -> bool:
        """التحقق من توفر LM Studio"""
        try:
            response = requests.get(
                f"{self.base_url}/models",
                timeout=5
            )
            return response.status_code == 200
        except:
            return False
    
    def chat(self, messages: List[Dict[str, str]], 
             stream: bool = False) -> str:
        """
        إرسال محادثة والحصول على رد
        
        Args:
            messages: قائمة الرسائل [{"role": "...", "content": "..."}]
            stream: استخدام البث التدفقي
            
        Returns:
            نص الرد
        """
        url = f"{self.base_url}/chat/completions"
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": stream
        }
        
        try:
            if stream:
                return self._stream_chat(url, payload)
            else:
                response = requests.post(
                    url,
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                result = response.json()
                return result['choices'][0]['message']['content']
                
        except requests.exceptions.ConnectionError:
            logger.error("تعذر الاتصال بـ LM Studio. تأكد من تشغيله.")
            raise ConnectionError("LM Studio غير متاح. تأكد من تشغيله على المنفذ 1234")
        except requests.exceptions.Timeout:
            logger.error("انتهت مهلة الاتصال")
            raise TimeoutError("انتهت مهلة الاتصال بـ LM Studio")
        except Exception as e:
            logger.error(f"خطأ في الاتصال: {e}")
            raise
    
    def _stream_chat(self, url: str, payload: Dict) -> Generator[str, None, None]:
        """بث الرد تدفقياً"""
        try:
            response = requests.post(
                url,
                json=payload,
                stream=True,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            full_response = ""
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = line[6:]
                        if data == '[DONE]':
                            break
                        try:
                            chunk = json.loads(data)
                            delta = chunk['choices'][0].get('delta', {})
                            content = delta.get('content', '')
                            if content:
                                full_response += content
                                yield content
                        except json.JSONDecodeError:
                            continue
            
            return full_response
            
        except Exception as e:
            logger.error(f"خطأ في البث: {e}")
            raise
    
    def complete(self, prompt: str) -> str:
        """
        إكمال نص (Completion API)
        
        Args:
            prompt: النص المراد إكماله
            
        Returns:
            النص المكمّل
        """
        url = f"{self.base_url}/completions"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['text']
            
        except Exception as e:
            logger.error(f"خطأ في الإكمال: {e}")
            raise
    
    def get_models(self) -> List[str]:
        """الحصول على قائمة النماذج المتاحة"""
        try:
            response = requests.get(
                f"{self.base_url}/models",
                timeout=5
            )
            response.raise_for_status()
            
            result = response.json()
            return [m['id'] for m in result.get('data', [])]
            
        except:
            return []
