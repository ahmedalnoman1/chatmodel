"""
Arabic Cybercrime RAG Chatbot - Streamlit Interface
واجهة المستخدم الرسومية
"""

import streamlit as st
from pathlib import Path
import sys

# إضافة مسار المشروع
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    LM_STUDIO_BASE_URL, LM_STUDIO_MODEL,
    LLM_TEMPERATURE, LLM_MAX_TOKENS,
    SYSTEM_PROMPT, KNOWLEDGE_BASE_DIR, REPORTS_DIR,
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RETRIEVAL, TOP_K_RERANK,
    CRIME_TYPES, SEVERITY_LEVELS
)

# إعداد الصفحة
st.set_page_config(
    page_title="مساعد بلاغات الجرائم الإلكترونية",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS مخصص للعربية
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700&display=swap');
    
    * {
        font-family: 'Tajawal', sans-serif !important;
    }
    
    .main {
        direction: rtl;
        text-align: right;
    }
    
    .stChatMessage {
        direction: rtl;
        text-align: right;
    }
    
    .stTextInput > div > div > input {
        direction: rtl;
        text-align: right;
    }
    
    .sidebar .sidebar-content {
        direction: rtl;
    }
    
    .status-low { color: #28a745; }
    .status-medium { color: #ffc107; }
    .status-high { color: #fd7e14; }
    .status-critical { color: #dc3545; font-weight: bold; }
    
    .chat-container {
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    h1, h2, h3 {
        direction: rtl;
        text-align: right;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_system():
    """تهيئة مكونات النظام"""
    from src.arabic_processor import ArabicProcessor
    from src.document_loader import DocumentLoader
    from src.llm_client import LMStudioClient
    from src.chatbot import CybercrimeAssistant
    from src.pdf_reporter import PDFReporter
    
    with st.spinner("جاري تحميل النظام..."):
        # المعالج العربي
        arabic_processor = ArabicProcessor()
        
        # محمّل المستندات
        doc_loader = DocumentLoader(arabic_processor)
        
        # تحميل نماذج RAG (اختياري)
        hybrid = None
        reranker = None
        
        # محاولة تحميل نماذج RAG - يمكن تفعيلها لاحقاً
        try:
            from src.embeddings import ArabicEmbeddings
            from src.vector_store import FAISSVectorStore
            from src.bm25_retriever import BM25Retriever
            from src.hybrid_retriever import HybridRetriever
            from src.reranker import ArabicReranker
            
            embeddings = ArabicEmbeddings()
            vector_store = FAISSVectorStore(dimension=embeddings.get_dimension())
            bm25 = BM25Retriever(arabic_processor=arabic_processor)
            hybrid = HybridRetriever(vector_store, bm25, embeddings)
            reranker = ArabicReranker()
            st.success("✅ تم تحميل نماذج RAG")
        except Exception as e:
            st.warning(f"⚠️ يعمل في الوضع غير المتصل (Offline Mode) - خدمة RAG غير متاحة لعدم توفر النماذج محلياً.")
            print(f"RAG Error: {e}")  # For debugging
        
        # عميل LLM
        llm = LMStudioClient(
            base_url=LM_STUDIO_BASE_URL,
            model=LM_STUDIO_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS
        )
        
        # المساعد
        assistant = CybercrimeAssistant(
            llm_client=llm,
            hybrid_retriever=hybrid,
            reranker=reranker,
            system_prompt=SYSTEM_PROMPT
        )
        
        # مولد التقارير
        pdf_reporter = PDFReporter(output_dir=str(REPORTS_DIR))
        
        return {
            "arabic_processor": arabic_processor,
            "doc_loader": doc_loader,
            "hybrid": hybrid,
            "assistant": assistant,
            "pdf_reporter": pdf_reporter,
            "llm": llm
        }


def main():
    """الدالة الرئيسية"""
    
    # العنوان
    st.title("🛡️ مساعد بلاغات الجرائم الإلكترونية")
    st.markdown("---")
    
    # تهيئة النظام
    try:
        components = initialize_system()
    except Exception as e:
        st.error(f"فشل تهيئة النظام: {e}")
        st.info("تأكد من تثبيت جميع التبعيات: `pip install -r requirements.txt`")
        return
    
    # الشريط الجانبي
    with st.sidebar:
        st.header("⚙️ الإعدادات")
        
        # حالة LM Studio
        if components["llm"].is_available():
            st.success("✅ LM Studio متصل")
        else:
            st.error("❌ LM Studio غير متصل")
            st.info("تأكد من تشغيل LM Studio على المنفذ 1234")
        
        st.markdown("---")
        
        # معلومات الجلسة
        st.header("📋 معلومات الجلسة")
        assistant = components["assistant"]
        
        complaint = assistant.get_complaint_summary()
        st.write(f"**رقم البلاغ:** {complaint['id']}")
        
        if complaint.get("crime_type"):
            crime_ar = CRIME_TYPES.get(complaint["crime_type"], complaint["crime_type"])
            st.write(f"**نوع الجريمة:** {crime_ar}")
        
        if complaint.get("severity"):
            severity_info = SEVERITY_LEVELS.get(complaint["severity"], {})
            severity_ar = severity_info.get("ar", complaint["severity"])
            color = severity_info.get("color", "#000")
            st.markdown(f"**مستوى الخطورة:** <span style='color:{color}'>{severity_ar}</span>", 
                       unsafe_allow_html=True)
        
        if complaint.get("platform"):
            st.write(f"**المنصة:** {complaint['platform']}")
        
        st.markdown("---")
        
        # أزرار التحكم
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 محادثة جديدة", use_container_width=True):
                assistant.reset_conversation()
                if "messages" in st.session_state:
                    del st.session_state["messages"]
                st.rerun()
        
        with col2:
            if st.button("📄 تصدير PDF", use_container_width=True):
                try:
                    full_complaint = assistant.get_full_complaint()
                    pdf_path = components["pdf_reporter"].generate_report(full_complaint)
                    st.success(f"تم إنشاء التقرير")
                    
                    # تحميل الملف
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            label="⬇️ تحميل التقرير",
                            data=f,
                            file_name=Path(pdf_path).name,
                            mime="application/pdf"
                        )
                except Exception as e:
                    st.error(f"فشل إنشاء التقرير: {e}")
        
        st.markdown("---")
        
        # إدارة قاعدة المعرفة
        st.header("📚 قاعدة المعرفة")
        
        uploaded_files = st.file_uploader(
            "رفع ملفات",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True
        )
        
        if uploaded_files and st.button("📤 إضافة للقاعدة"):
            hybrid = components["hybrid"]
            doc_loader = components["doc_loader"]
            
            if hybrid:
                for file in uploaded_files:
                    # حفظ الملف مؤقتاً
                    temp_path = KNOWLEDGE_BASE_DIR / file.name
                    with open(temp_path, "wb") as f:
                        f.write(file.getbuffer())
                    
                    # تحميل ومعالجة
                    doc = doc_loader.load_document(str(temp_path))
                    if doc["text"]:
                        processor = components["arabic_processor"]
                        chunks = processor.chunk_text(
                            doc["text"], 
                            chunk_size=CHUNK_SIZE, 
                            overlap=CHUNK_OVERLAP
                        )
                        hybrid.add_documents(chunks, [doc["metadata"]] * len(chunks))
                        st.success(f"تمت إضافة: {file.name}")
            else:
                st.warning("نظام RAG غير متاح")
    
    # المحادثة الرئيسية
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # رسالة ترحيب
        welcome = """مرحباً بك في مساعد بلاغات الجرائم الإلكترونية. 🛡️

أنا هنا لمساعدتك في الإبلاغ عن أي جريمة إلكترونية تعرضت لها. يمكنني مساعدتك في:

• **اختراق الحسابات** - حسابات البريد ومنصات التواصل
• **الاحتيال المالي** - عمليات النصب الإلكتروني
• **رسائل التصيد** - الروابط والرسائل المشبوهة
• **انتحال الهوية** - استخدام هويتك بشكل غير مصرح
• **الابتزاز الإلكتروني** - التهديد والابتزاز
• **تسريب البيانات** - الوصول غير المصرح لبياناتك

**كيف يمكنني مساعدتك اليوم؟**"""
        st.session_state.messages.append({
            "role": "assistant",
            "content": welcome
        })
    
    # عرض الرسائل
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # إدخال المستخدم
    if prompt := st.chat_input("اكتب رسالتك هنا..."):
        # إضافة رسالة المستخدم
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # الحصول على الرد
        with st.chat_message("assistant"):
            with st.spinner("جاري التفكير..."):
                try:
                    response = components["assistant"].chat(prompt)
                except ConnectionError:
                    response = "⚠️ تعذر الاتصال بـ LM Studio. تأكد من تشغيله والمحاولة مرة أخرى."
                except Exception as e:
                    response = f"⚠️ حدث خطأ: {str(e)}"
            
            st.markdown(response)
        
        # حفظ الرد
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
