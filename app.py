import streamlit as st
from backend import RAGSystem
import tempfile
import os
from dotenv import load_dotenv
import time

# Configurações iniciais
load_dotenv()
st.set_page_config(
    page_title="ChatPDF - Sistema RAG",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="expanded"
)

def load_css():
    st.markdown("""
    <style>
    :root {
        --primary: #FFD700;
        --secondary: #1A1A1A;
        --background: #121212;
        --text: #E0E0E0;
        --accent: #FFD700;
    }
    
    .stApp {
        background-color: var(--background);
        color: var(--text);
        max-width: 900px;
        margin: 0 auto;
    }
    
    .stTextInput input {
        font-size: 16px;
        padding: 12px;
        background-color: #333333;
        color: var(--text);
        border: 1px solid var(--primary);
    }
    
    .stButton>button {
        background-color: var(--primary);
        color: var(--secondary);
        font-weight: bold;
        padding: 12px 24px;
        border-radius: 8px;
    }
    
    .document-card {
        background-color: #1E1E1E;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
        border-left: 4px solid var(--primary);
    }
    
    .sidebar .sidebar-content {
        background-color: #1E1E1E !important;
        border-right: 2px solid var(--primary);
    }
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: var(--primary) !important;
    }
    
    .stFileUploader>div>div:first-child {
        min-height: 120px;
        border: 2px dashed var(--primary);
        background-color: #333333;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    load_css()
    
    st.title("📄 ChatPDF com RAG")
    st.markdown("Faça perguntas sobre seus documentos PDF e receba respostas precisas.")
    
    # Sidebar
    with st.sidebar:
        st.header("Configurações")
        
        uploaded_file = st.file_uploader(
            "Carregue seu PDF",
            type="pdf",
            help="Documentos com texto claro fornecem melhores resultados. Limite 200MB."
        )
        
        st.markdown("---")
        st.markdown("""
        **Como usar:**
        1. Carregue um arquivo PDF
        2. Aguarde o processamento
        3. Faça perguntas sobre o conteúdo
        """)
        st.markdown("---")
        st.markdown("[🔒 Segurança] Sua chave está armazenada com segurança")

    # Verificação da API Key
    if not os.getenv("OPENAI_API_KEY"):
        st.error("Configure sua OPENAI_API_KEY no arquivo .env")
        st.stop()

    # Processamento do documento
    if uploaded_file and ('rag' not in st.session_state or st.session_state.uploaded_file_name != uploaded_file.name):
        with st.spinner("Processando documento..."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                rag = RAGSystem()
                docs = rag.load_and_process_documents(tmp_path)
                rag.create_vector_store(docs)
                
                st.session_state.rag = rag
                st.session_state.uploaded_file_name = uploaded_file.name
                st.success("Documento processado com sucesso!")
                time.sleep(1)
                st.rerun()
                
            except Exception as e:
                st.error(f"Erro: {str(e)}")
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

    # Área de perguntas
    if 'rag' in st.session_state:
        st.subheader("Faça sua pergunta")
        question = st.text_input(
            "Digite sua pergunta:",
            placeholder="Qual é o tema principal?",
            label_visibility="collapsed"
        )

        if question:
            with st.spinner("Buscando resposta..."):
                try:
                    result = st.session_state.rag.query(question)
                    
                    st.markdown("### Resposta")
                    st.info(result["result"])
                    
                    st.markdown("### Referências")
                    cols = st.columns(3)
                    for i, doc in enumerate(result["source_documents"]):
                        with cols[i % 3]:
                            with st.expander(f"Trecho {i+1} (Página {doc.metadata.get('page', 'N/A')})"):
                                st.markdown(
                                    f"<div class='document-card'>{doc.page_content}</div>", 
                                    unsafe_allow_html=True
                                )
                except Exception as e:
                    st.error(f"Erro: {str(e)}")

if __name__ == "__main__":
    main()