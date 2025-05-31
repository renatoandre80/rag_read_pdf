import os
from typing import List, Dict, Any
from dotenv import load_dotenv

# Carrega variáveis do ambiente
load_dotenv()

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document

class RAGSystem:
    def __init__(self):
        """Inicializa o sistema RAG com configurações de segurança."""
        api_key = self._get_api_key()
        os.environ["OPENAI_API_KEY"] = api_key
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        self.vector_db = None
        self.qa_chain = None

    def _get_api_key(self) -> str:
        """Obtém a API key de forma segura."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY não encontrada. "
                "Crie um arquivo .env com sua chave."
            )
        return api_key

    def load_and_process_documents(self, file_path: str) -> List[Document]:
        """Carrega e processa documentos PDF."""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

            loader = PyPDFLoader(file_path)
            pages = loader.load_and_split()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                separators=["\n\n", "\n", " ", ""]
            )
            return text_splitter.split_documents(pages)
        except Exception as e:
            raise RuntimeError(f"Falha ao processar documento: {str(e)}")

    def create_vector_store(self, documents: List[Document]):
        """Cria o banco de dados vetorial."""
        if not documents:
            raise ValueError("Nenhum documento fornecido")

        try:
            self.vector_db = FAISS.from_documents(documents, self.embeddings)
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_db.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True
            )
        except Exception as e:
            raise RuntimeError(f"Falha ao criar vetorstore: {str(e)}")

    def query(self, question: str) -> Dict[str, Any]:
        """Executa consulta no sistema RAG."""
        if not self.qa_chain:
            raise RuntimeError("Banco de dados não inicializado")
        
        if not question or not isinstance(question, str):
            raise ValueError("Pergunta inválida")

        try:
            return self.qa_chain({"query": question})
        except Exception as e:
            raise RuntimeError(f"Falha na consulta: {str(e)}")