import logging
import sys
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Añadir el directorio src al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config.settings import AppConfig
from src.document_processing.loaders_prod import ProductionDocumentLoader
from src.document_processing.splitters import DocumentSplitter
from src.embeddings.vector_store import VectorStoreManager
from src.retrieval.retriever import RAGRetriever
from src.llm.external_llm import ExternalLLM

# Suprimir logs de telemetría de ChromaDB
logging.getLogger('chromadb.telemetry.product.posthog').setLevel(logging.CRITICAL)
logging.getLogger('chromadb.telemetry.product').setLevel(logging.CRITICAL)

# Tu configuración actual de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

# Inicializar FastAPI
app = FastAPI(
    title="RAG Production API",
    description="API de producción para sistema RAG",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelos Pydantic
class QueryRequest(BaseModel):
    question: str
    n_results: int = 4  # Cambiado a 4 por defecto
    use_llm: bool = True

class QueryResponse(BaseModel):
    question: str
    answer: str
    retrieved_documents: List[Dict[str, Any]]
    document_count: int
    status: str
    answer_type: str
    llm_used: bool
    processing_time: float

class InitializeRequest(BaseModel):
    document_sources: List[Dict[str, Any]] = None

class SystemInfoResponse(BaseModel):
    document_count: int
    llm_configured: bool
    status: str
    environment: str

# Sistema RAG para producción
class ProductionRAGSystem:
    def __init__(self):
        self.config = AppConfig()
        self.document_loader = ProductionDocumentLoader()
        self.document_splitter = DocumentSplitter()
        self.vector_store = VectorStoreManager(self.config)
        self.retriever = RAGRetriever(self.vector_store)
        self.llm_client = self._setup_llm()
        self.initialized = False
        
        logger.info("✅ Sistema RAG de producción inicializado")
    
    def _setup_llm(self):
        """Configura el cliente LLM"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("❌ OPENAI_API_KEY no configurada - LLM no disponible")
            return None
        
        try:
            client = ExternalLLM(api_key=api_key)
            if client.is_configured():
                logger.info("🤖 LLM configurado para producción")
                return client
            return None
        except Exception as e:
            logger.error(f"Error configurando LLM: {e}")
            return None
    
    def initialize(self, document_sources: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Inicializa el sistema con documentos desde GitHub Pages"""
        try:
            # Limpiar colección existente primero
            self.vector_store.clear_collection()
            
            if document_sources is None:
                # CONFIGURACIÓN AUTOMÁTICA - DETECTA TODOS LOS DOCUMENTOS
                document_sources = [
                    {
                        "type": "github_pages_auto",  # ← NUEVO TIPO PARA DETECCIÓN AUTOMÁTICA
                        "base_url": "https://fjmendez20.github.io/Documentos_RAG",
                        "file_pattern": "doc*.pdf",   # ← PATRÓN PARA doc1.pdf, doc2.pdf, etc.
                        "max_files": 50               # ← MÁXIMO NÚMERO DE ARCHIVOS A BUSCAR
                    }
                ]
            
            all_documents = []
            for source in document_sources:
                documents = self.document_loader.load_from_source(source)
                all_documents.extend(documents)
            
            if not all_documents:
                return {"success": False, "message": "No se pudieron cargar documentos desde las fuentes"}
            
            # Procesar documentos
            logger.info(f"Dividiendo {len(all_documents)} documentos en chunks...")
            chunks = self.document_splitter.split_documents(all_documents)
            
            logger.info(f"Agregando {len(chunks)} chunks a la base vectorial...")
            success = self.vector_store.add_documents(chunks)
            
            if success:
                self.initialized = True
                return {
                    "success": True,
                    "documents_loaded": len(all_documents),
                    "chunks_created": len(chunks),
                    "message": f"Sistema inicializado con {len(chunks)} chunks desde {len(all_documents)} documentos"
                }
            else:
                return {"success": False, "message": "Error almacenando documentos"}
                
        except Exception as e:
            logger.error(f"Error en inicialización: {e}")
            return {"success": False, "message": f"Error: {str(e)}"}
    
    def query(self, question: str, n_results: int = 4, use_llm: bool = True) -> Dict[str, Any]:
        """Realiza una consulta al sistema"""
        import time
        start_time = time.time()
        
        try:
            if not self.initialized:
                # Auto-inicializar si no está inicializado
                logger.info("Sistema no inicializado, auto-inicializando...")
                self.initialize()
            
            # Recuperar documentos usando el método con fallback
            retrieval_result = self.retriever.retrieve_with_fallback(question, n_results)
            status = retrieval_result["status"]
            llm_used = False
            
            if status == "success":
                if use_llm and self.llm_client:
                    context = self.retriever.get_context(question, n_results)
                    llm_result = self.llm_client.generate_response(
                        prompt=question,
                        context=context,
                        question=question,
                        max_tokens=500
                    )
                    
                    if llm_result["success"]:
                        answer = llm_result['answer']
                        answer_type = "llm"
                        llm_used = True
                    else:
                        answer = self.retriever.get_direct_answer(question, n_results)
                        answer_type = "direct_fallback"
                else:
                    answer = self.retriever.get_direct_answer(question, n_results)
                    answer_type = "direct"
            
            elif status == "no_relevant_documents":
                answer = "No encontré información relevante en los documentos. ¿Podrías reformular tu pregunta?"
                answer_type = "no_documents"
            elif status == "low_similarity":
                answer = "Encontré información relacionada pero no exactamente lo que buscas. ¿Quieres que te muestre lo que encontré?"
                answer_type = "low_similarity"
            else:
                answer = f"Error técnico: {retrieval_result.get('error', 'Desconocido')}"
                answer_type = "error"
            
            processing_time = time.time() - start_time
            
            return {
                "question": question,
                "answer": answer,
                "retrieved_documents": retrieval_result["results"],
                "document_count": retrieval_result["count"],
                "status": status,
                "answer_type": answer_type,
                "llm_used": llm_used,
                "processing_time": round(processing_time, 2)
            }
            
        except Exception as e:
            logger.error(f"Error en consulta: {e}")
            processing_time = time.time() - start_time
            return {
                "question": question,
                "answer": "Error técnico procesando la consulta. Por favor, intenta nuevamente.",
                "retrieved_documents": [],
                "document_count": 0,
                "status": "error",
                "answer_type": "error",
                "llm_used": False,
                "processing_time": round(processing_time, 2)
            }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Obtiene información del sistema"""
        info = self.vector_store.get_collection_info()
        return {
            "document_count": info.get('document_count', 0),
            "llm_configured": self.llm_client is not None,
            "status": "initialized" if self.initialized else "not_initialized",
            "environment": os.getenv("RAG_ENVIRONMENT", "production")
        }

# Instancia global del sistema
rag_system = ProductionRAGSystem()

# Endpoints
@app.get("/")
async def root():
    return {"message": "RAG Production API", "status": "running"}

#endpoint temporal 
@app.post("/debug/load-documents")
async def debug_load_documents():
    """Endpoint para debuguear la carga de documentos"""
    try:
        from src.document_processing.loaders_prod import ProductionDocumentLoader
        
        loader = ProductionDocumentLoader()
        
        # Probar conexión con GitHub Pages
        test_result = loader.test_github_pages_connection("https://fjmendez20.github.io/Documentos_RAG")
        
        # Intentar cargar doc1.pdf específicamente
        document_sources = [{
            "type": "github_pages",
            "base_url": "https://fjmendez20.github.io/Documentos_RAG",
            "files": ["doc1.pdf"]
        }]
        
        documents = loader.load_from_source(document_sources[0])
        
        # Analizar el contenido extraído
        content_analysis = []
        for i, doc in enumerate(documents):
            content_analysis.append({
                "document_index": i,
                "content_length": len(doc),
                "content_preview": doc[:500] + "..." if len(doc) > 500 else doc,
                "is_empty": len(doc.strip()) == 0,
                "has_text": any(char.isalpha() for char in doc)
            })
        
        return {
            "test_connection": test_result,
            "documents_loaded": len(documents),
            "content_analysis": content_analysis,
            "raw_documents": documents if len(documents) <= 3 else ["... demasiados documentos para mostrar ..."]
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
async def health_check():
    info = rag_system.get_system_info()
    return {
        "status": "healthy" if info["document_count"] > 0 else "no_documents",
        "system_info": info
    }

@app.get("/system/info", response_model=SystemInfoResponse)
async def get_system_info():
    return rag_system.get_system_info()

@app.post("/initialize")
async def initialize_system(request: InitializeRequest = None):
    """Inicializa el sistema con documentos"""
    if request and request.document_sources:
        result = rag_system.initialize(request.document_sources)
    else:
        result = rag_system.initialize()
    
    if result["success"]:
        return result
    else:
        raise HTTPException(status_code=400, detail=result["message"])

@app.post("/query", response_model=QueryResponse)
async def query_system(request: QueryRequest):
    """Realiza una consulta al sistema RAG"""
    result = rag_system.query(
        question=request.question,
        n_results=request.n_results,
        use_llm=request.use_llm
    )
    return QueryResponse(**result)

@app.post("/clear")
async def clear_system():
    """Limpia la base de datos vectorial"""
    try:
        rag_system.vector_store.clear_collection()
        rag_system.initialized = False
        return {"success": True, "message": "Sistema limpiado correctamente"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error limpiando sistema: {str(e)}")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "api_prod:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )