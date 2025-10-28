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
from src.hybrid_rag import HybridRAG  # ← CORREGIDO: Usar HybridRAG
from src.llm.external_llm import ExternalLLM

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

# Inicializar FastAPI
app = FastAPI(
    title="RAG Híbrido API",
    description="API de producción para sistema RAG con embeddings externos y fuzzy matching",
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
    n_results: int = 3
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
    search_method: str

# Sistema RAG híbrido
class ProductionRAGSystem:
    def __init__(self):
        self.config = AppConfig()
        self.document_loader = ProductionDocumentLoader()
        self.document_splitter = DocumentSplitter()
        self.rag_engine = HybridRAG()  # ← CORREGIDO: Usar HybridRAG
        self.llm_client = self._setup_llm()
        self.initialized = False
        
        logger.info("✅ Sistema RAG híbrido inicializado")
    
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
        """Inicializa el sistema con documentos"""
        try:
            # Limpiar documentos existentes
            self.rag_engine.clear_documents()
            
            if document_sources is None:
                # Configuración automática para GitHub Pages
                document_sources = [
                    {
                        "type": "github_pages",
                        "base_url": "https://fjmendez20.github.io/Documentos_RAG",
                        "files": ["doc1.pdf"]
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
            
            # Usar el RAG híbrido
            logger.info(f"Agregando {len(chunks)} chunks al sistema RAG...")
            success = self.rag_engine.add_documents(chunks)
            
            if success:
                self.initialized = True
                system_info = self.rag_engine.get_system_info()
                return {
                    "success": True,
                    "documents_loaded": len(all_documents),
                    "chunks_created": len(chunks),
                    "system_info": system_info,
                    "message": f"Sistema inicializado con {len(chunks)} chunks usando {system_info['search_method']}"
                }
            else:
                return {"success": False, "message": "Error almacenando documentos"}
                
        except Exception as e:
            logger.error(f"Error en inicialización: {e}")
            return {"success": False, "message": f"Error: {str(e)}"}
    
    def query(self, question: str, n_results: int = 3, use_llm: bool = True) -> Dict[str, Any]:
        """Realiza una consulta al sistema"""
        import time
        start_time = time.time()
        
        try:
            if not self.initialized:
                # Auto-inicializar si no está inicializado
                logger.info("Sistema no inicializado, auto-inicializando...")
                self.initialize()
            
            # Buscar documentos relevantes
            retrieval_result = self.rag_engine.search_with_fallback(question, n_results)
            status = "success" if retrieval_result else "no_relevant_documents"
            llm_used = False
            
            if status == "success":
                if use_llm and self.llm_client:
                    # Preparar contexto para LLM
                    context = "\n\n".join([f"[Documento {i+1} - Similitud: {doc['similarity']:.2f}]: {doc['content'][:400]}..." 
                                         for i, doc in enumerate(retrieval_result)])
                    
                    llm_result = self.llm_client.generate_response(
                        prompt=question,
                        context=context,
                        question=question,
                        max_tokens=400
                    )
                    
                    if llm_result["success"]:
                        answer = llm_result['answer']
                        answer_type = "llm"
                        llm_used = True
                    else:
                        answer = self._get_direct_answer(retrieval_result)
                        answer_type = "direct_fallback"
                else:
                    answer = self._get_direct_answer(retrieval_result)
                    answer_type = "direct"
            else:
                answer = "No encontré información relevante en los documentos. ¿Podrías reformular tu pregunta?"
                answer_type = "no_documents"
            
            processing_time = time.time() - start_time
            
            return {
                "question": question,
                "answer": answer,
                "retrieved_documents": retrieval_result,
                "document_count": len(retrieval_result),
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
                "answer": "Error técnico procesando la consulta.",
                "retrieved_documents": [],
                "document_count": 0,
                "status": "error",
                "answer_type": "error",
                "llm_used": False,
                "processing_time": round(processing_time, 2)
            }
    
    def _get_direct_answer(self, documents: List[Dict[str, Any]]) -> str:
        """Genera respuesta directa basada en documentos"""
        if not documents:
            return "No se encontró información relevante."
        
        best_doc = max(documents, key=lambda x: x['similarity'])
        similarity = best_doc.get('similarity', 0)
        method = best_doc.get('method', 'unknown')
        
        # Determinar confianza
        if similarity >= 0.8:
            confidence = "muy alta confianza"
        elif similarity >= 0.6:
            confidence = "alta confianza"
        elif similarity >= 0.4:
            confidence = "buena confianza" 
        else:
            confidence = "confianza moderada"
        
        content = best_doc['content']
        if len(content) > 500:
            content = content[:500] + "..."
        
        method_text = "búsqueda semántica" if method == "semantic" else "búsqueda por similitud"
        
        return f"Basado en la información más relevante ({confidence}, similitud: {similarity:.2f} - {method_text}):\n\n{content}"
    
    def get_system_info(self) -> Dict[str, Any]:
        """Obtiene información del sistema"""
        rag_info = self.rag_engine.get_system_info()
        return {
            "document_count": rag_info["total_documents"],
            "llm_configured": self.llm_client is not None,
            "status": "initialized" if self.initialized else "not_initialized",
            "environment": os.getenv("RAG_ENVIRONMENT", "production"),
            "search_method": rag_info["search_method"],
            "using_embeddings": rag_info["using_embeddings"]
        }

# Instancia global del sistema
rag_system = ProductionRAGSystem()

# Endpoints
@app.get("/")
async def root():
    return {"message": "RAG Híbrido API", "status": "running"}

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
    """Limpia la base de datos"""
    try:
        rag_system.rag_engine.clear_documents()
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