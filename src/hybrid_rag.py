import logging
import numpy as np
from typing import List, Dict, Any
from rapidfuzz import fuzz, process
from .external_embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

class HybridRAG:
    """Sistema RAG híbrido que usa embeddings cuando está disponible, sino fuzzy matching"""
    
    def __init__(self):
        self.embedding_client = HuggingFaceEmbeddings()
        self.documents = []
        self.embeddings = []
        self.metadatas = []
        self.use_embeddings = True  # Intentar usar embeddings primero
        
    def add_documents(self, documents: List[str], metadatas: List[Dict] = None) -> bool:
        """Añade documentos al sistema"""
        try:
            self.documents = documents
            
            # Preparar metadatos
            if metadatas is None:
                self.document_metadata = [{"chunk_id": i, "source": "production"} for i in range(len(documents))]
            else:
                self.document_metadata = metadatas
            
            # Intentar obtener embeddings
            if self.use_embeddings:
                logger.info("Intentando obtener embeddings de Hugging Face...")
                self.embeddings = self.embedding_client.get_embeddings(documents)
                
                # Verificar si los embeddings funcionaron
                valid_embeddings = sum(1 for emb in self.embeddings if emb)
                if valid_embeddings > 0:
                    logger.info(f"✅ {valid_embeddings}/{len(documents)} embeddings obtenidos exitosamente")
                    return True
                else:
                    logger.warning("❌ No se pudieron obtener embeddings, usando fuzzy matching")
                    self.use_embeddings = False
                    return True
            else:
                logger.info("Usando modo fuzzy matching (sin embeddings)")
                return True
                
        except Exception as e:
            logger.error(f"Error cargando documentos: {e}")
            self.use_embeddings = False
            return True
    
    def search(self, query: str, n_results: int = 3, similarity_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Busca documentos usando el método disponible"""
        if self.use_embeddings and self.embeddings:
            return self._semantic_search(query, n_results, similarity_threshold)
        else:
            return self._fuzzy_search(query, n_results, similarity_threshold)
    
    def _semantic_search(self, query: str, n_results: int = 3, similarity_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Búsqueda semántica con embeddings"""
        try:
            logger.info("🔍 Usando búsqueda semántica...")
            
            query_embedding = self.embedding_client.get_embedding(query)
            if not query_embedding:
                logger.warning("No se pudo obtener embedding para consulta, cambiando a fuzzy matching")
                return self._fuzzy_search(query, n_results, similarity_threshold)
            
            results = []
            valid_embeddings_count = 0
            
            for i, doc_embedding in enumerate(self.embeddings):
                if not doc_embedding or len(doc_embedding) != len(query_embedding):
                    continue
                    
                valid_embeddings_count += 1
                similarity = self.embedding_client.cosine_similarity(query_embedding, doc_embedding)
                
                if similarity >= similarity_threshold:
                    results.append({
                        'content': self.documents[i],
                        'metadata': self.document_metadata[i] if i < len(self.document_metadata) else {},
                        'similarity': round(similarity, 3),
                        'index': i,
                        'method': 'semantic'
                    })
            
            results.sort(key=lambda x: x['similarity'], reverse=True)
            logger.info(f"Búsqueda semántica: {len(results)}/{valid_embeddings_count} documentos relevantes")
            return results[:n_results]
            
        except Exception as e:
            logger.error(f"Error en búsqueda semántica: {e}")
            return self._fuzzy_search(query, n_results, similarity_threshold)
    
    def _fuzzy_search(self, query: str, n_results: int = 3, similarity_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Búsqueda con fuzzy matching"""
        try:
            logger.info("🔍 Usando búsqueda fuzzy...")
            
            query_clean = self._preprocess_text(query)
            
            matches = process.extract(
                query_clean,
                [self._preprocess_text(doc[:800]) for doc in self.documents],
                scorer=fuzz.token_sort_ratio,
                limit=min(n_results * 3, len(self.documents))
            )
            
            results = []
            for doc_text, score, index in matches:
                similarity = score / 100
                
                if similarity >= similarity_threshold:
                    results.append({
                        'content': self.documents[index],
                        'metadata': self.document_metadata[index] if index < len(self.document_metadata) else {},
                        'similarity': round(similarity, 3),
                        'index': index,
                        'method': 'fuzzy'
                    })
            
            results.sort(key=lambda x: x['similarity'], reverse=True)
            logger.info(f"Búsqueda fuzzy: {len(results)} documentos encontrados")
            return results[:n_results]
            
        except Exception as e:
            logger.error(f"Error en búsqueda fuzzy: {e}")
            return []
    
    def _preprocess_text(self, text: str) -> str:
        """Limpia y normaliza texto para búsqueda"""
        import re
        if not text:
            return ""
        
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def search_with_fallback(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Búsqueda con múltiples estrategias"""
        # Primero intentar con el método principal
        results = self.search(query, n_results, similarity_threshold=0.4)
        
        if results:
            return results
        
        # Si no hay resultados, bajar el umbral
        results = self.search(query, n_results, similarity_threshold=0.2)
        
        if results:
            logger.info(f"Usando resultados con umbral bajo: {len(results)} documentos")
            return results
        
        return []
    
    def get_document_count(self) -> int:
        return len(self.documents)
    
    def clear_documents(self) -> bool:
        """Limpia todos los documentos"""
        try:
            self.documents = []
            self.embeddings = []
            self.document_metadata = []
            logger.info("✅ Documentos limpiados")
            return True
        except Exception as e:
            logger.error(f"Error limpiando documentos: {e}")
            return False
    
    def get_system_info(self) -> Dict[str, Any]:
        """Obtiene información del sistema"""
        valid_embeddings = sum(1 for emb in self.embeddings if emb)
        return {
            "total_documents": len(self.documents),
            "using_embeddings": self.use_embeddings,
            "valid_embeddings": valid_embeddings,
            "search_method": "semantic" if self.use_embeddings else "fuzzy"
        }