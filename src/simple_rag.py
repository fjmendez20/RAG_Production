import logging
import re
from typing import List, Dict, Any
from rapidfuzz import fuzz, process

logger = logging.getLogger(__name__)

class SimpleRAG:
    """RAG ultra-liviano usando fuzzy matching sin embeddings"""
    
    def __init__(self):
        self.documents = []
        self.document_metadata = []
    
    def add_documents(self, documents: List[str], metadatas: List[Dict] = None) -> bool:
        """Añade documentos a la memoria"""
        try:
            self.documents = documents
            
            # Preparar metadatos
            if metadatas is None:
                self.document_metadata = [{"chunk_id": i, "source": "production"} for i in range(len(documents))]
            else:
                self.document_metadata = metadatas
            
            logger.info(f"✅ {len(documents)} documentos cargados en memoria")
            return True
        except Exception as e:
            logger.error(f"Error cargando documentos: {e}")
            return False
    
    def search(self, query: str, n_results: int = 3, similarity_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Búsqueda usando fuzzy matching"""
        try:
            if not self.documents:
                logger.info("No hay documentos cargados")
                return []
            
            # Preprocesar query
            query_clean = self._preprocess_text(query)
            logger.info(f"Buscando: '{query}' (limpio: '{query_clean}')")
            
            results = []
            
            # Usar rapidfuzz para búsqueda eficiente
            matches = process.extract(
                query_clean,
                [self._preprocess_text(doc[:800]) for doc in self.documents],  # Limitar para eficiencia
                scorer=fuzz.token_sort_ratio,
                limit=min(n_results * 3, len(self.documents))
            )
            
            for doc_text, score, index in matches:
                similarity = score / 100  # Convertir a 0-1
                
                if similarity >= similarity_threshold:
                    results.append({
                        'content': self.documents[index],
                        'metadata': self.document_metadata[index] if index < len(self.document_metadata) else {},
                        'similarity': round(similarity, 3),
                        'index': index
                    })
            
            # Ordenar por similitud
            results.sort(key=lambda x: x['similarity'], reverse=True)
            
            logger.info(f"Búsqueda completada: {len(results)} documentos encontrados")
            return results[:n_results]
            
        except Exception as e:
            logger.error(f"Error en búsqueda: {e}")
            return []
    
    def search_with_fallback(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Búsqueda con múltiples estrategias de fallback"""
        # Primera búsqueda con umbral normal
        results = self.search(query, n_results, similarity_threshold=0.4)
        
        if results:
            return results
        
        # Segunda búsqueda con umbral más bajo
        results = self.search(query, n_results, similarity_threshold=0.2)
        
        if results:
            logger.info(f"Usando resultados con umbral bajo: {len(results)} documentos")
            return results
        
        # Último intento: búsqueda por palabras clave
        keyword_results = self._keyword_search(query, n_results)
        if keyword_results:
            logger.info(f"Usando resultados por palabras clave: {len(keyword_results)} documentos")
            return keyword_results
        
        return []
    
    def _keyword_search(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Búsqueda simple por palabras clave"""
        try:
            query_words = set(self._preprocess_text(query).split())
            if not query_words:
                return []
            
            results = []
            for i, doc in enumerate(self.documents):
                doc_words = set(self._preprocess_text(doc).split())
                common_words = query_words.intersection(doc_words)
                
                if common_words:
                    # Calcular score basado en palabras en común
                    similarity = len(common_words) / len(query_words)
                    
                    results.append({
                        'content': doc,
                        'metadata': self.document_metadata[i] if i < len(self.document_metadata) else {},
                        'similarity': round(similarity, 3),
                        'index': i,
                        'match_type': 'keyword'
                    })
            
            results.sort(key=lambda x: x['similarity'], reverse=True)
            return results[:n_results]
            
        except Exception as e:
            logger.error(f"Error en búsqueda por keywords: {e}")
            return []
    
    def _preprocess_text(self, text: str) -> str:
        """Limpia y normaliza texto para búsqueda"""
        if not text:
            return ""
        
        # Convertir a minúsculas
        text = text.lower().strip()
        
        # Eliminar caracteres especiales pero mantener letras, números y espacios
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Eliminar espacios múltiples
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def get_document_count(self) -> int:
        return len(self.documents)
    
    def clear_documents(self) -> bool:
        """Limpia todos los documentos"""
        try:
            self.documents = []
            self.document_metadata = []
            logger.info("✅ Documentos limpiados")
            return True
        except Exception as e:
            logger.error(f"Error limpiando documentos: {e}")
            return False