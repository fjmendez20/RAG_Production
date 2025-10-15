import logging
import numpy as np
from typing import List, Dict, Any
import json
import os
from .external_embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

class LightweightVectorStore:
    """Almacén vectorial liviano que usa embeddings externos"""
    
    def __init__(self, config):
        self.config = config
        self.embedding_client = HuggingFaceEmbeddings()
        self.documents = []
        self.embeddings = []
        self.metadatas = []
        
    def add_documents(self, documents: List[str], metadatas: List[Dict] = None) -> bool:
        """Añade documentos al almacén vectorial"""
        try:
            if not documents:
                logger.warning("No hay documentos para añadir")
                return False
            
            logger.info(f"Obteniendo embeddings para {len(documents)} documentos...")
            
            # Obtener embeddings en lotes pequeños para evitar timeouts
            batch_size = 5
            all_embeddings = []
            
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i + batch_size]
                logger.info(f"Procesando lote {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
                
                batch_embeddings = self.embedding_client.get_embeddings(batch_docs)
                if batch_embeddings:
                    all_embeddings.extend(batch_embeddings)
                else:
                    # Si falla un lote, añadir vectores vacíos
                    all_embeddings.extend([[]] * len(batch_docs))
            
            # Preparar metadatos
            if metadatas is None:
                metadatas = [{"chunk_id": i, "source": "production"} for i in range(len(documents))]
            
            # Almacenar todo en memoria
            self.documents.extend(documents)
            self.embeddings.extend(all_embeddings)
            self.metadatas.extend(metadatas)
            
            logger.info(f"✅ {len(documents)} documentos añadidos al almacén vectorial")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error añadiendo documentos: {e}")
            return False
    
    def search(self, query: str, n_results: int = 3, similarity_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Busca documentos similares a la consulta"""
        try:
            logger.info(f"Buscando: '{query}'")
            
            # Obtener embedding de la consulta
            query_embedding = self.embedding_client.get_embedding(query)
            if not query_embedding:
                logger.error("No se pudo obtener embedding para la consulta")
                return []
            
            # Calcular similitudes
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
                        'metadata': self.metadatas[i] if i < len(self.metadatas) else {},
                        'similarity': round(similarity, 3),
                        'index': i
                    })
            
            # Ordenar por similitud
            results.sort(key=lambda x: x['similarity'], reverse=True)
            
            logger.info(f"Búsqueda completada: {len(results)}/{valid_embeddings_count} documentos relevantes")
            return results[:n_results]
            
        except Exception as e:
            logger.error(f"Error en búsqueda: {e}")
            return []
    
    def search_with_fallback(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Búsqueda con umbrales progresivos"""
        thresholds = [0.6, 0.5, 0.4, 0.3]
        
        for threshold in thresholds:
            results = self.search(query, n_results, similarity_threshold=threshold)
            if results:
                logger.info(f"✅ Encontrados {len(results)} documentos con umbral {threshold}")
                return results
        
        logger.info("❌ No se encontraron documentos con ningún umbral")
        return []
    
    def get_document_count(self) -> int:
        return len(self.documents)
    
    def clear_documents(self) -> bool:
        """Limpia todos los documentos"""
        try:
            self.documents = []
            self.embeddings = []
            self.metadatas = []
            logger.info("✅ Almacén vectorial limpiado")
            return True
        except Exception as e:
            logger.error(f"Error limpiando documentos: {e}")
            return False
    
    def get_store_info(self) -> Dict[str, Any]:
        """Obtiene información del almacén"""
        valid_embeddings = sum(1 for emb in self.embeddings if emb)
        return {
            "total_documents": len(self.documents),
            "total_embeddings": len(self.embeddings),
            "valid_embeddings": valid_embeddings,
            "embedding_quality": f"{valid_embeddings}/{len(self.embeddings)}" if self.embeddings else "N/A"
        }