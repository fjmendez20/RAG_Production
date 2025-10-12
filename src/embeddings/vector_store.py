import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import logging
import os

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Gestor de base de datos vectorial para producción"""
    
    def __init__(self, config):
        self.config = config
        self.embedding_model = SentenceTransformer(config.embedding.model_name)
        
        # En producción, usar directorio temporal
        os.makedirs(config.vector_store.persist_directory, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=config.vector_store.persist_directory)
        self.collection = self._get_or_create_collection()
    
    def _get_or_create_collection(self):
        """Obtiene o crea la colección en ChromaDB"""
        try:
            collection = self.client.get_collection(self.config.vector_store.collection_name)
            logger.info(f"Colección existente cargada: {self.config.vector_store.collection_name}")
            return collection
        except Exception:
            collection = self.client.create_collection(
                name=self.config.vector_store.collection_name,
                metadata={"hnsw:space": self.config.vector_store.similarity_metric}
            )
            logger.info(f"Nueva colección creada: {self.config.vector_store.collection_name}")
            return collection
    
    def add_documents(self, documents: List[str], metadatas: List[Dict] = None) -> bool:
        """Añade documentos a la base vectorial"""
        try:
            if not documents:
                logger.warning("No hay documentos para añadir")
                return False
            
            # Generar embeddings
            logger.info(f"Generando embeddings para {len(documents)} documentos...")
            embeddings = self.embedding_model.encode(documents).tolist()
            
            # Preparar metadatas e IDs
            if metadatas is None:
                metadatas = [{"source": "production", "chunk_id": i} for i in range(len(documents))]
            
            ids = [f"chunk_{i}" for i in range(len(documents))]
            
            # Añadir a la colección
            logger.info("Añadiendo documentos a la base vectorial...")
            self.collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"✅ Añadidos {len(documents)} documentos a la base vectorial")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error añadiendo documentos: {e}")
            return False
    
    def search(self, query: str, n_results: int = 3, similarity_threshold: float = 0.6) -> List[Dict[str, Any]]:
        """Busca documentos similares a la consulta con umbral de similitud"""
        try:
            query_embedding = self.embedding_model.encode([query]).tolist()
            
            # Buscar más resultados para tener mejor selección
            search_results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=min(n_results * 3, 20)
            )
            
            # Si no hay resultados, retornar lista vacía
            if not search_results['documents'] or not search_results['documents'][0]:
                return []
            
            # Filtrar por similitud (distancia)
            filtered_results = []
            documents = search_results['documents'][0]
            metadatas = search_results['metadatas'][0]
            distances = search_results['distances'][0] if search_results['distances'] else []
            
            for i in range(len(documents)):
                # Convertir distancia a similitud (1 - distancia para cosine similarity)
                similarity = 1 - distances[i] if distances else 1.0
                
                if similarity >= similarity_threshold:
                    filtered_results.append({
                        'content': documents[i],
                        'metadata': metadatas[i],
                        'similarity': similarity,
                        'distance': distances[i] if distances else None
                    })
            
            # Ordenar por similitud (mayor a menor)
            filtered_results.sort(key=lambda x: x['similarity'], reverse=True)
            
            logger.info(f"Búsqueda con umbral {similarity_threshold}: {len(filtered_results)}/{len(documents)} documentos relevantes")
            
            return filtered_results[:n_results]
            
        except Exception as e:
            logger.error(f"Error en búsqueda: {e}")
            return []
    
    def search_with_fallback(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Búsqueda con umbral adaptativo - más permisivo"""
        logger.info(f"Buscando: '{query}'")
        
        # Umbrales progresivos (más permisivos)
        thresholds = [0.7, 0.6, 0.5, 0.4]
        
        for threshold in thresholds:
            results = self.search(query, n_results, similarity_threshold=threshold)
            if results:
                logger.info(f"✅ Encontrados {len(results)} documentos con umbral {threshold}")
                return results
        
        # Si no encuentra con ningún umbral
        logger.info("❌ No se encontraron documentos con ningún umbral de similitud")
        return []
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Obtiene información sobre la colección"""
        try:
            count = self.collection.count()
            return {
                "collection_name": self.config.vector_store.collection_name,
                "document_count": count,
                "persist_directory": self.config.vector_store.persist_directory
            }
        except Exception as e:
            logger.error(f"Error obteniendo información: {e}")
            return {"error": str(e)}
    
    def clear_collection(self) -> bool:
        """Limpia la colección completa"""
        try:
            # ChromaDB no tiene un método directo para limpiar, así que recreamos
            self.client.delete_collection(self.config.vector_store.collection_name)
            self.collection = self._get_or_create_collection()
            logger.info("✅ Colección limpiada")
            return True
        except Exception as e:
            logger.error(f"Error limpiando colección: {e}")
            return False