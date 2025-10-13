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
                metadata={"hnsw:space": "cosine"}  # Fuerza cosine similarity
            )
            logger.info(f"Nueva colección creada: {self.config.vector_store.collection_name}")
            return collection
    
    def add_documents(self, documents: List[str], metadatas: List[Dict] = None) -> bool:
        """Añade documentos a la base vectorial - DEJA QUE CHROMADB MANEJE LOS EMBEDDINGS"""
        try:
            if not documents:
                logger.warning("No hay documentos para añadir")
                return False
            
            logger.info(f"Procesando {len(documents)} documentos...")
            
            # Preparar metadatas e IDs
            if metadatas is None:
                metadatas = [{"source": "production", "chunk_id": i} for i in range(len(documents))]
            
            ids = [f"doc_{i}_{hash(doc[:50])}" for i, doc in enumerate(documents)]
            
            # DEJA QUE CHROMADB GENERE LOS EMBEDDINGS AUTOMÁTICAMENTE
            logger.info("Añadiendo documentos a la base vectorial...")
            self.collection.add(
                documents=documents,  # Solo pasa los documentos
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"✅ Añadidos {len(documents)} documentos a la base vectorial")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error añadiendo documentos: {e}")
            return False
    
    def search(self, query: str, n_results: int = None, similarity_threshold: float = None) -> List[Dict[str, Any]]:
        """Busca documentos similares a la consulta"""
        if n_results is None:
            n_results = self.config.search.default_n_results
        if similarity_threshold is None:
            similarity_threshold = self.config.search.similarity_threshold
        try:
            logger.info(f"Buscando: '{query}'")
            
            # DEJA QUE CHROMADB MANEJE LA BÚSQUEDA CON SU PROPIO EMBEDDING
            search_results = self.collection.query(
                query_texts=[query],
                n_results=n_results * 2,  # Buscar más para filtrar después
                include=["documents", "metadatas", "distances"]
            )
            
            # Si no hay resultados, retornar lista vacía
            if not search_results['documents'] or not search_results['documents'][0]:
                logger.info("No se encontraron documentos")
                return []
            
            # Filtrar por similitud
            filtered_results = []
            documents = search_results['documents'][0]
            metadatas = search_results['metadatas'][0]
            distances = search_results['distances'][0] if search_results['distances'] else []
            
            for i in range(len(documents)):
                # Convertir distancia a similitud (cosine distance -> similarity)
                similarity = 1 - distances[i] if distances else 1.0
                
                if similarity >= similarity_threshold:
                    filtered_results.append({
                        'content': documents[i],
                        'metadata': metadatas[i] if i < len(metadatas) else {},
                        'similarity': round(similarity, 3),
                        'distance': distances[i] if distances else None
                    })
            
            # Ordenar por similitud
            filtered_results.sort(key=lambda x: x['similarity'], reverse=True)
            
            logger.info(f"Búsqueda con umbral {similarity_threshold}: {len(filtered_results)}/{len(documents)} documentos relevantes")
            
            return filtered_results[:n_results]
            
        except Exception as e:
            logger.error(f"Error en búsqueda: {e}")
            return []
    
    def search_with_fallback(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Búsqueda con umbral adaptativo"""
        # Umbrales progresivos (más permisivos)
        thresholds = [0.6, 0.5, 0.4, 0.3]
        
        for threshold in thresholds:
            results = self.search(query, n_results, similarity_threshold=threshold)
            if results:
                logger.info(f"✅ Encontrados {len(results)} documentos con umbral {threshold}")
                return results
        
        # Último intento: devolver los mejores sin importar el umbral
        try:
            search_results = self.collection.query(
                query_texts=[query],
                n_results=min(n_results, 3),
                include=["documents", "metadatas", "distances"]
            )
            
            if search_results['documents'] and search_results['documents'][0]:
                documents = search_results['documents'][0]
                metadatas = search_results['metadatas'][0]
                distances = search_results['distances'][0] if search_results['distances'] else []
                
                results = []
                for i in range(len(documents)):
                    similarity = 1 - distances[i] if distances else 1.0
                    results.append({
                        'content': documents[i],
                        'metadata': metadatas[i] if i < len(metadatas) else {},
                        'similarity': round(similarity, 3),
                        'distance': distances[i] if distances else None
                    })
                
                results.sort(key=lambda x: x['similarity'], reverse=True)
                logger.info(f"✅ Devolviendo {len(results)} documentos (sin umbral)")
                return results
                
        except Exception as e:
            logger.error(f"Error en búsqueda de fallback: {e}")
        
        logger.info("❌ No se encontraron documentos")
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
            self.client.delete_collection(self.config.vector_store.collection_name)
            self.collection = self._get_or_create_collection()
            logger.info("✅ Colección limpiada")
            return True
        except Exception as e:
            logger.error(f"Error limpiando colección: {e}")
            return False