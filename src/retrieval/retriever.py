from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class RAGRetriever:
    """Sistema de recuperación para producción"""
    
    def __init__(self, vector_store):
        self.vector_store = vector_store
    
    def retrieve(self, query: str, n_results: int = 3) -> Dict[str, Any]:
        """Recupera documentos relevantes para una consulta"""
        try:
            # Usar búsqueda con umbral adaptativo
            results = self.vector_store.search_with_fallback(query, n_results)
            
            # Si no hay resultados después de todos los umbrales
            if not results:
                return {
                    "query": query,
                    "results": [],
                    "count": 0,
                    "status": "no_relevant_documents",
                    "message": "No se encontró información relevante en los documentos."
                }
            
            # Filtrar resultados de baja calidad
            filtered_results = self._filter_low_quality_results(query, results)
            
            if not filtered_results:
                # Si el filtrado es muy estricto, mostrar al menos los mejores resultados
                best_results = sorted(results, key=lambda x: x.get('similarity', 0), reverse=True)[:n_results]
                if best_results and best_results[0].get('similarity', 0) >= 0.5:
                    return {
                        "query": query,
                        "results": best_results,
                        "count": len(best_results),
                        "status": "success",
                        "message": f"Encontrados {len(best_results)} documentos relevantes."
                    }
                else:
                    return {
                        "query": query,
                        "results": [],
                        "count": 0,
                        "status": "low_similarity",
                        "message": "La información encontrada no es suficientemente relevante."
                    }
            
            # Obtener el documento con mayor similitud para respuesta directa
            best_document = self._get_best_document(filtered_results)
            
            return {
                "query": query,
                "results": filtered_results,
                "count": len(filtered_results),
                "status": "success",
                "message": f"Encontrados {len(filtered_results)} documentos relevantes.",
                "best_document": best_document
            }
            
        except Exception as e:
            logger.error(f"Error en recuperación: {e}")
            return {
                "query": query,
                "results": [],
                "count": 0,
                "status": "error",
                "error": str(e),
                "message": "Error técnico en la búsqueda."
            }
    
    def _filter_low_quality_results(self, query: str, results: List[Dict]) -> List[Dict]:
        """Filtra resultados de baja similitud o irrelevantes"""
        if not results:
            return []
        
        filtered_results = []
        query_lower = query.lower()
        
        # Palabras clave importantes de la consulta (excluyendo stop words simples)
        stop_words = {'que', 'es', 'la', 'el', 'de', 'en', 'y', 'o', 'un', 'una', 'los', 'las', 'eran'}
        query_terms = [word for word in query_lower.split() if word not in stop_words and len(word) > 2]
        
        for result in results:
            content_lower = result['content'].lower()
            similarity = result.get('similarity', 0)
            
            # CRITERIO 1: Buena similitud (> 0.6) - incluir siempre
            if similarity > 0.6:
                filtered_results.append(result)
                continue
            
            # CRITERIO 2: Similitud aceptable (0.5-0.6) 
            if similarity >= 0.5:
                filtered_results.append(result)
                continue
            
            # CRITERIO 3: Para similitud baja (0.4-0.5), verificar términos clave
            if similarity >= 0.4 and query_terms:
                matching_terms = sum(1 for term in query_terms if term in content_lower)
                # Requerir al menos 1 término coincidente para similitud baja
                if matching_terms >= 1:
                    filtered_results.append(result)
        
        # Ordenar por similitud
        filtered_results.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        
        logger.info(f"Filtrado: {len(filtered_results)}/{len(results)} documentos considerados relevantes")
        
        return filtered_results
    
    def _get_best_document(self, results: List[Dict]) -> Dict[str, Any]:
        """Obtiene el documento con mayor similitud"""
        if not results:
            return {}
        
        best_doc = max(results, key=lambda x: x.get('similarity', 0))
        
        return {
            'content': best_doc['content'],
            'similarity': best_doc.get('similarity', 0),
            'metadata': best_doc.get('metadata', {})
        }
    
    def get_direct_answer(self, query: str, n_results: int = 3) -> str:
        """Obtiene respuesta directa basada en el documento más relevante"""
        retrieval_result = self.retrieve(query, n_results)
        
        if retrieval_result["status"] in ["no_relevant_documents", "low_similarity"]:
            return retrieval_result["message"]
        
        if retrieval_result["status"] == "error":
            return f"Error: {retrieval_result.get('error', 'Desconocido')}"
        
        if not retrieval_result.get("best_document"):
            return "No se pudo determinar la información más relevante."
        
        best_doc = retrieval_result["best_document"]
        similarity = best_doc.get('similarity', 0)
        content = best_doc['content']
        
        # Limpiar y formatear la respuesta
        cleaned_content = self._clean_content(content)
        
        # Determinar confianza basada en similitud
        if similarity >= 0.8:
            confidence = "alta confianza"
        elif similarity >= 0.6:
            confidence = "buena confianza"
        else:
            confidence = "confianza moderada"
        
        return f"Basado en la información más relevante ({confidence}, similitud: {similarity:.2f}):\n\n{cleaned_content}"
    
    def _clean_content(self, content: str) -> str:
        """Limpia y formatea el contenido para respuesta directa"""
        # Eliminar espacios múltiples y saltos de línea excesivos
        import re
        cleaned = re.sub(r'\s+', ' ', content)
        cleaned = cleaned.strip()
        
        # Limitar longitud si es muy extenso
        if len(cleaned) > 500:
            cleaned = cleaned[:500] + "..."
        
        return cleaned
    
    def get_context(self, query: str, n_results: int = 3) -> str:
        """Obtiene contexto formateado para LLM"""
        retrieval_result = self.retrieve(query, n_results)
        
        if retrieval_result["status"] in ["no_relevant_documents", "low_similarity"]:
            return retrieval_result["message"]
        
        if retrieval_result["status"] == "error":
            return f"Error: {retrieval_result.get('error', 'Desconocido')}"
        
        if not retrieval_result["results"]:
            return "No se encontró información relevante."
        
        context_parts = []
        for i, result in enumerate(retrieval_result["results"]):
            similarity = result.get('similarity', 0)
            # Mostrar más contenido para mejor contexto
            content_preview = result['content'][:300] + "..." if len(result['content']) > 300 else result['content']
            context_parts.append(f"[Documento {i+1} - Similitud: {similarity:.2f}]: {content_preview}")
        
        return "\n\n".join(context_parts)