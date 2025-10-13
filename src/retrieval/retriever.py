from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class RAGRetriever:
    """Sistema de recuperación para producción"""
    
    def __init__(self, vector_store):
        self.vector_store = vector_store
    
    def retrieve(self, query: str, n_results: int = 4) -> Dict[str, Any]:
        """Recupera documentos relevantes para una consulta"""
        try:
            # Usar búsqueda con umbral más permisivo para producción
            results = self.vector_store.search(query, n_results, similarity_threshold=0.3)
            
            # Si no hay resultados, intentar con umbral más bajo
            if not results:
                results = self.vector_store.search(query, n_results, similarity_threshold=0.2)
            
            # Si no hay resultados después de todos los umbrales
            if not results:
                return {
                    "query": query,
                    "results": [],
                    "count": 0,
                    "status": "no_relevant_documents",
                    "message": "No se encontró información relevante en los documentos."
                }
            
            # Filtrar resultados de baja calidad (menos estricto en producción)
            filtered_results = self._filter_low_quality_results(query, results)
            
            if not filtered_results:
                # En producción, ser más permisivo - mostrar al menos los mejores resultados
                best_results = sorted(results, key=lambda x: x.get('similarity', 0), reverse=True)[:n_results]
                if best_results:
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
    
    def retrieve_with_fallback(self, query: str, n_results: int = 4) -> Dict[str, Any]:
        """Recuperación con estrategia de fallback más robusta"""
        try:
            # PRIMERO: Usar el método con fallback del vector store
            results = self.vector_store.search_with_fallback(query, n_results)
            
            if not results:
                return {
                    "query": query,
                    "results": [],
                    "count": 0,
                    "status": "no_relevant_documents",
                    "message": "No se encontró información relevante en los documentos."
                }
            
            # En producción, ser menos estricto con el filtrado
            filtered_results = self._filter_for_production(query, results)
            
            if filtered_results:
                best_document = self._get_best_document(filtered_results)
                return {
                    "query": query,
                    "results": filtered_results,
                    "count": len(filtered_results),
                    "status": "success",
                    "message": f"Encontrados {len(filtered_results)} documentos relevantes.",
                    "best_document": best_document
                }
            else:
                # Si el filtrado es muy estricto, devolver los mejores resultados encontrados
                best_results = sorted(results, key=lambda x: x.get('similarity', 0), reverse=True)[:n_results]
                if best_results:
                    best_document = self._get_best_document(best_results)
                    return {
                        "query": query,
                        "results": best_results,
                        "count": len(best_results),
                        "status": "success",
                        "message": f"Encontrados {len(best_results)} documentos con información relacionada.",
                        "best_document": best_document
                    }
                else:
                    return {
                        "query": query,
                        "results": [],
                        "count": 0,
                        "status": "low_similarity",
                        "message": "La información encontrada no es suficientemente relevante."
                    }
                    
        except Exception as e:
            logger.error(f"Error en recuperación con fallback: {e}")
            # Fallback a método básico
            return self.retrieve(query, n_results)
    
    def _filter_low_quality_results(self, query: str, results: List[Dict]) -> List[Dict]:
        """Filtra resultados de baja similitud o irrelevantes"""
        if not results:
            return []
        
        filtered_results = []
        query_lower = query.lower()
        
        # Palabras clave importantes de la consulta (excluyendo stop words simples)
        stop_words = {'que', 'es', 'la', 'el', 'de', 'en', 'y', 'o', 'un', 'una', 'los', 'las', 'eran', 'donde', 'como', 'cuando'}
        query_terms = [word for word in query_lower.split() if word not in stop_words and len(word) > 2]
        
        for result in results:
            content_lower = result['content'].lower()
            similarity = result.get('similarity', 0)
            
            # EN PRODUCCIÓN: Ser más permisivo con los umbrales
            
            # CRITERIO 1: Buena similitud (> 0.5) - incluir siempre
            if similarity > 0.5:
                filtered_results.append(result)
                continue
            
            # CRITERIO 2: Similitud aceptable (0.4-0.5) 
            if similarity >= 0.4:
                filtered_results.append(result)
                continue
            
            # CRITERIO 3: Para similitud baja (0.3-0.4), verificar términos clave
            if similarity >= 0.3 and query_terms:
                matching_terms = sum(1 for term in query_terms if term in content_lower)
                # Requerir al menos 1 término coincidente para similitud baja
                if matching_terms >= 1:
                    filtered_results.append(result)
        
        # Ordenar por similitud
        filtered_results.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        
        logger.info(f"Filtrado: {len(filtered_results)}/{len(results)} documentos considerados relevantes")
        
        return filtered_results
    
    def _filter_for_production(self, query: str, results: List[Dict]) -> List[Dict]:
        """Filtro más permisivo específico para producción"""
        if not results:
            return []
        
        filtered_results = []
        query_lower = query.lower()
        
        # Términos clave de la consulta
        stop_words = {'que', 'es', 'la', 'el', 'de', 'en', 'y', 'o', 'un', 'una', 'los', 'las', 'eran', 'donde', 'como', 'cuando', 'qué'}
        query_terms = [word for word in query_lower.split() if word not in stop_words and len(word) > 2]
        
        for result in results:
            content_lower = result['content'].lower()
            similarity = result.get('similarity', 0)
            
            # EN PRODUCCIÓN: Umbrales más bajos
            if similarity >= 0.3:  # Bajar el umbral mínimo
                # Verificar relevancia semántica básica
                if not query_terms or any(term in content_lower for term in query_terms):
                    filtered_results.append(result)
        
        # Ordenar por similitud
        filtered_results.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        
        logger.info(f"Filtrado producción: {len(filtered_results)}/{len(results)} documentos")
        
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
    
    def get_direct_answer(self, query: str, n_results: int = 4) -> str:
        """Obtiene respuesta directa basada en el documento más relevante"""
        retrieval_result = self.retrieve_with_fallback(query, n_results)
        
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
        
        # Determinar confianza basada en similitud (ajustado para producción)
        if similarity >= 0.7:
            confidence = "alta confianza"
        elif similarity >= 0.5:
            confidence = "buena confianza"
        elif similarity >= 0.3:
            confidence = "confianza moderada"
        else:
            confidence = "información relacionada"
        
        return f"Basado en la información más relevante ({confidence}, similitud: {similarity:.2f}):\n\n{cleaned_content}"
    
    def _clean_content(self, content: str) -> str:
        """Limpia y formatea el contenido para respuesta directa"""
        # Eliminar espacios múltiples y saltos de línea excesivos
        import re
        cleaned = re.sub(r'\s+', ' ', content)
        cleaned = cleaned.strip()
        
        # Limitar longitud si es muy extenso
        if len(cleaned) > 400:
            # Intentar cortar en un punto lógico
            if '.' in cleaned[:400]:
                last_dot = cleaned[:400].rfind('.')
                if last_dot > 200:  # Asegurar que no sea muy corto
                    cleaned = cleaned[:last_dot + 1]
            else:
                cleaned = cleaned[:400] + "..."
        
        return cleaned
    
    def get_context(self, query: str, n_results: int = 4) -> str:
        """Obtiene contexto formateado para LLM"""
        retrieval_result = self.retrieve_with_fallback(query, n_results)
        
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
            content_preview = result['content'][:400] + "..." if len(result['content']) > 400 else result['content']
            context_parts.append(f"[Documento {i+1} - Similitud: {similarity:.2f}]: {content_preview}")
        
        return "\n\n".join(context_parts)