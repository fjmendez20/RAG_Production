import os
import logging
from typing import Dict, Any
import requests
import json

logger = logging.getLogger(__name__)

class ExternalLLM:
    """Cliente para LLM externo via API - Versión producción"""
    
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.base_url = base_url or os.getenv("LLM_BASE_URL")
        self.model = model or os.getenv("LLM_MODEL")
        
        if not self.api_key:
            logger.error("❌ No se proporcionó API key para LLM")
            raise ValueError("API key es requerida para ExternalLLM")
        
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        logger.info(f"LLM configurado: {self.model} en {self.base_url}")
    
    def generate_response(self, prompt: str, context: str, question: str, max_tokens: int = 500) -> Dict[str, Any]:
        """Genera respuesta usando LLM externo"""
        try:
            # Construir el prompt optimizado para producción
            system_message = """Eres un asistente especializado que responde preguntas basándose ÚNICAMENTE en el contexto proporcionado. 

INSTRUCCIONES:
1. Responde usando SOLO la información del contexto
2. Si el contexto no contiene información relevante, di: "No tengo suficiente información en mis documentos para responder esta pregunta"
3. Sé conciso, directo y factual
4. No inventes información que no esté en el contexto
5. Si el contexto tiene información relevante, úsala para construir una respuesta completa"""

            user_message = f"""CONTEXTO DISPONIBLE:
{context}

PREGUNTA DEL USUARIO: {question}

Basándote ÚNICAMENTE en el contexto proporcionado, responde la pregunta:"""
            
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                "max_completion_tokens": max_tokens,
                "temperature": 0.3
            }
            
            logger.info(f"Enviando solicitud a {self.base_url} con modelo {self.model}")
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=45  # Timeout más largo para producción
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result["choices"][0]["message"]["content"].strip()
                
                logger.info(f"✅ Respuesta LLM generada exitosamente")
                
                return {
                    "success": True,
                    "answer": answer,
                    "model": self.model,
                    "tokens_used": result.get("usage", {}).get("total_tokens", 0)
                }
            else:
                logger.error(f"Error en LLM API: {response.status_code} - {response.text}")
                return {
                    "success": False,
                    "error": f"API Error: {response.status_code}",
                    "fallback_answer": "No pude generar una respuesta en este momento. Por favor, intenta nuevamente."
                }
                
        except requests.exceptions.Timeout:
            logger.error("Timeout en LLM API - la solicitud tardó demasiado")
            return {
                "success": False,
                "error": "Timeout",
                "fallback_answer": "El servicio de IA está tardando demasiado en responder. Por favor, intenta nuevamente."
            }
        except requests.exceptions.ConnectionError:
            logger.error("Error de conexión con LLM API")
            return {
                "success": False,
                "error": "ConnectionError",
                "fallback_answer": "No pude conectarme al servicio de IA. Verifica tu conexión e intenta nuevamente."
            }
        except Exception as e:
            logger.error(f"Error generando respuesta con LLM: {e}")
            return {
                "success": False,
                "error": str(e),
                "fallback_answer": "Error técnico al generar la respuesta. Por favor, intenta nuevamente."
            }
    
    def is_configured(self) -> bool:
        """Verifica si el LLM está configurado correctamente"""
        return bool(self.api_key and self.base_url and self.model)

# Cliente específico para OpenAI
class OpenAIClient(ExternalLLM):
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
        super().__init__(
            api_key=api_key,
            base_url="https://api.openai.com/v1",
            model=model
        )