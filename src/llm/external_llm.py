import os
import logging
from typing import Dict, Any
import requests
import json

logger = logging.getLogger(__name__)

class ExternalLLM:
    """Cliente para LLM externo via API - Sin dependencias pesadas"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = "https://api.groq.com/openai/v1/"
        self.model = "llama-3.3-70b-versatile"
        
        if not self.api_key:
            logger.error("❌ No se proporcionó API key para LLM")
            raise ValueError("API key es requerida para ExternalLLM")
        
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        logger.info(f"LLM configurado: {self.model}")
    
    def generate_response(self, prompt: str, context: str, question: str, max_tokens: int = 500) -> Dict[str, Any]:
        """Genera respuesta usando OpenAI API directamente"""
        try:
            system_message = """Eres un asistente que responde preguntas basándose ÚNICAMENTE en el contexto proporcionado."""
            
            user_message = f"""CONTEXTO:
                {context}

                PREGUNTA: {question}

                Responde basándote ÚNICAMENTE en el contexto:"""
            
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                "max_completion_tokens": max_tokens,
                "temperature": 0.1
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result["choices"][0]["message"]["content"].strip()
                
                return {
                    "success": True,
                    "answer": answer,
                    "model": self.model
                }
            else:
                logger.error(f"Error en OpenAI API: {response.status_code}")
                return {
                    "success": False,
                    "error": f"API Error: {response.status_code}",
                    "fallback_answer": "No pude generar una respuesta en este momento."
                }
                
        except Exception as e:
            logger.error(f"Error generando respuesta: {e}")
            return {
                "success": False,
                "error": str(e),
                "fallback_answer": "Error técnico al generar la respuesta."
            }
    
    def is_configured(self) -> bool:
        return bool(self.api_key)