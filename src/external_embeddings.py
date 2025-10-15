import requests
import logging
import numpy as np
from typing import List, Dict, Any
import os
import time

logger = logging.getLogger(__name__)

class HuggingFaceEmbeddings:
    """Cliente para embeddings externos usando Hugging Face Inference API"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.api_token = os.getenv("HF_API_TOKEN")
        self.base_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_name}"
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Obtiene embeddings para una lista de textos"""
        if not self.api_token:
            logger.error("❌ HF_API_TOKEN no configurado")
            return []
        
        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json={
                    "inputs": texts,
                    "options": {"wait_for_model": True}
                },
                timeout=60
            )
            
            if response.status_code == 200:
                embeddings = response.json()
                logger.info(f"✅ Embeddings obtenidos para {len(texts)} textos")
                return embeddings
            else:
                logger.error(f"❌ Error API: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"❌ Error obteniendo embeddings: {e}")
            return []
    
    def get_embedding(self, text: str) -> List[float]:
        """Obtiene embedding para un solo texto"""
        embeddings = self.get_embeddings([text])
        return embeddings[0] if embeddings else []
    
    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calcula similitud coseno entre dos vectores"""
        if not a or not b or len(a) != len(b):
            return 0.0
        
        a_np = np.array(a)
        b_np = np.array(b)
        
        dot_product = np.dot(a_np, b_np)
        norm_a = np.linalg.norm(a_np)
        norm_b = np.linalg.norm(b_np)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        return dot_product / (norm_a * norm_b)