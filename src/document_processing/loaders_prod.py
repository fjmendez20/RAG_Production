import requests
import logging
from typing import List, Dict, Any
import tempfile
import os

logger = logging.getLogger(__name__)

class ProductionDocumentLoader:
    """Loader de documentos para producción desde fuentes externas"""
    
    def __init__(self):
        self.supported_types = ['url', 'github', 'raw_text']
    
    def load_from_source(self, source_config: Dict[str, Any]) -> List[str]:
        """Carga documentos desde una fuente configurada"""
        source_type = source_config.get('type')
        
        if source_type == 'url':
            return self._load_from_urls(source_config['urls'])
        elif source_type == 'github':
            return self._load_from_github(
                source_config['repo'],
                source_config.get('paths', []),
                source_config.get('branch', 'main')
            )
        elif source_type == 'raw_text':
            return source_config.get('texts', [])
        else:
            raise ValueError(f"Tipo de fuente no soportado: {source_type}")
    
    def _load_from_urls(self, urls: List[str]) -> List[str]:
        """Carga documentos desde URLs públicas"""
        documents = []
        
        for url in urls:
            try:
                logger.info(f"Descargando desde URL: {url}")
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                if url.endswith('.txt'):
                    documents.append(response.text)
                elif url.endswith('.pdf'):
                    # Para PDFs, necesitarías una librería como PyPDF2
                    pdf_text = self._extract_text_from_pdf(response.content)
                    documents.append(pdf_text)
                else:
                    # Asumir texto plano
                    documents.append(response.text)
                    
                logger.info(f"✅ Documento cargado desde: {url}")
                
            except Exception as e:
                logger.error(f"❌ Error cargando {url}: {e}")
                continue
        
        return documents
    
    def _load_from_github(self, repo: str, paths: List[str], branch: str = 'main') -> List[str]:
        """Carga documentos desde repositorio GitHub"""
        documents = []
        base_url = f"https://raw.githubusercontent.com/{repo}/{branch}/"
        
        for path in paths:
            try:
                url = base_url + path.lstrip('/')
                logger.info(f"Descargando desde GitHub: {url}")
                
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                documents.append(response.text)
                logger.info(f"✅ Documento cargado desde GitHub: {path}")
                
            except Exception as e:
                logger.error(f"❌ Error cargando {path} desde GitHub: {e}")
                continue
        
        return documents
    
    def _extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """Extrae texto de un PDF (implementación básica)"""
        try:
            # Intentar con PyPDF2 si está disponible
            import PyPDF2
            from io import BytesIO
            
            with BytesIO(pdf_content) as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except ImportError:
            logger.warning("PyPDF2 no disponible, no se puede extraer texto de PDF")
            return "[[Contenido PDF - instala PyPDF2 para extraer texto]]"
        except Exception as e:
            logger.error(f"Error extrayendo texto de PDF: {e}")
            return "[[Error extrayendo texto del PDF]]"
    
    def create_sample_sources(self) -> List[Dict[str, Any]]:
        """Crea fuentes de ejemplo para inicialización"""
        return [
            {
                "type": "raw_text",
                "texts": [
                    "Python es un lenguaje de programación interpretado, de alto nivel y de propósito general.",
                    "Los sistemas RAG combinan recuperación de información con generación de texto.",
                    "Los vikingos eran pueblos nórdicos de Escandinavia entre los siglos VIII y XI."
                ]
            }
        ]