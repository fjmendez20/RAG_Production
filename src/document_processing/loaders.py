import os
from typing import List
import logging

logger = logging.getLogger(__name__)

class DocumentLoader:
    """Cargador de documentos con soporte para TXT y PDF"""
    
    def __init__(self):
        self.supported_extensions = {'.pdf', '.txt'}
    
    def load_single_file(self, file_path: str) -> List[str]:
        """Carga un solo archivo - soporte para TXT y PDF básico"""
        try:
            if file_path.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return [content]
                
            elif file_path.endswith('.pdf'):
                return self._load_pdf(file_path)
                
            else:
                logger.warning(f"Formato no soportado: {file_path}")
                return []
                
        except Exception as e:
            logger.error(f"Error cargando {file_path}: {e}")
            return []
    
    def _load_pdf(self, file_path: str) -> List[str]:
        """Carga básica de PDF usando PyPDF2"""
        try:
            import PyPDF2
            documents = []
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    text = page.extract_text()
                    if text.strip():
                        documents.append(text)
            return documents
        except ImportError:
            logger.error("PyPDF2 no instalado. Instala con: pip install pypdf2")
            return []
        except Exception as e:
            logger.error(f"Error leyendo PDF {file_path}: {e}")
            return []
    
    def load_from_list(self, file_paths: List[str]) -> List[str]:
        """Carga documentos desde una lista de rutas"""
        all_documents = []
        for file_path in file_paths:
            documents = self.load_single_file(file_path)
            # Añadir metadatos básicos
            for doc in documents:
                all_documents.append(doc)
        return all_documents
    
    def create_sample_documents(self) -> List[str]:
        """Crea documentos de ejemplo para pruebas"""
        sample_docs = [
            "Python es un lenguaje de programación interpretado, de alto nivel y de propósito general.",
            "Los sistemas RAG (Retrieval-Augmented Generation) combinan recuperación de información con modelos de lenguaje.",
            "Las embeddings son representaciones vectoriales que capturan el significado semántico del texto.",
            "ChromaDB es una base de datos vectorial de código abierto para aplicaciones de IA.",
            "LangChain es un framework para desarrollar aplicaciones con modelos de lenguaje.",
            "El machine learning es un subcampo de la inteligencia artificial que se centra en algoritmos que pueden aprender de datos.",
            "Los transformers son arquitecturas de deep learning muy efectivas para procesamiento de lenguaje natural.",
            "La inteligencia artificial está transformando industrias como healthcare, finanzas y educación."
        ]
        return sample_docs