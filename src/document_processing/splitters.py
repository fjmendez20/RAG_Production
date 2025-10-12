from typing import List
import logging

logger = logging.getLogger(__name__)

class DocumentSplitter:
    """Splitter de documentos para producción"""
    
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_documents(self, documents: List[str]) -> List[str]:
        """Divide documentos en chunks - versión optimizada"""
        if not documents:
            return []
        
        all_chunks = []
        
        for doc in documents:
            # División simple pero efectiva
            words = doc.split()
            current_chunk = []
            current_length = 0
            
            for word in words:
                word_length = len(word) + 1  # +1 por el espacio
                
                if current_length + word_length <= self.chunk_size:
                    current_chunk.append(word)
                    current_length += word_length
                else:
                    # Guardar chunk actual
                    if current_chunk:
                        chunk_text = ' '.join(current_chunk)
                        all_chunks.append(chunk_text)
                    
                    # Iniciar nuevo chunk con overlap
                    overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                    current_chunk = current_chunk[overlap_start:]
                    current_chunk.append(word)
                    
                    # Calcular nueva longitud
                    current_length = len(' '.join(current_chunk))
            
            # Añadir el último chunk
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                all_chunks.append(chunk_text)
        
        logger.info(f"Divididos {len(documents)} documentos en {len(all_chunks)} chunks")
        return all_chunks