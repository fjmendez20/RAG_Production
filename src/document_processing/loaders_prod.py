import requests
import logging
from typing import List, Dict, Any
import tempfile
import os
from urllib.parse import urljoin, urlparse
import re

logger = logging.getLogger(__name__)

class ProductionDocumentLoader:
    """Loader de documentos para producción desde GitHub Pages con detección automática"""
    
    def __init__(self):
        self.supported_types = ['github_pages', 'github_pages_auto', 'url', 'raw_text']
        self.supported_extensions = ['.pdf', '.txt', '.md', '.html', '.htm', '.json', '.rst']
    
    def load_from_source(self, source_config: Dict[str, Any]) -> List[str]:
        """Carga documentos desde una fuente configurada"""
        source_type = source_config.get('type')
        
        if source_type == 'github_pages_auto':
            return self._load_from_github_pages_auto(
                source_config['base_url'],
                source_config.get('file_pattern', 'doc*.pdf'),
                source_config.get('max_files', 20)
            )
        elif source_type == 'github_pages':
            return self._load_from_github_pages(
                source_config['base_url'],
                source_config.get('files', []),
                source_config.get('paths', [])
            )
        elif source_type == 'url':
            return self._load_from_urls(source_config['urls'])
        elif source_type == 'raw_text':
            return source_config.get('texts', [])
        else:
            raise ValueError(f"Tipo de fuente no soportado: {source_type}")
    
    def _load_from_github_pages_auto(self, base_url: str, file_pattern: str = 'doc*.pdf', max_files: int = 20) -> List[str]:
        """Carga documentos automáticamente desde GitHub Pages detectando archivos"""
        documents = []
        
        try:
            # Primero intentar cargar archivos específicos comunes
            common_files = self._discover_files_by_pattern(base_url, file_pattern, max_files)
            
            if not common_files:
                logger.warning("No se encontraron archivos que coincidan con el patrón")
                return documents
            
            logger.info(f"Encontrados {len(common_files)} archivos: {common_files}")
            
            # Cargar cada archivo encontrado
            for file_path in common_files:
                try:
                    # Construir URL completa en GitHub Pages
                    url = urljoin(base_url.rstrip('/') + '/', file_path.lstrip('/'))
                    
                    logger.info(f"Descargando: {url}")
                    
                    response = requests.get(url, timeout=30)
                    response.raise_for_status()
                    
                    # Guardar archivo temporalmente para debug
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        tmp_file.write(response.content)
                        tmp_path = tmp_file.name
                    
                    # Extraer contenido
                    content = self._extract_content_from_file(response.content, file_path, url)
                    
                    # Limpiar archivo temporal
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
                    
                    if content and len(content.strip()) > 10:  # Validar que tenga contenido
                        documents.append(content)
                        logger.info(f"✅ Cargado: {file_path} ({len(content)} caracteres)")
                    else:
                        logger.warning(f"⚠️ Contenido vacío o muy corto en: {file_path}")
                        # Intentar métodos alternativos de extracción
                        alternative_content = self._extract_with_alternative_methods(response.content, file_path)
                        if alternative_content and len(alternative_content.strip()) > 10:
                            documents.append(alternative_content)
                            logger.info(f"✅ Cargado (método alternativo): {file_path}")
                        
                except Exception as e:
                    logger.error(f"❌ Error cargando {file_path}: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error en carga automática: {e}")
        
        logger.info(f"Total de documentos cargados automáticamente: {len(documents)}")
        return documents
    
    def _extract_with_alternative_methods(self, content: bytes, file_path: str) -> str:
        """Intenta extraer contenido con métodos alternativos"""
        file_ext = file_path.lower().split('.')[-1] if '.' in file_path else ''
        
        if file_ext == 'pdf':
            # Método 2: PyMuPDF (si está disponible)
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(stream=content, filetype="pdf")
                text = ""
                for page in doc:
                    text += page.get_text() + "\n"
                doc.close()
                if text.strip():
                    return text
            except ImportError:
                logger.info("PyMuPDF no disponible")
            except Exception as e:
                logger.error(f"Error con PyMuPDF: {e}")
            
            # Método 3: pdfplumber (si está disponible)
            try:
                import pdfplumber
                with pdfplumber.open(stream=content) as pdf:
                    text = ""
                    for page in pdf.pages:
                        if page.extract_text():
                            text += page.extract_text() + "\n"
                    return text
            except ImportError:
                logger.info("pdfplumber no disponible")
            except Exception as e:
                logger.error(f"Error con pdfplumber: {e}")
        
        return ""
    
    def _extract_repo_info(self, github_pages_url: str) -> Dict[str, str]:
        """Extrae información del repositorio de la URL de GitHub Pages"""
        try:
            parsed = urlparse(github_pages_url)
            path_parts = parsed.path.strip('/').split('/')
            
            # Para URLs como https://username.github.io/repo-name/
            if 'github.io' in parsed.netloc:
                if len(path_parts) >= 1:
                    repo_name = path_parts[0]
                    username = parsed.netloc.split('.')[0]
                    return {'owner': username, 'repo': repo_name}
            
            return {}
            
        except Exception as e:
            logger.error(f"Error extrayendo info del repo: {e}")
            return {}
    
    def _get_repo_files_via_api(self, owner: str, repo: str) -> List[str]:
        """Intenta obtener lista de archivos via GitHub API"""
        try:
            api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/"
            response = requests.get(api_url, timeout=10)
            
            if response.status_code == 200:
                files_data = response.json()
                file_paths = []
                
                for item in files_data:
                    if isinstance(item, dict) and item.get('type') == 'file':
                        file_name = item.get('name', '')
                        if any(file_name.endswith(ext) for ext in self.supported_extensions):
                            file_paths.append(file_name)
                
                return file_paths
            else:
                logger.info(f"GitHub API no disponible (status: {response.status_code})")
                return []
                
        except Exception as e:
            logger.info(f"GitHub API no disponible: {e}")
            return []
    
    def _discover_files_by_pattern(self, base_url: str, file_pattern: str, max_files: int = 20) -> List[str]:
        """Descubre archivos probando patrones comunes"""
        discovered_files = []
        
        # Convertir patrón a regex
        pattern = file_pattern.replace('*', '.*')
        regex = re.compile(pattern, re.IGNORECASE)
        
        # Probar números del 1 al max_files
        for i in range(1, max_files + 1):
            # Probar diferentes formatos
            test_files = [
                f"doc{i}.pdf",
                f"documento{i}.pdf", 
                f"archivo{i}.pdf",
                f"doc{i}.txt",
                f"documento{i}.txt",
                f"doc{i}.md",
                f"documento{i}.md"
            ]
            
            for test_file in test_files:
                if regex.match(test_file):
                    test_url = urljoin(base_url.rstrip('/') + '/', test_file)
                    if self._check_file_exists(test_url):
                        discovered_files.append(test_file)
                        logger.info(f"✅ Descubierto: {test_file}")
                        break
        
        return discovered_files
    
    def _check_file_exists(self, url: str) -> bool:
        """Verifica si un archivo existe haciendo HEAD request"""
        try:
            response = requests.head(url, timeout=5, allow_redirects=True)
            return response.status_code == 200
        except:
            return False
    
    def _load_from_github_pages(self, base_url: str, files: List[str] = None, paths: List[str] = None) -> List[str]:
        """Carga documentos específicos desde GitHub Pages"""
        documents = []
        
        all_files = files or []
        if paths:
            all_files.extend(paths)
        
        if not all_files:
            logger.warning("No se especificaron archivos para cargar desde GitHub Pages")
            return documents
        
        for file_path in all_files:
            try:
                # Construir URL completa
                url = urljoin(base_url.rstrip('/') + '/', file_path.lstrip('/'))
                
                logger.info(f"Descargando desde GitHub Pages: {url}")
                
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                # Extraer contenido
                content = self._extract_content_from_file(response.content, file_path, url)
                if content and len(content.strip()) > 10:
                    documents.append(content)
                    logger.info(f"✅ Documento cargado: {file_path} ({len(content)} caracteres)")
                else:
                    logger.warning(f"⚠️ Contenido vacío en: {file_path}")
                    # Intentar métodos alternativos
                    alternative_content = self._extract_with_alternative_methods(response.content, file_path)
                    if alternative_content and len(alternative_content.strip()) > 10:
                        documents.append(alternative_content)
                        logger.info(f"✅ Cargado (método alternativo): {file_path}")
                    
            except Exception as e:
                logger.error(f"❌ Error cargando {file_path} desde GitHub Pages: {e}")
                continue
        
        logger.info(f"Total de documentos cargados desde GitHub Pages: {len(documents)}")
        return documents
    
    def _extract_content_from_file(self, content: bytes, file_path: str, url: str = "") -> str:
        """Extrae contenido de diferentes tipos de archivos"""
        file_ext = file_path.lower().split('.')[-1] if '.' in file_path else ''
        
        try:
            if file_ext in ['pdf']:
                return self._extract_text_from_pdf(content)
            elif file_ext in ['txt', 'md', 'rst']:
                text_content = content.decode('utf-8')
                return text_content if text_content.strip() else ""
            elif file_ext in ['html', 'htm']:
                return self._extract_text_from_html(content)
            elif file_ext in ['json']:
                return self._extract_text_from_json(content)
            else:
                # Intentar decodificar como texto plano
                try:
                    text_content = content.decode('utf-8')
                    return text_content if text_content.strip() and any(char.isalpha() for char in text_content) else ""
                except:
                    return ""
                    
        except Exception as e:
            logger.error(f"Error extrayendo contenido de {file_path}: {e}")
            return ""
    
    def _extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """Extrae texto de un PDF con múltiples métodos"""
        extracted_text = ""
        
        # Método 1: PyPDF2 (primario)
        try:
            import PyPDF2
            from io import BytesIO
            
            with BytesIO(pdf_content) as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                extracted_text = text.strip()
                if extracted_text:
                    logger.info("✅ PDF extraído con PyPDF2")
                    return extracted_text
        except Exception as e:
            logger.warning(f"PyPDF2 no pudo extraer texto: {e}")
        
        # Método 2: PyMuPDF (fallback)
        try:
            import fitz
            doc = fitz.open(stream=pdf_content, filetype="pdf")
            text = ""
            for page in doc:
                page_text = page.get_text()
                if page_text:
                    text += page_text + "\n"
            doc.close()
            extracted_text = text.strip()
            if extracted_text:
                logger.info("✅ PDF extraído con PyMuPDF")
                return extracted_text
        except ImportError:
            logger.info("PyMuPDF no disponible")
        except Exception as e:
            logger.warning(f"PyMuPDF no pudo extraer texto: {e}")
        
        # Método 3: pdfplumber (fallback)
        try:
            import pdfplumber
            with pdfplumber.open(stream=pdf_content) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                extracted_text = text.strip()
                if extracted_text:
                    logger.info("✅ PDF extraído con pdfplumber")
                    return extracted_text
        except ImportError:
            logger.info("pdfplumber no disponible")
        except Exception as e:
            logger.warning(f"pdfplumber no pudo extraer texto: {e}")
        
        logger.error("❌ No se pudo extraer texto del PDF con ningún método")
        return ""
    
    def _extract_text_from_html(self, html_content: bytes) -> str:
        """Extrae texto limpio de HTML"""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            for script in soup(["script", "style", "nav", "header", "footer"]):
                script.decompose()
            
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
        except ImportError:
            return html_content.decode('utf-8')
        except Exception as e:
            logger.error(f"Error extrayendo texto de HTML: {e}")
            return html_content.decode('utf-8')
    
    def _extract_text_from_json(self, json_content: bytes) -> str:
        """Extrae texto de JSON"""
        try:
            import json
            data = json.loads(json_content.decode('utf-8'))
            
            text_parts = []
            
            def extract_text(obj):
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        if isinstance(value, str) and len(value) > 10:
                            text_parts.append(f"{key}: {value}")
                        else:
                            extract_text(value)
                elif isinstance(obj, list):
                    for item in obj:
                        extract_text(item)
                elif isinstance(obj, str) and len(obj) > 10:
                    text_parts.append(obj)
            
            extract_text(data)
            return "\n".join(text_parts)
        except Exception as e:
            logger.error(f"Error extrayendo texto de JSON: {e}")
            return json_content.decode('utf-8')
    
    def _load_from_urls(self, urls: List[str]) -> List[str]:
        """Carga documentos desde URLs públicas"""
        documents = []
        
        for url in urls:
            try:
                logger.info(f"Descargando desde URL: {url}")
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                file_name = url.split('/')[-1] if '/' in url else 'document'
                content = self._extract_content_from_file(response.content, file_name, url)
                if content:
                    documents.append(content)
                    logger.info(f"✅ Documento cargado desde: {url}")
                
            except Exception as e:
                logger.error(f"❌ Error cargando {url}: {e}")
                continue
        
        return documents
    
    def create_sample_sources(self) -> List[Dict[str, Any]]:
        """Crea fuentes de ejemplo para inicialización"""
        return []
    
    def test_github_pages_connection(self, base_url: str) -> Dict[str, Any]:
        """Método para probar la conexión con GitHub Pages"""
        try:
            response = requests.get(base_url, timeout=10)
            status = "✅ Conectado" if response.status_code == 200 else f"❌ Error {response.status_code}"
            
            test_file_url = urljoin(base_url.rstrip('/') + '/', 'doc1.pdf')
            test_response = requests.head(test_file_url, timeout=5)
            file_status = "✅ Existe" if test_response.status_code == 200 else "❌ No existe"
            
            discovered_files = self._discover_files_by_pattern(base_url, 'doc*.pdf', 10)
            
            return {
                "base_url": base_url,
                "connection_status": status,
                "doc1.pdf_status": file_status,
                "discovered_files": discovered_files,
                "total_files_found": len(discovered_files)
            }
            
        except Exception as e:
            return {
                "base_url": base_url,
                "connection_status": f"❌ Error: {str(e)}",
                "doc1.pdf_status": "No probado",
                "discovered_files": [],
                "total_files_found": 0
            }