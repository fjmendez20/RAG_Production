import requests
import logging
from typing import List, Dict, Any
import tempfile
import os
from urllib.parse import urljoin, urlparse
import re

logger = logging.getLogger(__name__)

class ProductionDocumentLoader:
    """Loader de documentos para producción desde GitHub Pages con detección exacta via GitHub API"""
    
    def __init__(self):
        self.supported_types = ['github_pages', 'github_pages_auto', 'url', 'raw_text']
        self.supported_extensions = ['.pdf', '.txt', '.md', '.html', '.htm', '.json', '.rst']
    
    def load_from_source(self, source_config: Dict[str, Any]) -> List[str]:
        """Carga documentos desde una fuente configurada"""
        source_type = source_config.get('type')
        
        if source_type == 'github_pages_auto':
            # NUEVO: Carga automática usando GitHub API
            return self.load_all_pdfs_from_repo(source_config['base_url'])
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
    
    def get_exact_files_from_repo(self, github_pages_url: str) -> List[str]:
        """Obtiene la lista EXACTA de archivos del repositorio usando GitHub API"""
        try:
            # Extraer owner y repo de la URL de GitHub Pages
            repo_info = self._extract_repo_info_from_pages_url(github_pages_url)
            if not repo_info:
                logger.error("No se pudo extraer información del repositorio")
                return []
            
            owner = repo_info['owner']
            repo = repo_info['repo']
            
            logger.info(f"📂 Buscando archivos en repositorio: {owner}/{repo}")
            
            # Usar GitHub API para obtener contenido del repositorio
            api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/"
            
            # Si tienes token, úsalo para mayor límite de requests
            headers = {}
            github_token = os.getenv("GITHUB_TOKEN")
            if github_token:
                headers["Authorization"] = f"token {github_token}"
            
            response = requests.get(api_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                contents = response.json()
                pdf_files = []
                
                for item in contents:
                    if item.get('type') == 'file' and item.get('name', '').lower().endswith('.pdf'):
                        pdf_files.append(item['name'])
                        logger.info(f"✅ Encontrado: {item['name']}")
                
                logger.info(f"📊 Total de archivos PDF encontrados: {len(pdf_files)}")
                return pdf_files
                
            elif response.status_code == 403:
                # Límite de API excedido, usar método alternativo
                logger.warning("Límite de GitHub API excedido, usando método alternativo")
                return self._discover_files_fallback(github_pages_url)
            else:
                logger.error(f"Error GitHub API: {response.status_code} - {response.text}")
                return self._discover_files_fallback(github_pages_url)
                
        except Exception as e:
            logger.error(f"Error obteniendo archivos del repo: {e}")
            return self._discover_files_fallback(github_pages_url)
    
    def _extract_repo_info_from_pages_url(self, github_pages_url: str) -> Dict[str, str]:
        """Extrae owner y repo de la URL de GitHub Pages"""
        try:
            # Para URLs como: https://username.github.io/repo-name/
            parsed = urlparse(github_pages_url)
            
            if 'github.io' in parsed.netloc:
                # username.github.io → owner = username
                owner = parsed.netloc.split('.')[0]
                
                # /repo-name/ → repo = repo-name
                path_parts = parsed.path.strip('/').split('/')
                repo = path_parts[0] if path_parts else None
                
                if owner and repo:
                    return {'owner': owner, 'repo': repo}
            
            logger.error(f"No se pudo extraer info de: {github_pages_url}")
            return {}
            
        except Exception as e:
            logger.error(f"Error extrayendo info del repo: {e}")
            return {}
    
    def _discover_files_fallback(self, base_url: str) -> List[str]:
        """Método de fallback si la API no funciona"""
        logger.info("Usando método de descubrimiento alternativo...")
        
        # Probar los archivos más comunes
        test_files = []
        for i in range(1, 20):
            test_files.extend([
                f"doc{i}.pdf", f"documento{i}.pdf", f"archivo{i}.pdf",
                f"doc_{i}.pdf", f"document_{i}.pdf", f"file{i}.pdf"
            ])
        
        # Agregar nombres sin números
        common_names = [
            "documento.pdf", "archivo.pdf", "informacion.pdf", "datos.pdf",
            "manual.pdf", "guia.pdf", "tutorial.pdf", "documentacion.pdf"
        ]
        test_files.extend(common_names)
        
        discovered_files = []
        for test_file in test_files:
            test_url = f"{base_url.rstrip('/')}/{test_file}"
            if self._check_file_exists(test_url):
                discovered_files.append(test_file)
                logger.info(f"✅ Encontrado (fallback): {test_file}")
        
        return discovered_files
    
    def load_all_pdfs_from_repo(self, github_pages_url: str) -> List[str]:
        """Carga TODOS los PDFs del repositorio usando lista exacta"""
        try:
            # Obtener lista exacta de archivos PDF
            pdf_files = self.get_exact_files_from_repo(github_pages_url)
            
            if not pdf_files:
                logger.warning("No se encontraron archivos PDF en el repositorio")
                return []
            
            documents = []
            successful_files = []
            
            for pdf_file in pdf_files:
                try:
                    url = f"{github_pages_url.rstrip('/')}/{pdf_file}"
                    logger.info(f"📥 Descargando: {url}")
                    
                    response = requests.get(url, timeout=30)
                    response.raise_for_status()
                    
                    # Extraer contenido del PDF
                    content = self._extract_content_from_file(response.content, pdf_file, url)
                    if content and len(content.strip()) > 10:
                        documents.append(content)
                        successful_files.append(pdf_file)
                        logger.info(f"✅ Cargado: {pdf_file} ({len(content)} caracteres)")
                    else:
                        logger.warning(f"⚠️ Contenido vacío: {pdf_file}")
                        
                except Exception as e:
                    logger.error(f"❌ Error cargando {pdf_file}: {e}")
                    continue
            
            logger.info(f"📊 Resumen final: {len(successful_files)}/{len(pdf_files)} archivos cargados")
            logger.info(f"📝 Archivos exitosos: {successful_files}")
            return documents
            
        except Exception as e:
            logger.error(f"Error en carga desde repo: {e}")
            return []
    
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
        """Extrae texto de PDF usando solo pypdf (más liviano)"""
        try:
            from pypdf import PdfReader
            from io import BytesIO
            
            with BytesIO(pdf_content) as pdf_file:
                reader = PdfReader(pdf_file)
                text = ""
                
                # Limitar número de páginas si es muy grande
                max_pages = 20
                for i, page in enumerate(reader.pages):
                    if i >= max_pages:
                        break
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                
                extracted_text = text.strip()
                if extracted_text:
                    logger.info(f"✅ PDF extraído con pypdf ({len(extracted_text)} caracteres)")
                    return extracted_text
                else:
                    logger.warning("PDF extraído pero texto vacío")
                    return ""
                    
        except Exception as e:
            logger.error(f"❌ Error extrayendo PDF con pypdf: {e}")
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
            
            # Probar con varios archivos comunes
            test_files = ["doc1.pdf", "documento.pdf", "archivo.pdf"]
            file_status = {}
            
            for test_file in test_files:
                test_url = f"{base_url.rstrip('/')}/{test_file}"
                test_response = requests.head(test_url, timeout=5)
                file_status[test_file] = "✅ Existe" if test_response.status_code == 200 else "❌ No existe"
            
            # Descubrir TODOS los PDFs usando GitHub API
            discovered_files = self.get_exact_files_from_repo(base_url)
            
            return {
                "base_url": base_url,
                "connection_status": status,
                "test_files_status": file_status,
                "discovered_files": discovered_files,
                "total_files_found": len(discovered_files)
            }
            
        except Exception as e:
            return {
                "base_url": base_url,
                "connection_status": f"❌ Error: {str(e)}",
                "test_files_status": {},
                "discovered_files": [],
                "total_files_found": 0
            }