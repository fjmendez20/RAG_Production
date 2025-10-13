FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primero (para cache de Docker)
COPY requirements.txt .

# Instalar dependencias Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-descargar modelos durante el build
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Copiar el código de la aplicación
COPY . .

# Configurar variables de entorno para optimización
ENV CHROMA_DB_PATH=/app/chroma_db
ENV ANONYMIZED_TELEMETRY=false
ENV PERSIST_DIRECTORY=/app/chroma_db
ENV IS_PERSISTENT_TO_DISK=true
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Crear directorios necesarios
RUN mkdir -p /app/chroma_db
RUN mkdir -p /root/.cache/chroma
RUN mkdir -p /root/.cache/sentence_transformers
RUN mkdir -p /root/.cache/torch
RUN mkdir -p /root/.cache/huggingface

# Limpiar cache de apt
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Exponer puerto
EXPOSE 8000

# Comando de inicio
CMD ["uvicorn", "api_prod:app", "--host", "0.0.0.0", "--port", "8000"]