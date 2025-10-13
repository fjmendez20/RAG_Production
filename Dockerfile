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

# Copiar el código de la aplicación
COPY . .

# Crear directorio para ChromaDB
RUN mkdir -p /tmp/chroma_db

# Exponer puerto
EXPOSE 8000

# Comando de inicio
CMD ["uvicorn", "api_prod:app", "--host", "0.0.0.0", "--port", "8000"]