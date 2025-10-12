#!/bin/bash

echo "🚀 Iniciando RAG Production API..."
echo "🔧 Entorno: ${RAG_ENVIRONMENT:-development}"

# Ejecutar la API
exec python api_prod.py