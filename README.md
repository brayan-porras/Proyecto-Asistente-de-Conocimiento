# Asistente de Conocimiento

Asistente agéntico de conocimiento para desarrolladores: RAG orquestado como grafo de agentes (supervisor, subagente de documentación y subagente de investigación).

## Estructura

- `services/ingestion`: ingesta, chunking y embeddings
- `services/agent`: grafo agéntico y API de consultas
- `frontend`: interfaz de chat
- `eval`: evaluación y comparación con RAG lineal
- `docs`: diagramas y documentación

## Configuración

```bash
cp .env.example .env   
docker compose up -d
```

_En construcción._
