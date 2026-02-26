# Microservice Migration Scaffold

This repository now contains a staged microservice layout under `services/` with FastAPI + layered structure (`api/service/adapter`).

## Services

- `services/api-gateway`: external entry point and service routing
- `services/file-service`: image upload and file handling
- `services/retrieve-service`: retrieval API
- `services/embedding-service`: embedding API
- `services/health-service`: dependency health checks
- `packages/shared/app_shared`: shared config, schema, and auth

## Run (Docker)

```bash
docker compose -f docker-compose.microservices.yml up --build
```

## Endpoints (gateway)

- `GET /health`
- `POST /upload`
- `POST /retrieve`

## Notes

- Existing monolith paths (`orchestrator_api`) are intentionally kept for gradual migration.
