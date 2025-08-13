# Load environment variables from .env if present
ifneq (,$(wildcard .env))
	include .env
	export
endif

DOCKER_COMPOSE := docker compose
NETWORK := antman
MC_IMAGE := quay.io/minio/mc
MC_ALIAS_ENV := -e MC_HOST_antman=http://$(MINIO_ROOT_USER):$(MINIO_ROOT_PASSWORD)@minio:9000

.PHONY: up down logs console alias bucket health duck

up:
	$(DOCKER_COMPOSE) up -d

down:
	$(DOCKER_COMPOSE) down

logs:
	$(DOCKER_COMPOSE) logs -f minio

console:
	@echo "Open MinIO Console: http://localhost:$(MINIO_CONSOLE_PORT) (user: $$MINIO_ROOT_USER)"

alias:
	docker run --rm --network $(NETWORK) $(MC_ALIAS_ENV) $(MC_IMAGE) ls antman || true

bucket: up alias
	docker run --rm --network $(NETWORK) $(MC_ALIAS_ENV) $(MC_IMAGE) mb -p antman/$(MINIO_BUCKET) || true
	docker run --rm --network $(NETWORK) $(MC_ALIAS_ENV) $(MC_IMAGE) ls antman/$(MINIO_BUCKET) || true

health:
	@echo "S3 endpoint: $(S3_ENDPOINT_URL)"
	@echo "Bucket: $(MINIO_BUCKET)"
	@echo "If alias fails, ensure container is up and creds are correct in .env"

# Open DuckDB CLI with httpfs preconfigured (requires local duckdb CLI installed)
# If duckdb is missing, install via your package manager or use Python/Notebook instead.
duck:
	@if command -v duckdb >/dev/null 2>&1; then \
		duckdb -init configs/duckdb_init.sql ; \
	else \
		echo "duckdb CLI not found. Install it or run the SQL in configs/duckdb_init.sql from a Python session."; \
	fi
