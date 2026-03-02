.PHONY: help up down build logs pull-models shell-agent shell-qdrant dashboard dashboard-logs clean

COMPOSE = docker compose
AGENT_SVC = agent

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-22s\033[0m %s\n", $$1, $$2}'

up: ## Start all services
	$(COMPOSE) up -d

down: ## Stop all services
	$(COMPOSE) down

build: ## Build agent, taiga-mcp and telegram-mcp images
	$(COMPOSE) build agent taiga-mcp telegram-mcp

dashboard: ## Start LangGraph Studio dashboard (http://localhost:8123)
	$(COMPOSE) up -d --build langgraph-dashboard

dashboard-logs: ## Follow LangGraph dashboard logs
	$(COMPOSE) logs -f langgraph-dashboard

logs: ## Follow logs from all services
	$(COMPOSE) logs -f

logs-agent: ## Follow agent logs
	$(COMPOSE) logs -f agent

pull-models: ## Pull Ollama models (llama3.2 + nomic-embed-text)
	$(COMPOSE) exec ollama ollama pull llama3.2
	$(COMPOSE) exec ollama ollama pull nomic-embed-text

shell-agent: ## Open bash in agent container
	$(COMPOSE) exec $(AGENT_SVC) bash

shell-qdrant: ## Open Qdrant REST CLI via curl
	@echo "Qdrant dashboard: http://localhost:6333/dashboard"

clean: ## Remove containers, images, volumes (DESTRUCTIVE)
	$(COMPOSE) down -v --rmi local

status: ## Show service health
	$(COMPOSE) ps
