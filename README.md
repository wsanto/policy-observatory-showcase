# Policy Observatory Platform (showcase excerpt)

An excerpt from a multi-tenant SaaS platform I built for running large-scale, multi-agent policy
simulations. This repo shows the **platform/backend layer** — auth, RBAC, multi-tenancy, project
and run management, audit logging, and an LLM provider connector — as a demonstration of backend
system design, not the simulation/analysis engine itself or any specific client's work product.

This is a curated excerpt, not the full platform: the simulation engine and any client-specific
configuration live in the private codebase, so this repo is for reading, not running.

## What's included here

- **`platform/auth.py`, `rbac.py`, `tokens.py`** — authentication and role-based access control for
  a multi-tenant SaaS product.
- **`platform/database.py`, `storage.py`, `middleware.py`** — data access layer and request
  middleware.
- **`platform/projects.py`, `runs.py`, `scenarios.py`, `compare.py`, `search.py`, `comments.py`,
  `audit.py`** — project/run lifecycle management, scenario comparison, search, collaboration
  (comments), and audit logging.
- **`platform/models.py`, `models_catalog.py`, `model_assistant.py`** — data models and an
  LLM-assisted helper layer.
- **`platform/orchestrator.py`** — background job orchestration for long-running simulation runs.
- **`platform/document_processor.py`, `news_monitor.py`** — generic document ingestion/extraction
  and news monitoring.
- **`connectors/kimi.py`** — an LLM provider connector.

## What was built but isn't shown here

- **The simulation/experiment engine** — a multi-agent stakeholder swarm that runs hundreds of
  LLM-powered agents per study, plus empirical risk modeling (Monte Carlo GDP projections) and
  real-time geopolitical feed integration.
- **A curiosity-driven sampling engine** ("Neuroplastic Curiosity Sampling") used to prioritize what
  the simulation explores next — related to a synthetic-intelligence framework I've built elsewhere.
- **Client-specific configuration and studies** — this platform has run real, commissioned policy
  studies; none of that client work, its inputs, or its outputs are included here.
- **The front-end dashboards** (observatory, research portal, promo site) that sit on top of this
  backend.

I'm happy to walk through the design of any of these in conversation — they're just not published
as code.

## Stack

Python (FastAPI) backend with Neo4j, containerized for Fly.io deployment. Multi-tenant SaaS
patterns: JWT auth, RBAC, audit logging, background job orchestration.
