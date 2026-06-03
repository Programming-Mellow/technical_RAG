# Technical RAG

A production-grade Retrieval-Augmented Generation (RAG) system for querying technical documents (equipment manuals, OSHA safety docs, technical specifications). Built with a focus on security, observability, and enterprise deployment patterns.

> **Status:** Phase 2 in progress — local prototype. See [project phases](#project-phases) for the full roadmap.

---

## Problem & Motivation

General-purpose LLMs hallucinate when asked about specific technical documentation — they either fabricate plausible-sounding but incorrect specs or fail to answer at all. RAG grounds responses in a retrieved document corpus, returning answers that cite their sources and degrade gracefully when the answer isn't in the docs.

This project builds that system end-to-end: from a local proof-of-concept to a cloud-deployed, security-tested, production-ready API.

---

## Architecture

The system separates **ingestion** from **inference** at every layer. This is an intentional constraint, not an accident — it allows each path to scale independently and means new documents can be indexed without touching the inference service.

```
                    ┌─────────────────────────────────────────────┐
                    │                   Local                     │
                    │                                             │
  Documents ──────► │  Ingestion Script ──► ChromaDB              │
                    │                          │                  │
  Query ──────────► │  FastAPI /query ─────────┘ ──► LM Studio    │
                    └─────────────────────────────────────────────┘

                    ┌─────────────────────────────────────────────┐
                    │                    AWS                      │
                    │                                             │
  S3 Upload ──────► │  Lambda (+ DLQ) ──► OpenSearch Serverless   │
                    │                          │                  │
  Query ──────────► │  API Gateway ──► WAF ──► ECS Fargate ───────┘
                    │                          │                  │
                    │             Secrets Manager / CloudWatch    │
                    └─────────────────────────────────────────────┘
```

---

## Tech Stack

| Concern | Local | AWS |
|---|---|---|
| Document storage | Local filesystem | S3 (versioning + KMS CMK encryption) |
| Embedding model | `sentence-transformers` | Bedrock Titan Embeddings or same model on ECS |
| Vector store | ChromaDB | OpenSearch Serverless |
| LLM | LM Studio (Mistral or LLaMA 3) | Bedrock Claude/Mistral or ECS-hosted open model |
| Orchestration | LangChain or LlamaIndex | Same, running inside container |
| API layer | FastAPI | FastAPI on ECS Fargate |
| Containerization | Docker + docker-compose | ECR + ECS |
| API exposure | localhost | API Gateway + AWS WAF |
| IaC | N/A | CloudFormation (nested stacks) |
| Monitoring | Console logs | CloudWatch Logs + Alarms |
| Ingestion trigger | Manual script | S3 Event → Lambda + DLQ |
| Secrets | `.env` (gitignored) | AWS Secrets Manager |
| CI/CD | N/A | GitHub Actions (ci, cfn-validate, deploy) |
| AWS auth in CI | N/A | GitHub OIDC → IAM role (no long-lived keys) |

---

## Running Locally

> **Phase 1–2 in progress.** Instructions will be added here once the local prototype is complete.

Prerequisites:
- Python 3.11+
- Docker + Docker Compose
- [LM Studio](https://lmstudio.ai/) with Mistral or LLaMA 3 loaded and local server enabled (`localhost:1234`)

```bash
# Clone and set up
git clone https://github.com/Programming-Mellow/technical_RAG.git
cd technical_RAG
cp .env.example .env  # fill in as needed

# Install dependencies
pip install -r requirements.txt

# Ingest documents (separate entry point from the API)
python ingest.py --corpus-dir ./docs

# Start the API + ChromaDB
docker-compose up
```

---

## Deploying to AWS

> **Phase 3–4 in progress.** Deployment instructions will be added here once CloudFormation stacks are complete.

The deployment uses nested CloudFormation stacks deployed in dependency order: networking → storage → compute. The CD pipeline authenticates to AWS via GitHub OIDC — no long-lived access keys are stored in GitHub Secrets.

```bash
# Validate templates before deploying
cfn-lint infra/**/*.yml
checkov -d infra/

# Deploy (handled by CD pipeline on merge to main)
aws cloudformation deploy --stack-name rag-networking ...
```

---

## Security Posture

Security controls are defined at every layer, from local development through production deployment.

**Input & output**
- Input validation (max length, control character stripping) at the API boundary only — not inside the pipeline
- All LLM outputs include source attribution before being returned to the caller
- AWS WAF WebACL on API Gateway with AWS Managed Rules (Common Rule Set + Known Bad Inputs)

**Secrets & credentials**
- No secrets in code or environment variables — Secrets Manager at runtime, `.env` (gitignored) locally
- GitHub Actions authenticates to AWS via OIDC — no long-lived access keys anywhere

**Infrastructure**
- IAM least-privilege — no wildcard resource statements on data-plane actions
- S3 encrypted with KMS customer-managed key (CMK), versioning enabled, public access blocked
- VPC endpoints for S3 and Bedrock keep traffic off the public internet
- Dead Letter Queue on the Lambda ingestion function — failed ingestions are captured, not silently dropped

**CI pipeline gates**
- `cfn-lint` + `checkov` run on every PR touching `infra/` before any template reaches AWS
- `pip-audit` scans Python dependencies for known CVEs (Phase 5)
- Dependabot monitors Python and Docker dependencies for new vulnerabilities

**Adversarial testing**
- Phase 1: baseline prompt injection behavior documented against local loop
- Phase 2: `/query` endpoint tested against OWASP LLM Top 10 LLM01 and LLM02
- Phase 5: structured prompt injection test suite (direct, indirect via poisoned chunk, jailbreak) + full OWASP LLM Top 10 annotation

---

## GitHub Actions Workflows

| Workflow | Trigger | What it does |
|---|---|---|
| `ci.yml` | PR to main | Lint (`ruff`), test (`pytest`), Docker build |
| `cfn-validate.yml` | PR touching `infra/` | `cfn-lint` + `checkov` IaC scan |
| `deploy.yml` | Merge to main | OIDC auth → deploy CloudFormation → push to ECR |

---

## Project Phases

| Phase | Focus | Weeks | Status |
|---|---|---|---|
| 1 | Local RAG prototype + repo hardening | 1–2 | Finished |
| 2 | FastAPI + Docker + CI | 3–4 | In Progress |
| 3 | CloudFormation architecture | 5–6 | Planned |
| 4 | AWS deployment + CD pipeline | 7–8 | Planned |
| 5 | AI security testing & governance | 9 | Planned |
| 6 | Polish, docs, GitHub Release | 10 | Planned |

---

## Architectural Decisions

**Why separate ingestion from inference?**
Wiring them together means you can't re-ingest new documents without restarting the API, and the two paths can't scale independently. The separation is enforced from day one — locally as distinct entry points, on AWS as Lambda (ingestion) vs ECS (inference).

**Why GitHub OIDC instead of access keys in CI?**
Long-lived access keys stored in GitHub Secrets have a wide blast radius if leaked via a log line or a fork. OIDC credentials are scoped to a specific repo, expire automatically, and leave no persistent secret to rotate.

**Why CloudFormation + Checkov over console-click deployment?**
Policy-as-code catches misconfigurations (unencrypted S3, overly permissive security groups, missing KMS) before they ever reach AWS. A misconfiguration caught in a PR review is free; one caught after a week of runtime is not.

**Why a DLQ on the Lambda ingestion function?**
Silent failures are worse than loud ones. If a document fails to ingest, the DLQ captures the event so it can be replayed or investigated — without losing it.
