"""Qdrant vector store client with fastembed embeddings + fast reranker."""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

import httpx
from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.http.models import (
    Distance,
    HnswConfigDiff,
    OptimizersConfigDiff,
    PointStruct,
    ScoredPoint,
    VectorParams,
)
from qdrant_client.models import Filter
import structlog

from src.config import get_settings, load_yaml_config

logger = structlog.get_logger(__name__)


class QdrantStore:
    """Thin wrapper around QdrantClient exposing operations used by the agent."""

    def __init__(self) -> None:
        cfg = get_settings()
        yaml_cfg = load_yaml_config()
        qdrant_cfg = yaml_cfg.get("qdrant", {})

        self._collection = cfg.qdrant_collection
        self._vector_size: int = qdrant_cfg.get("vector_size", 768)
        self._distance = Distance.COSINE
        op_raw = (cfg.rerank_operator or "qdrant").strip().lower()
        op_norm = op_raw.replace("-", "_").replace(".", "_")
        self._rerank_operator: str = {
            "llamacpp": "llama_cpp",
            "llama_cpp": "llama_cpp",
            "vllm": "vllm",
            "ollama": "ollama",
            "qdrant": "qdrant",
        }.get(op_norm, op_norm)
        self._rerank_base_url: str = (cfg.rerank_base_url or cfg.qdrant_url).strip()
        self._reranker_model: str = (
            cfg.rerank_model
            or qdrant_cfg.get("reranker_model", "Xenova/ms-marco-MiniLM-L-6-v2")
        )
        self._positive_examples_limit: int = qdrant_cfg.get("positive_examples_limit", 5)

        self._client = QdrantClient(url=cfg.qdrant_url)
        self._rerank_client = (
            self._client
            if self._rerank_base_url.rstrip("/") == cfg.qdrant_url.rstrip("/")
            else QdrantClient(url=self._rerank_base_url)
        )
        self._embed_model = cfg.ollama_embed_model  # used as fastembed model name fallback

        self._ensure_collection()

    @staticmethod
    def _hit_to_result(hit: ScoredPoint) -> Dict[str, Any]:
        payload = hit.payload or {}
        return {
            "id": str(hit.id),
            "score": hit.score,
            "text": payload.get("text", ""),
            "label": payload.get("label", ""),
            "metadata": payload.get("metadata", {}),
        }

    def _ann_results(self, hits: List[ScoredPoint], limit: int) -> List[Dict[str, Any]]:
        return [self._hit_to_result(h) for h in hits[:limit]]

    @staticmethod
    def _normalize_rerank_score(score: float, operator: str) -> float:
        """Normalize external reranker score to [0,1] for stable thresholding.

        BGE cross-encoder style rerankers often return unbounded logits. For
        HTTP operators (llama.cpp/vLLM/Ollama), we map logits through sigmoid.
        """
        if operator in {"llama_cpp", "vllm", "ollama"}:
            # Clamp to avoid overflow in exp for extreme logits.
            clipped = max(min(float(score), 60.0), -60.0)
            return 1.0 / (1.0 + math.exp(-clipped))
        return float(score)

    def _rerank_with_http_operator(
        self,
        query_text: str,
        hits: List[ScoredPoint],
        limit: int,
        endpoints: List[str],
    ) -> List[Dict[str, Any]]:
        """Rerank candidates using HTTP rerank endpoints compatible with vLLM/llama.cpp/Ollama."""
        base = self._rerank_base_url.rstrip("/")
        resolved_endpoints = [f"{base}{ep}" if ep.startswith("/") else f"{base}/{ep}" for ep in endpoints]

        docs: List[str] = [str((h.payload or {}).get("text") or "") for h in hits]
        payload = {
            "model": self._reranker_model,
            "query": query_text,
            "documents": docs,
            "top_n": limit,
        }

        last_error: Exception | None = None
        for url in resolved_endpoints:
            try:
                resp = httpx.post(url, json=payload, timeout=30)
                # Try next known endpoint if this one is not implemented.
                if resp.status_code == 404:
                    continue
                resp.raise_for_status()
                data = resp.json()

                ranking = data.get("results") if isinstance(data, dict) else data
                if isinstance(data, dict) and not isinstance(ranking, list):
                    ranking = data.get("data")
                if not isinstance(ranking, list):
                    raise ValueError("Invalid rerank response format")

                rescored: List[Dict[str, Any]] = []
                for item in ranking:
                    if not isinstance(item, dict):
                        continue
                    idx = item.get("index", item.get("document_index"))
                    if idx is None:
                        continue
                    idx_int = int(idx)
                    if idx_int < 0 or idx_int >= len(hits):
                        continue

                    score = item.get("relevance_score", item.get("score", 0.0))
                    hit = hits[idx_int]
                    result = self._hit_to_result(hit)
                    result["score"] = self._normalize_rerank_score(float(score), self._rerank_operator)
                    result["raw_score"] = float(score)
                    rescored.append(result)

                if not rescored:
                    raise ValueError("Rerank response had no usable candidates")

                rescored.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
                return rescored[:limit]
            except Exception as exc:
                last_error = exc
                continue

        if last_error:
            raise last_error
        raise RuntimeError("No valid llama.cpp rerank endpoint available")

    # ── collection setup ──────────────────────────────────────────────

    def _ensure_collection(self) -> None:
        existing = [c.name for c in self._client.get_collections().collections]
        if self._collection not in existing:
            logger.info("qdrant.create_collection", name=self._collection)
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=self._vector_size,
                    distance=self._distance,
                    on_disk=True,
                ),
                hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
                optimizers_config=OptimizersConfigDiff(default_segment_number=2),
            )

    # ── embedding ─────────────────────────────────────────────────────

    def embed_text(self, text: str) -> List[float]:
        """
        Generate a dense vector for *text*.

        Tries Ollama first (configured model), falls back to fastembed locally.
        """
        cfg = get_settings()
        if cfg.llm_provider == "ollama":
            return self._embed_via_ollama(text)
        return self._embed_via_fastembed(text)

    def _embed_via_ollama(self, text: str) -> List[float]:
        import httpx
        cfg = get_settings()
        resp = httpx.post(
            f"{cfg.ollama_base_url}/api/embeddings",
            json={"model": cfg.ollama_embed_model, "prompt": text},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["embedding"]

    def _embed_via_fastembed(self, text: str) -> List[float]:
        from fastembed import TextEmbedding
        model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        return list(next(model.embed([text])))

    # ── rerank search ─────────────────────────────────────────────────

    def rerank_search(
        self,
        query_text: str,
        query_vector: List[float],
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        1. ANN search to get `limit * 3` candidates.
        2. Qdrant fast-rerank via fastembed cross-encoder.
        Returns up to `limit` results sorted by rerank score descending.
        """
        # Step 1: ANN  (qdrant-client >= 1.7 removed .search(); use .query_points())
        hits: List[ScoredPoint] = self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            limit=limit * 3,
            with_payload=True,
        ).points
        if not hits:
            return []

        # Step 2: Rerank via configured operator/backend
        if self._rerank_operator in {"none", "off", "ann", "ann_only"}:
            return self._ann_results(hits, limit)

        if self._rerank_operator in {"llama_cpp", "vllm", "ollama"}:
            operator_endpoints: Dict[str, List[str]] = {
                "llama_cpp": ["/v1/rerank", "/rerank"],
                "vllm": ["/v1/rerank", "/rerank"],
                "ollama": ["/api/rerank", "/v1/rerank", "/rerank"],
            }
            try:
                return self._rerank_with_http_operator(
                    query_text=query_text,
                    hits=hits,
                    limit=limit,
                    endpoints=operator_endpoints[self._rerank_operator],
                )
            except Exception as exc:
                logger.warning(
                    "rerank.fallback",
                    error=str(exc),
                    operator=self._rerank_operator,
                    base_url=self._rerank_base_url,
                    model=self._reranker_model,
                )
                return self._ann_results(hits, limit)

        if self._rerank_operator != "qdrant":
            logger.warning(
                "rerank.unknown_operator_fallback",
                operator=self._rerank_operator,
                fallback="ann_only",
            )
            return self._ann_results(hits, limit)

        # Qdrant rerank (requires fastembed on operator side)
        try:
            reranked = self._rerank_client.query_points(
                collection_name=self._collection,
                prefetch=[
                    {"query": query_vector, "limit": limit * 3}
                ],
                query={"type": "reranker", "query": query_text, "model": self._reranker_model},
                limit=limit,
                with_payload=True,
            )
            results = []
            for pt in reranked.points:
                item = self._hit_to_result(pt)
                item["score"] = pt.score
                results.append(item)
            return results
        except Exception as exc:
            # Fallback: return raw ANN results
            logger.warning(
                "rerank.fallback",
                error=str(exc),
                operator=self._rerank_operator,
                base_url=self._rerank_base_url,
                model=self._reranker_model,
            )
            return self._ann_results(hits, limit)

    # ── upsert example ────────────────────────────────────────────────

    def upsert_example(
        self,
        message_id: str,
        text: str,
        vector: List[float],
        label: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store a labelled example (positive or negative) in Qdrant."""
        import hashlib, uuid
        uid = str(uuid.uuid5(uuid.NAMESPACE_OID, message_id))
        self._client.upsert(
            collection_name=self._collection,
            points=[
                PointStruct(
                    id=uid,
                    vector=vector,
                    payload={
                        "text": text,
                        "label": label,
                        "message_id": message_id,
                        "metadata": metadata or {},
                    },
                )
            ],
        )
        logger.info("qdrant.upsert", message_id=message_id, label=label)
