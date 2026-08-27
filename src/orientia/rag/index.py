"""Index RAG lexical (TF-IDF) + boosts structurés sur le corpus ISPM."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from orientia.config import (
    CORPUS_CHUNKS_PATH,
    RAG_INDEX_PATH,
    ensure_dirs,
    load_parcours_codes,
)


@dataclass
class SearchHit:
    chunk_id: str
    score: float
    titre: str
    text: str
    type: str
    parcours: list
    source_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "score": round(float(self.score), 4),
            "titre": self.titre,
            "text": self.text,
            "type": self.type,
            "parcours": self.parcours,
            "source_id": self.source_id,
        }


def load_chunks(path: Path | None = None) -> list[dict]:
    chunks_path = path or CORPUS_CHUNKS_PATH
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Chunks introuvables: {chunks_path}. Lancez: python -m orientia.data.process_corpus"
        )
    chunks = []
    with chunks_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s+/À-ÿ'-]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def build_index(chunks_path: Path | None = None, out_path: Path | None = None) -> dict:
    ensure_dirs()
    chunks = load_chunks(chunks_path)
    documents = [_normalize(c.get("text", "") + " " + c.get("titre", "")) for c in chunks]
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(documents)
    artifact = {
        "vectorizer": vectorizer,
        "matrix": matrix,
        "chunks": chunks,
        "n_chunks": len(chunks),
    }
    dest = out_path or RAG_INDEX_PATH
    joblib.dump(artifact, dest)
    return {"path": str(dest), "n_chunks": len(chunks)}


def load_index(path: Path | None = None) -> dict:
    index_path = path or RAG_INDEX_PATH
    if not index_path.exists():
        build_index(out_path=index_path)
    return joblib.load(index_path)


def _detect_parcours_codes(query: str) -> list[str]:
    q = query.upper()
    found = []
    for code in load_parcours_codes():
        if re.search(rf"\b{re.escape(code)}\b", q, flags=re.IGNORECASE):
            found.append(code)
    return found


def search(
    query: str,
    top_k: int = 5,
    index: dict | None = None,
    type_filter: str | None = None,
) -> list[SearchHit]:
    """Recherche hybride : similarité TF-IDF + boost si code parcours mentionné."""
    if not query or not query.strip():
        return []

    idx = index or load_index()
    vectorizer: TfidfVectorizer = idx["vectorizer"]
    matrix = idx["matrix"]
    chunks: list[dict] = idx["chunks"]

    q_vec = vectorizer.transform([_normalize(query)])
    sims = cosine_similarity(q_vec, matrix).ravel()

    codes = _detect_parcours_codes(query)
    for i, chunk in enumerate(chunks):
        boost = 0.0
        parcours = chunk.get("parcours") or []
        if codes and any(c in parcours for c in codes):
            boost += 0.25
        if codes and chunk.get("chunk_id") in {f"parcours_{c}" for c in codes}:
            boost += 0.35
        if type_filter and chunk.get("type") != type_filter:
            sims[i] = -1.0
            continue
        # boost lexical exact titre
        titre = (chunk.get("titre") or "").lower()
        if titre and titre in query.lower():
            boost += 0.15
        sims[i] = float(sims[i]) + boost

    order = np.argsort(sims)[::-1]
    hits: list[SearchHit] = []
    for i in order:
        if sims[i] <= 0:
            continue
        c = chunks[int(i)]
        hits.append(
            SearchHit(
                chunk_id=c.get("chunk_id", ""),
                score=float(sims[i]),
                titre=c.get("titre", ""),
                text=c.get("text", ""),
                type=c.get("type", ""),
                parcours=list(c.get("parcours") or []),
                source_id=c.get("source_id", ""),
            )
        )
        if len(hits) >= top_k:
            break
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(description="Construit / teste l'index RAG Orient'IA")
    parser.add_argument("--build", action="store_true", help="Reconstruit l'index")
    parser.add_argument("--query", type=str, default="", help="Requête de test")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    if args.build or not RAG_INDEX_PATH.exists():
        report = build_index()
        print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.query:
        hits = search(args.query, top_k=args.top_k)
        print(json.dumps([h.to_dict() for h in hits], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
