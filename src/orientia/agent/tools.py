"""Outils techniques de l'agent Orient'IA (opérations réelles, pas du prompt)."""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from orientia.config import load_parcours_codes, load_parcours_index
from orientia.ml.predict import predict_profile
from orientia.rag.index import search


def _parcours_meta(code: str) -> dict | None:
    code_u = code.upper().strip()
    for p in load_parcours_index():
        if p["code"] == code_u:
            return p
    return None


def rechercher_formation(query: str, top_k: int = 5) -> dict[str, Any]:
    """Recherche documentaire RAG sur le corpus pédagogique ISPM."""
    hits = search(query, top_k=max(top_k, 8))
    # Si un code parcours est cité, filtrer / prioriser ce parcours
    q_upper = query.upper()
    codes = [
        c
        for c in load_parcours_codes()
        if re.search(rf"\b{re.escape(c)}\b", q_upper)
    ]
    if codes:
        code = codes[0]
        focused = [
            h
            for h in hits
            if code in (h.parcours or []) or h.chunk_id == f"parcours_{code}"
        ]
        if not focused:
            from orientia.rag.index import SearchHit, load_index

            idx = load_index()
            for c in idx["chunks"]:
                if c.get("chunk_id") == f"parcours_{code}" or (
                    code in (c.get("parcours") or []) and c.get("type") == "parcours"
                ):
                    focused = [
                        SearchHit(
                            chunk_id=c.get("chunk_id", ""),
                            score=1.0,
                            titre=c.get("titre", ""),
                            text=c.get("text", ""),
                            type=c.get("type", ""),
                            parcours=list(c.get("parcours") or []),
                            source_id=c.get("source_id", ""),
                        )
                    ]
                    break
        extras = [
            h
            for h in hits
            if code in (h.parcours or []) and h.chunk_id not in {x.chunk_id for x in focused}
        ]
        hits = (focused + extras)[:top_k]

    if not hits:
        return {
            "tool": "rechercher_formation",
            "found": False,
            "message": (
                "Aucune information pertinente trouvée dans le corpus. "
                "Je ne peux pas inventer une formation ou une règle d'admission."
            ),
            "results": [],
            "source": "documents",
        }
    return {
        "tool": "rechercher_formation",
        "found": True,
        "results": [h.to_dict() for h in hits],
        "citations": [
            {
                "chunk_id": h.chunk_id,
                "titre": h.titre,
                "source_id": h.source_id,
                "score": round(h.score, 4),
            }
            for h in hits
        ],
        "source": "documents",
    }


def comparer_parcours(parcours_a: str, parcours_b: str) -> dict[str, Any]:
    """Compare deux parcours à partir du catalogue structuré + passages RAG."""
    a = parcours_a.upper().strip()
    b = parcours_b.upper().strip()
    codes = set(load_parcours_codes())
    missing = [c for c in (a, b) if c not in codes]
    if missing:
        return {
            "tool": "comparer_parcours",
            "ok": False,
            "message": (
                f"Parcours inconnu(s) dans le catalogue ISPM: {missing}. "
                "Je refuse d'inventer une filière."
            ),
            "source": "documents",
        }

    meta_a, meta_b = _parcours_meta(a), _parcours_meta(b)
    hits_a = search(a, top_k=3)
    hits_b = search(b, top_k=3)

    return {
        "tool": "comparer_parcours",
        "ok": True,
        "parcours_a": {
            "code": a,
            "nom": meta_a["nom"] if meta_a else a,
            "mention": meta_a.get("mention") if meta_a else None,
            "matieres_principales": meta_a.get("matieres_principales") if meta_a else [],
            "axes_forts": meta_a.get("axes_forts") if meta_a else [],
            "passages": [h.to_dict() for h in hits_a],
        },
        "parcours_b": {
            "code": b,
            "nom": meta_b["nom"] if meta_b else b,
            "mention": meta_b.get("mention") if meta_b else None,
            "matieres_principales": meta_b.get("matieres_principales") if meta_b else [],
            "axes_forts": meta_b.get("axes_forts") if meta_b else [],
            "passages": [h.to_dict() for h in hits_b],
        },
        "citations": [
            {"chunk_id": h.chunk_id, "source_id": h.source_id, "titre": h.titre}
            for h in hits_a + hits_b
        ],
        "source": "documents",
    }


def analyser_profil_ml(profil: dict, top_k: int = 3) -> dict[str, Any]:
    """Appelle le modèle ML d'orientation (outil obligatoire)."""
    if not isinstance(profil, dict) or not profil:
        return {
            "tool": "analyser_profil_ml",
            "ok": False,
            "message": "Profil manquant ou invalide. Indiquez des affinités (scores) ou compétences.",
            "source": "ml_model",
        }
    try:
        result = predict_profile(profil, top_k=top_k)
    except FileNotFoundError as exc:
        return {
            "tool": "analyser_profil_ml",
            "ok": False,
            "message": str(exc),
            "source": "ml_model",
        }
    result["tool"] = "analyser_profil_ml"
    result["ok"] = True
    result["source"] = "ml_model"
    return result


def identifier_debouches(parcours: str, top_k: int = 12) -> dict[str, Any]:
    """Identifie compétences / débouchés liés à un parcours via le corpus."""
    code = parcours.upper().strip()
    if code not in set(load_parcours_codes()):
        return {
            "tool": "identifier_debouches",
            "ok": False,
            "message": f"Parcours inconnu: {code}. Je ne peux pas inventer de débouchés.",
            "source": "documents",
        }

    meta = _parcours_meta(code)
    from orientia.config import CORPUS_DIR
    from orientia.rag.index import load_index

    # 1) Source prioritaire : corpusDebauche structuré
    structured_path = CORPUS_DIR / "debouches_structured.json"
    debouches: list[str] = []
    categories: dict[str, list[str]] = {}
    if structured_path.exists():
        data = json.loads(structured_path.read_text(encoding="utf-8"))
        for block in data.get("parcours") or []:
            if block.get("parcours") == code:
                debouches = list(block.get("debouches") or [])
                categories = dict(block.get("categories") or {})
                break

    # 2) Compléter / fallback via chunks RAG
    idx = load_index()
    parcours_chunks = [
        c
        for c in idx["chunks"]
        if code in (c.get("parcours") or [])
        or c.get("chunk_id") in {f"parcours_{code}", f"debouches_{code}"}
        or str(c.get("chunk_id", "")).startswith(f"debouches_{code}_")
    ]
    parcours_chunks.sort(
        key=lambda c: (
            0 if c.get("type") == "debouches" else 1,
            0 if re.search(r"débouch", c.get("text", ""), flags=re.I) else 1,
            0 if c.get("type") == "matiere" else 1,
        )
    )

    competences: list[str] = []
    used_chunks: list[dict] = []
    for c in parcours_chunks[:top_k]:
        text = c.get("text") or ""
        used_chunks.append(
            {
                "chunk_id": c.get("chunk_id"),
                "titre": c.get("titre"),
                "text": text,
                "type": c.get("type"),
                "parcours": c.get("parcours") or [],
                "source_id": c.get("source_id"),
                "score": 1.0 if c.get("type") == "debouches" else 0.5,
            }
        )
        if not debouches:
            m_deb = re.search(
                r"Débouchés[^:]*:\s*(.+?)(?:\s*Prérequis|\s*Relation|\s*Passerelle|\s*Compétences|\s*$)",
                text,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if m_deb:
                for item in re.split(r"\s*;\s*", m_deb.group(1)):
                    item = item.strip(" .")
                    if item and 2 < len(item) < 80 and item.lower() not in {
                        d.lower() for d in debouches
                    }:
                        debouches.append(item)

        m_comp = re.search(
            r"Compétences[^:]*:\s*(.+?)(?:\s*Prérequis|\s*Débouchés|\s*$)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if m_comp and len(competences) < 12:
            for item in re.split(r"\s*;\s*", m_comp.group(1)):
                item = item.strip(" .")
                if item and 2 < len(item) < 100 and item.lower() not in {
                    x.lower() for x in competences
                }:
                    competences.append(item)

    if not debouches and meta:
        debouches = [
            f"Orientation professionnelle liée aux axes : {', '.join(meta.get('axes_forts') or [])}",
        ]

    return {
        "tool": "identifier_debouches",
        "ok": True,
        "parcours": code,
        "nom": meta.get("nom") if meta else code,
        "mention": meta.get("mention") if meta else None,
        "categories": categories,
        "debouches": debouches[:40],
        "competences": competences[:12],
        "results": used_chunks[:8],
        "citations": [
            {
                "chunk_id": c["chunk_id"],
                "titre": c.get("titre"),
                "source_id": c.get("source_id"),
            }
            for c in used_chunks[:8]
        ],
        "source": "documents",
        "incertitude": None,
    }


TOOL_SPECS: dict[str, dict[str, Any]] = {
    "rechercher_formation": {
        "description": "Recherche dans le corpus pédagogique ISPM (RAG).",
        "fn": rechercher_formation,
        "params": ["query", "top_k"],
    },
    "comparer_parcours": {
        "description": "Compare deux parcours ISPM avec citations.",
        "fn": comparer_parcours,
        "params": ["parcours_a", "parcours_b"],
    },
    "analyser_profil_ml": {
        "description": "Recommande des parcours via le modèle ML entraîné.",
        "fn": analyser_profil_ml,
        "params": ["profil", "top_k"],
    },
    "identifier_debouches": {
        "description": "Liste compétences / débouchés documentés pour un parcours.",
        "fn": identifier_debouches,
        "params": ["parcours", "top_k"],
    },
}


def list_tools() -> list[dict[str, str]]:
    return [
        {"name": name, "description": spec["description"]}
        for name, spec in TOOL_SPECS.items()
    ]


def call_tool(name: str, **kwargs) -> dict[str, Any]:
    if name not in TOOL_SPECS:
        return {"ok": False, "message": f"Outil inconnu: {name}"}
    fn: Callable = TOOL_SPECS[name]["fn"]
    # filtrer kwargs supportés
    allowed = set(TOOL_SPECS[name]["params"])
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    return fn(**filtered)


def tools_as_json() -> str:
    return json.dumps(list_tools(), indent=2, ensure_ascii=False)
