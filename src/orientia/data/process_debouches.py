"""Intègre corpusDebauche → chunks structurés par parcours + mise à jour du corpus."""

from __future__ import annotations

import argparse
import json
import re
from datetime import date
from pathlib import Path

from orientia.config import CORPUS_CHUNKS_PATH, CORPUS_DIR, METADATA_DIR, PROJECT_ROOT, ensure_dirs
from orientia.data.process_corpus import process_corpus, update_registre

RAW_DEBAUCHE = PROJECT_ROOT / "data" / "raw" / "corpus" / "corpusDebauche.txt"
RAW_DEBAUCHE_ALT = PROJECT_ROOT / "corpusDebauche"
DEBAUCHE_STRUCTURED = CORPUS_DIR / "debouches_structured.json"

# Mapping titres / numéros → codes parcours
CODE_ALIASES = {
    "IGGLIA": "IGGLIA",
    "ESIIA": "ESIIA",
    "IMTICIA": "IMTICIA",
    "ISAIA": "ISAIA",
    "EMII": "EMII",
    "ICMP": "ICMP",
    "GCA": "GCA",
    "CAA": "CAA",
    "FIC": "FIC",
    "DTJA": "DTJA",
    "EMP": "EMP",
    "IAA": "IAA",
    "AEE": "AEE",
    "PIP": "PIP",
    "TEE": "TEE",
    "TEH": "TEH",
}


def _read_raw() -> tuple[str, Path]:
    for path in (RAW_DEBAUCHE, RAW_DEBAUCHE_ALT):
        if path.exists():
            return path.read_text(encoding="utf-8"), path
    raise FileNotFoundError("corpusDebauche introuvable")


def _strip_emoji_header(line: str) -> str:
    # enlève emojis / numérotation type "1.1 IGGLIA" ou "2. ESIIA"
    line = re.sub(r"^[\d.]+\s*", "", line.strip())
    line = re.sub(r"^[^\w]*", "", line)
    return line.strip()


def parse_debouches(text: str) -> list[dict]:
    """Découpe le fichier en blocs par code parcours."""
    lines = text.splitlines()
    blocks: list[dict] = []
    current_code: str | None = None
    current_lines: list[str] = []

    header_re = re.compile(
        r"(?:^|\s)(IGGLIA|ESIIA|IMTICIA|ISAIA|EMII|ICMP|GCA|CAA|FIC|DTJA|EMP|IAA|AEE|PIP|TEE|TEH)\b",
        re.IGNORECASE,
    )

    def flush():
        nonlocal current_code, current_lines
        if not current_code or not current_lines:
            current_code, current_lines = None, []
            return
        body = "\n".join(current_lines).strip()
        categories = _extract_categories(body)
        all_jobs = []
        for cat, jobs in categories.items():
            all_jobs.extend(jobs)
        seen = set()
        unique_jobs = []
        for j in all_jobs:
            key = j.lower()
            if key not in seen:
                seen.add(key)
                unique_jobs.append(j)
        # fusionner si le parcours existe déjà (ex. ICMP doublé)
        for existing in blocks:
            if existing["parcours"] == current_code:
                for cat, jobs in categories.items():
                    existing["categories"].setdefault(cat, [])
                    for j in jobs:
                        if j.lower() not in {x.lower() for x in existing["categories"][cat]}:
                            existing["categories"][cat].append(j)
                for j in unique_jobs:
                    if j.lower() not in {x.lower() for x in existing["debouches"]}:
                        existing["debouches"].append(j)
                existing["resume"] = _build_resume(
                    current_code, existing["categories"], existing["debouches"]
                )
                current_code, current_lines = None, []
                return
        blocks.append(
            {
                "parcours": current_code,
                "categories": categories,
                "debouches": unique_jobs,
                "texte_brut": body,
                "resume": _build_resume(current_code, categories, unique_jobs),
            }
        )
        current_code, current_lines = None, []

    for raw in lines:
        line = raw.strip()
        if not line:
            if current_code:
                current_lines.append("")
            continue

        # Nouvelle section parcours si la ligne contient un code connu en en-tête court
        cleaned = _strip_emoji_header(line)
        m = header_re.search(cleaned)
        is_header = False
        if m and len(cleaned) < 40:
            code = m.group(1).upper()
            # éviter de matcher "ICMP" deux fois dans le fichier (sections 6 et 15)
            is_header = True
            flush()
            current_code = CODE_ALIASES.get(code, code)
            current_lines = [cleaned]
            continue

        if current_code:
            current_lines.append(line)

    flush()
    return blocks


def _extract_categories(body: str) -> dict[str, list[str]]:
    """Extrait sous-catégories (Développement, Data, …) et métiers."""
    idx = re.search(r"Débouchés", body, flags=re.IGNORECASE)
    zone = body[idx.end() :] if idx else body

    CATEGORY_WHITELIST = {
        "développement",
        "systèmes d'information",
        "génie logiciel",
        "données / ia",
        "gestion de projet",
        "autres",
        "électronique",
        "informatique",
        "télécommunications",
        "systèmes embarqués",
        "ia",
        "maintenance",
        "multimédia",
        "web / digital",
        "communication numérique",
        "technologies",
        "statistiques",
        "data",
        "finance",
        "économie",
        "entreprises",
        "électromécanique",
        "automatisation",
        "industrie",
        "informatique industrielle",
        "énergie",
        "industrie chimique",
        "mines",
        "pétrole",
        "laboratoire",
        "hse",
        "construction",
        "architecture",
        "bâtiment",
        "infrastructures",
        "urbanisme",
        "bureau d'études",
        "commerce",
        "marketing",
        "administration",
        "management",
        "digital",
        "comptabilité",
        "audit",
        "contrôle",
        "banque",
        "fiscalité",
        "direction",
        "juridique",
        "entreprise",
        "banque / finance",
        "droit commercial",
        "projet",
        "institutions",
        "ong",
        "production",
        "qualité",
        "r&d",
        "sécurité alimentaire",
        "production / transformation",
        "entrepreneuriat",
        "agriculture",
        "élevage",
        "agrobusiness",
        "ong / développement rural",
        "coopératives",
        "industrie pharmaceutique",
        "recherche",
        "pharmacologie",
        "phytothérapie / plantes",
        "chimie",
        "environnement / sécurité",
        "gestion industrielle",
        "tourisme",
        "hôtellerie / tourisme",
        "environnement",
        "écotourisme",
        "patrimoine",
        "hôtellerie",
        "restauration",
        "événementiel",
    }

    categories: dict[str, list[str]] = {}
    current_cat = "Général"
    categories[current_cat] = []

    for line in zone.splitlines():
        line = line.strip()
        if not line or line.startswith("⚠️") or line.startswith("Les projets"):
            continue
        if line.lower() in {"débouchés", "debouches"}:
            continue
        key = line.lower().strip()
        if key in CATEGORY_WHITELIST:
            current_cat = line
            categories.setdefault(current_cat, [])
            continue
        categories.setdefault(current_cat, []).append(line)

    if not categories.get("Général"):
        categories.pop("Général", None)
    return {k: v for k, v in categories.items() if v}


def _build_resume(code: str, categories: dict[str, list[str]], jobs: list[str]) -> str:
    parts = [f"Débouchés du parcours {code}."]
    for cat, cat_jobs in categories.items():
        sample = ", ".join(cat_jobs[:6])
        more = f" (+{len(cat_jobs) - 6})" if len(cat_jobs) > 6 else ""
        parts.append(f"{cat} : {sample}{more}.")
    parts.append(f"Total métiers documentés : {len(jobs)}.")
    return " ".join(parts)


def build_debouche_chunks(blocks: list[dict]) -> list[dict]:
    chunks = []
    for b in blocks:
        code = b["parcours"]
        # chunk résumé global
        chunks.append(
            {
                "chunk_id": f"debouches_{code}",
                "type": "debouches",
                "parcours": [code],
                "titre": f"Débouchés {code}",
                "text": (
                    f"Parcours {code}. Débouchés : " + " ; ".join(b["debouches"][:40])
                    + (f" ; … ({len(b['debouches'])} au total)" if len(b["debouches"]) > 40 else "")
                    + ". "
                    + b["resume"]
                ),
                "source_id": "corpus_debouche_v1",
            }
        )
        # un chunk par catégorie pour un RAG plus précis
        for cat, jobs in b["categories"].items():
            slug = re.sub(r"[^a-zA-Z0-9]+", "_", cat.lower()).strip("_")[:40]
            chunks.append(
                {
                    "chunk_id": f"debouches_{code}_{slug}",
                    "type": "debouches",
                    "parcours": [code],
                    "titre": f"Débouchés {code} — {cat}",
                    "text": (
                        f"Parcours {code}, catégorie {cat}. "
                        f"Débouchés : " + " ; ".join(jobs) + "."
                    ),
                    "source_id": "corpus_debouche_v1",
                }
            )
    return chunks


def merge_chunks(base_chunks: list[dict], debouche_chunks: list[dict]) -> list[dict]:
    """Remplace / ajoute les chunks débouchés (source corpus_debouche_v1)."""
    kept = [c for c in base_chunks if not str(c.get("chunk_id", "")).startswith("debouches_")]
    # aussi retirer anciens si re-run
    return kept + debouche_chunks


def process_debouches() -> dict:
    ensure_dirs()
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    text, source_path = _read_raw()
    RAW_DEBAUCHE.parent.mkdir(parents=True, exist_ok=True)
    if not RAW_DEBAUCHE.exists():
        RAW_DEBAUCHE.write_text(text, encoding="utf-8")

    # s'assurer que le corpus principal existe
    if not CORPUS_CHUNKS_PATH.exists():
        process_corpus()

    blocks = parse_debouches(text)
    DEBAUCHE_STRUCTURED.write_text(
        json.dumps(
            {
                "meta": {
                    "titre": "Débouchés ISPM par parcours",
                    "fichier_source": str(source_path),
                    "url_publique": "https://ispm-edu.com/publications.php",
                    "date_traitement": date.today().isoformat(),
                },
                "parcours": blocks,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    debouche_chunks = build_debouche_chunks(blocks)
    base = []
    with CORPUS_CHUNKS_PATH.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                base.append(json.loads(line))
    merged = merge_chunks(base, debouche_chunks)
    with CORPUS_CHUNKS_PATH.open("w", encoding="utf-8") as f:
        for ch in merged:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")

    # registre
    registre_path = METADATA_DIR / "registre_sources.json"
    registre = []
    if registre_path.exists():
        registre = json.loads(registre_path.read_text(encoding="utf-8"))
    entry = {
        "source_id": "corpus_debouche_v1",
        "titre": "Débouchés ISPM par parcours (corpusDebauche)",
        "origine_ou_url": "https://ispm-edu.com/publications.php",
        "origine_locale": str(source_path),
        "date_consultation": date.today().isoformat(),
        "statut": "institutionnel_compile",
        "donnees_extraites": [
            f"débouchés pour {len(blocks)} parcours",
            "catégories métier par filière",
        ],
        "limites_incertitudes": [
            "Professions réglementées (avocat, pharmacien, etc.) : diplôme seul insuffisant",
            "Certains métiers nécessitent spécialisation / expérience",
        ],
        "n_parcours": len(blocks),
        "n_chunks_debouches": len(debouche_chunks),
    }
    registre = [e for e in registre if e.get("source_id") != entry["source_id"]]
    registre.append(entry)
    registre_path.write_text(json.dumps(registre, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "n_parcours": len(blocks),
        "n_chunks_debouches": len(debouche_chunks),
        "n_chunks_total": len(merged),
        "parcours": [b["parcours"] for b in blocks],
        "structured": str(DEBAUCHE_STRUCTURED),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Intègre corpusDebauche dans le RAG Orient'IA")
    parser.parse_args()
    report = process_debouches()
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
