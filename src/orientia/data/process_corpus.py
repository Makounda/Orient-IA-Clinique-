"""Traite corpusISPMNote → catalogue structuré, chunks RAG, registre des sources."""

from __future__ import annotations

import argparse
import json
import re
from datetime import date
from pathlib import Path

from orientia.config import METADATA_DIR, PROJECT_ROOT, ensure_dirs

RAW_CORPUS = PROJECT_ROOT / "data" / "raw" / "corpus" / "corpusISPMNote.txt"
# fallback si copié seulement à la racine
RAW_CORPUS_ALT = PROJECT_ROOT / "corpusISPMNote"
CORPUS_DIR = PROJECT_ROOT / "data" / "corpus"
STRUCTURED_PATH = CORPUS_DIR / "corpus_structured.json"
CHUNKS_PATH = CORPUS_DIR / "corpus_chunks.jsonl"
REGISTRE_PATH = METADATA_DIR / "registre_sources.json"


def _read_raw() -> tuple[str, Path]:
    for path in (RAW_CORPUS, RAW_CORPUS_ALT):
        if path.exists():
            return path.read_text(encoding="utf-8"), path
    raise FileNotFoundError("corpusISPMNote introuvable")


def _parse_matiere_blocks(text: str) -> list[dict]:
    """Extrait les blocs numérotés IGGLIA (1. Algorithmique …)."""
    # section matières détaillées commence après "1. Algorithmique"
    pattern = re.compile(
        r"(?m)^(\d+)\.\s+(.+?)\n(.*?)(?=^\d+\.\s+|\Z)",
        re.DOTALL,
    )
    blocks = []
    # limiter à la zone après "Les principales matières" et avant "23. Les passerelles"
    start = text.find("1. Algorithmique")
    end = text.find("23. Les passerelles")
    if start < 0:
        return blocks
    zone = text[start : end if end > 0 else None]

    for m in pattern.finditer(zone):
        num, title, body = m.group(1), m.group(2).strip(), m.group(3).strip()
        if title.startswith("Synthèse"):
            continue

        def _section(label: str) -> list[str]:
            # Compétences / Prérequis / Débouchés
            rx = re.compile(
                rf"(?is){label}\s*\n(.*?)(?=\n(?:Compétences|Prérequis|Débouchés|Relation|Passerelle|Un mémoire|C'est |\Z))",
            )
            sm = rx.search(body)
            if not sm:
                return []
            lines = []
            for line in sm.group(1).splitlines():
                line = line.strip().lstrip("•-").strip()
                if line and not line.startswith(("Relation", "Passerelle")):
                    lines.append(line)
            return lines

        blocks.append(
            {
                "id": f"IGGLIA_matiere_{num}",
                "parcours": ["IGGLIA"],
                "titre": title,
                "competences": _section("Compétences(?: développées)?"),
                "prerequis": _section("Prérequis"),
                "debouches": _section("Débouchés(?: professionnels| indirects)?"),
                "texte_brut": body[:2000],
            }
        )
    return blocks


def _matieres_principales_par_parcours(text: str) -> dict[str, str]:
    mapping = {}
    pairs = [
        ("IGGLIA", r"IGGLIA\n\n(.+?)(?=\n\nESIIA)"),
        ("ESIIA", r"ESIIA\n\n(.+?)(?=\n\nIMTICIA)"),
        ("IMTICIA", r"IMTICIA\n\n(.+?)(?=\n\nISAIA)"),
        ("ISAIA", r"ISAIA\n\n(.+?)(?=\n\nEMII)"),
        ("EMII", r"EMII\n\n(.+?)(?=\n\nGCA)"),
        ("GCA", r"GCA\n\n(.+?)(?=\n\nCAA)"),
        ("CAA_EMP_FIC_DTJA", r"CAA / EMP / FIC / DTJA\n\n(.+?)(?=\n\nIAA)"),
        ("IAA_AEE_PIP", r"IAA / AEE / PIP\n\n(.+?)(?=\n\nTEE)"),
        ("TEE_TEH", r"TEE / TEH\n\n(.+?)(?=\n\n1\. Algorithmique)"),
    ]
    for key, rx in pairs:
        m = re.search(rx, text, re.DOTALL)
        if m:
            mapping[key] = " ".join(m.group(1).split())
    # expand group keys
    out = {}
    for k, v in mapping.items():
        if k == "CAA_EMP_FIC_DTJA":
            for c in ("CAA", "EMP", "FIC", "DTJA"):
                out[c] = v
        elif k == "IAA_AEE_PIP":
            for c in ("IAA", "AEE", "PIP"):
                out[c] = v
        elif k == "TEE_TEH":
            for c in ("TEE", "TEH"):
                out[c] = v
        else:
            out[k] = v
    # ICMP: pas de paragraphe dédié — dérivé mention
    out.setdefault(
        "ICMP",
        "Mathématiques + chimie + physique + procédés industriels (mention Génie Industriel).",
    )
    return out


def build_structured(text: str, source_path: Path) -> dict:
    labels_path = METADATA_DIR / "parcours_labels.json"
    with labels_path.open(encoding="utf-8") as f:
        labels = json.load(f)

    matieres_detail = _parse_matiere_blocks(text)
    matieres_principales = _matieres_principales_par_parcours(text)

    return {
        "meta": {
            "titre": "Corpus pédagogique ISPM (compilé)",
            "fichier_source": str(source_path.relative_to(PROJECT_ROOT)),
            "date_traitement": date.today().isoformat(),
            "statut": "institutionnel_compile",
        },
        "catalogue": labels,
        "matieres_principales_par_parcours": matieres_principales,
        "matieres_detaillees": matieres_detail,
        "niveaux_diplomes": labels.get("niveaux", []),
        "passerelles": labels.get("passerelles", []),
    }


def build_chunks(structured: dict) -> list[dict]:
    chunks = []
    # un chunk par parcours (catalogue + matières principales)
    for mention in structured["catalogue"]["mentions"]:
        for p in mention["parcours"]:
            code = p["code"]
            mat = structured["matieres_principales_par_parcours"].get(code, "")
            text = (
                f"Parcours {code} — {p['nom']}. "
                f"Mention : {mention['nom']}. "
                f"Matières principales : {mat}. "
                f"Axes forts : {', '.join(p.get('axes_forts', []))}."
            )
            chunks.append(
                {
                    "chunk_id": f"parcours_{code}",
                    "type": "parcours",
                    "parcours": [code],
                    "mention": mention["code"],
                    "titre": p["nom"],
                    "text": text,
                    "source_id": "corpus_ispm_note_v1",
                }
            )

    for m in structured["matieres_detaillees"]:
        parts = [f"Matière {m['titre']} (contexte {', '.join(m['parcours'])})."]
        if m["competences"]:
            parts.append("Compétences : " + " ; ".join(m["competences"][:8]))
        if m["prerequis"]:
            parts.append("Prérequis : " + " ; ".join(m["prerequis"][:6]))
        if m["debouches"]:
            parts.append("Débouchés : " + " ; ".join(m["debouches"][:8]))
        chunks.append(
            {
                "chunk_id": m["id"],
                "type": "matiere",
                "parcours": m["parcours"],
                "titre": m["titre"],
                "text": " ".join(parts),
                "source_id": "corpus_ispm_note_v1",
            }
        )

    # chunk niveaux
    chunks.append(
        {
            "chunk_id": "niveaux_diplomes",
            "type": "structure",
            "parcours": [],
            "titre": "Niveaux de formation ISPM",
            "text": " ; ".join(structured.get("niveaux_diplomes") or []),
            "source_id": "corpus_ispm_note_v1",
        }
    )
    return chunks


def update_registre(source_path: Path, n_chunks: int, n_matieres: int) -> None:
    ensure_dirs()
    registre = []
    if REGISTRE_PATH.exists():
        registre = json.loads(REGISTRE_PATH.read_text(encoding="utf-8"))

    entry = {
        "source_id": "corpus_ispm_note_v1",
        "titre": "Notes corpus pédagogique ISPM (corpusISPMNote)",
        "origine_ou_url": "https://ispm-edu.com/publications.php",
        "origine_locale": str(source_path),
        "date_consultation": date.today().isoformat(),
        "statut": "institutionnel_compile",
        "donnees_extraites": [
            "mentions et parcours ISPM",
            "matières principales par parcours",
            "détail matières/compétences/prérequis/débouchés (surtout IGGLIA)",
            "passerelles IGGLIA ↔ ISAIA/ESIIA/IMTICIA",
            "niveaux Licence / Ingénieur / Doctorat",
        ],
        "limites_incertitudes": [
            "Compilation manuelle ; détail inégal selon les parcours",
            "ICMP peu documenté dans la note",
            "Ne pas confondre avec une maquette officielle complète",
        ],
        "n_chunks": n_chunks,
        "n_matieres_detaillees": n_matieres,
    }
    # replace if same source_id
    registre = [e for e in registre if e.get("source_id") != entry["source_id"]]
    registre.append(entry)
    REGISTRE_PATH.write_text(json.dumps(registre, indent=2, ensure_ascii=False), encoding="utf-8")


def process_corpus() -> dict:
    ensure_dirs()
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    text, source_path = _read_raw()
    # s'assurer que la copie raw existe
    RAW_CORPUS.parent.mkdir(parents=True, exist_ok=True)
    if not RAW_CORPUS.exists():
        RAW_CORPUS.write_text(text, encoding="utf-8")

    structured = build_structured(text, source_path)
    chunks = build_chunks(structured)

    STRUCTURED_PATH.write_text(
        json.dumps(structured, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")

    update_registre(source_path, len(chunks), len(structured["matieres_detaillees"]))
    return {
        "structured": str(STRUCTURED_PATH),
        "chunks": str(CHUNKS_PATH),
        "n_chunks": len(chunks),
        "n_matieres": len(structured["matieres_detaillees"]),
        "n_parcours": sum(len(m["parcours"]) for m in structured["catalogue"]["mentions"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Traite le corpus ISPM note")
    parser.parse_args()
    report = process_corpus()
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
