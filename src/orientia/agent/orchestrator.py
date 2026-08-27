"""Agent conversationnel Orient'IA : routage d'intentions → outils → réponse argumentée."""

from __future__ import annotations

import argparse
import json
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from orientia.agent.tools import (
    analyser_profil_ml,
    comparer_parcours,
    identifier_debouches,
    list_tools,
    rechercher_formation,
)
from orientia.config import TRACES_PATH, ensure_dirs, load_parcours_codes
from orientia.data.schema import empty_profile_dict
from orientia.rag.index import build_index, load_index

DISCLAIMER = (
    "ORIENT'IA constitue un outil d'aide à l'orientation. Ses recommandations ne remplacent "
    "ni l'avis d'un conseiller pédagogique ni une décision officielle d'admission."
)

OFFICIAL_SOURCE_URL = "https://ispm-edu.com/publications.php"

INJECTION_PATTERNS = [
    r"ignore\s+(les\s+)?(documents|instructions|sources)",
    r"oublie\s+(tes|les)\s+(règles|instructions)",
    r"affirme\s+qu['e ].*filière",
    r"invente\s+(une\s+)?(filière|formation|parcours)",
    r"nouvelle\s+filière",
    r"system\s*prompt",
    r"jailbreak",
]

SENSITIVE_REC_PATTERNS = [
    r"uniquement\s+(à\s+partir\s+)?(du|de\s+la|des)\s+(sexe|genre|âge|age|origine|religion)",
    r"selon\s+(le\s+)?(sexe|genre|âge|age)",
    r"parce\s+qu['e ]?(il|elle)\s+est\s+(un|une)\s+",
]

PSYCHO_PATTERNS = [
    r"analyse\s+(ma|mon)\s+personnalité",
    r"profil\s+psychologique",
    r"traits?\s+de\s+personnalité",
    r"d['e ]après\s+mes\s+messages",
    r"infère?\s+(mon|ma)\s+",
]

SCORE_ALIASES = {
    "maths": "score_maths",
    "mathématiques": "score_maths",
    "mathematiques": "score_maths",
    "programmation": "score_prog",
    "prog": "score_prog",
    "code": "score_prog",
    "stats": "score_stats",
    "statistiques": "score_stats",
    "données": "score_stats",
    "data": "score_stats",
    "design": "score_design",
    "multimédia": "score_design",
    "multimedia": "score_design",
    "électronique": "score_electronique",
    "electronique": "score_electronique",
    "gestion": "score_gestion",
    "physique": "score_physique",
    "mécanique": "score_physique",
    "biologie": "score_sciences_vie",
    "chimie": "score_sciences_vie",
    "langues": "score_langues",
    "droit": "score_droit_eco",
    "économie": "score_droit_eco",
    "economie": "score_droit_eco",
    "finance": "score_droit_eco",
}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_trace(trace: dict) -> None:
    ensure_dirs()
    with TRACES_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(trace, ensure_ascii=False) + "\n")


def security_check(message: str) -> dict | None:
    low = message.lower()
    for pat in INJECTION_PATTERNS:
        if re.search(pat, low):
            return {
                "refuse": True,
                "reason": "prompt_injection",
                "answer": (
                    "Je refuse d'ignorer les documents officiels ou d'inventer une filière. "
                    "Je m'appuie uniquement sur le corpus ISPM et le modèle ML documentés. "
                    + DISCLAIMER
                ),
            }
    for pat in SENSITIVE_REC_PATTERNS:
        if re.search(pat, low):
            return {
                "refuse": True,
                "reason": "sensitive_criteria",
                "answer": (
                    "Je ne recommande pas un parcours sur la base du sexe, de l'âge ou d'autres "
                    "caractéristiques personnelles sensibles. "
                    + DISCLAIMER
                ),
            }
    for pat in PSYCHO_PATTERNS:
        if re.search(pat, low):
            return {
                "refuse": True,
                "reason": "psychological_profiling",
                "answer": (
                    "Je n'analyse pas la personnalité ni n'infère de traits psychologiques. "
                    "Seuls les intérêts et préférences que vous déclarez explicitement sont pris en compte. "
                    + DISCLAIMER
                ),
            }
    return None


def extract_parcours_codes(text: str) -> list[str]:
    """Retourne les codes dans l'ordre d'apparition dans le texte."""
    upper = text.upper()
    found: list[tuple[int, str]] = []
    for code in load_parcours_codes():
        m = re.search(rf"\b{re.escape(code)}\b", upper)
        if m:
            found.append((m.start(), code))
    found.sort(key=lambda x: x[0])
    return [c for _, c in found]


def extract_profile_hints(text: str, existing: dict | None = None) -> dict:
    """Extrait un profil partiel depuis un message libre (scores déclarés)."""
    profile = empty_profile_dict()
    if existing:
        profile.update({k: v for k, v in existing.items() if v is not None})

    low = text.lower().replace("’", "'")

    likes = bool(re.search(r"\b(j'aime|aime|adore|fort(?:e)?\s+en|intéresse|interesse|passionn)\b", low))
    dislikes = bool(re.search(r"\b(peu|pas beaucoup|déteste|deteste|n'aime pas|naime pas)\b", low))

    for alias, feat in SCORE_ALIASES.items():
        if not re.search(rf"\b{re.escape(alias)}\b", low):
            continue
        # fenêtre locale autour de l'alias
        for m in re.finditer(rf".{{0,40}}\b{re.escape(alias)}\b.{{0,20}}", low):
            window = m.group(0)
            if re.search(r"\b(peu|pas|déteste|deteste|moins)\b", window):
                profile[feat] = min(float(profile.get(feat) or 10), 2.0)
            elif re.search(r"\b(aime|adore|fort|intéresse|interesse|beaucoup)\b", window) or likes:
                # si mention dans une phrase d'affinité globale sans négation locale
                if not re.search(r"\b(peu|pas|déteste|deteste)\b", window):
                    profile[feat] = max(float(profile.get(feat) or 0), 8.0)

    if "python" in low:
        profile["comp_python"] = 1
    if re.search(r"\b(web|javascript|php)\b", low):
        profile["comp_web"] = 1
    if re.search(r"\b(data|statistique|analyse de donn|machine learning|\bia\b)\b", low):
        profile["comp_data"] = 1
        profile["comp_python"] = 1
        profile["interet_ia"] = 1
        profile["score_stats"] = max(float(profile.get("score_stats") or 0), 8.0)
        profile["pref_professionnelle"] = "data_science"
    if re.search(r"\b(électronique|embarqué|embarque|hardware)\b", low):
        profile["comp_hardware"] = 1
        profile["score_electronique"] = max(float(profile.get("score_electronique") or 0), 7.0)
    if re.search(r"\b(design|multimédia|multimedia|ux)\b", low) and not re.search(
        r"\b(peu|pas).{0,20}\b(design|multimédia|multimedia)\b", low
    ):
        profile["comp_design"] = 1
        profile["interet_multimedia"] = 1

    if re.search(r"\b(programmation|prog|code)\b", low) and likes:
        profile["score_prog"] = max(float(profile.get("score_prog") or 0), 8.0)
        profile["comp_python"] = max(int(profile.get("comp_python") or 0), 1)
    if re.search(r"\b(maths|mathématiques|mathematiques)\b", low) and likes:
        profile["score_maths"] = max(float(profile.get("score_maths") or 0), 8.0)
    if dislikes and re.search(r"\b(peu|pas).{0,15}\b(design|multimédia|multimedia|interface)\b", low):
        profile["score_design"] = 2.0
        profile["comp_design"] = 0
        profile["interet_multimedia"] = 0

    if float(profile.get("note_moyenne") or 0) <= 0:
        profile["note_moyenne"] = 13.0

    profile.pop("parcours_recommande", None)
    return profile


def profile_completeness(profile: dict) -> tuple[float, list[str]]:
    keys = [
        "score_maths",
        "score_prog",
        "score_stats",
        "score_design",
        "score_electronique",
        "score_gestion",
    ]
    filled = [k for k in keys if float(profile.get(k) or 0) > 0]
    missing = [k for k in keys if k not in filled]
    return len(filled) / len(keys), missing


def detect_intent(message: str) -> str:
    low = message.lower()
    codes = extract_parcours_codes(message)
    if re.search(r"\bcompar(e|er|aison)\b", low) and len(codes) >= 2:
        # multi-étapes : comparaison + débouchés
        if re.search(r"\b(débouchés?|debouches?|métiers?)\b", low):
            return "compare_debouches"
        return "compare"
    if re.search(
        r"\b(données réelles|données générées|synthétiques?|provenance|enquête)\b",
        low,
    ):
        return "provenance"
    if re.search(
        r"\b(frais de scolarité|astronautique|date exacte du concours|concours d'entrée|"
        r"filière secrète|horaires exacts d'inscription)\b",
        low,
    ):
        return "absent"
    if re.search(
        r"\b(débouchés?|debouches?|métiers?|metiers?|compétences?|competences?)\b",
        low,
    ):
        if re.search(r"\b(recommand|oriente|j['e ]aime)\b", low):
            return "recommend_debouches"
        return "debouches"
    if re.search(
        r"\b(recommand|oriente|quel parcours|quels parcours|profil|j['e ]aime|correspond)\b",
        low,
    ):
        return "recommend"
    if re.search(r"\b(niveau|licence|ingénieur|ingenieur|doctorat|admission)\b", low):
        return "factual"
    if codes or re.search(r"\b(matière|matiere|formation|parcours|présente|presente)\b", low):
        return "factual"
    if re.search(r"\b(manque|incertitude|fiable|pourquoi|ne sais pas|un peu de tout)\b", low):
        return "meta"
    if len(low.split()) <= 8 and not codes:
        return "meta"
    return "factual"


def _format_citations(tool_outputs: list[dict]) -> list[dict]:
    cites = []
    seen = set()
    for out in tool_outputs:
        for c in out.get("citations") or []:
            key = c.get("chunk_id") or c.get("titre")
            if key and key not in seen:
                seen.add(key)
                cites.append(c)
        for r in out.get("results") or []:
            key = r.get("chunk_id")
            if key and key not in seen:
                seen.add(key)
                cites.append(
                    {
                        "chunk_id": key,
                        "titre": r.get("titre"),
                        "source_id": r.get("source_id"),
                        "score": r.get("score"),
                    }
                )
    return cites


def _summarize_parcours_hit(r: dict) -> str:
    """Résumé court d'un chunk sans dump brut."""
    titre = r.get("titre") or r.get("chunk_id")
    text = r.get("text") or ""
    if (r.get("type") == "parcours") or str(r.get("chunk_id", "")).startswith("parcours_"):
        head = text.split("Matières principales")[0].strip()
        mats = ""
        if "Matières principales" in text:
            mats = text.split("Matières principales", 1)[1].split("Axes forts")[0].strip(" :.")
            return f"**{titre}** — {head} Matières principales : {mats}."
        return f"**{titre}** — {head}"
    m = re.search(
        r"Compétences[^:]*:\s*(.+?)(?:\s*Prérequis|\s*Débouchés|\s*$)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m:
        comps = [x.strip() for x in m.group(1).split(";") if x.strip()][:4]
        return f"**{titre}** : " + " ; ".join(comps)
    return f"**{titre}** : {text[:180].rstrip()}…"


def synthesize(intent: str, message: str, tool_outputs: list[dict], profile: dict) -> str:
    parts: list[str] = []

    for out in tool_outputs:
        tool = out.get("tool")
        if tool == "comparer_parcours" and out.get("ok"):
            a, b = out["parcours_a"], out["parcours_b"]
            parts.append(
                f"**Comparaison {a['code']} vs {b['code']}** (sources documentaires)\n\n"
                f"- **{a['code']}** ({a.get('mention')}) : {a.get('nom')}.\n"
                f"  - Axes : {', '.join(a.get('axes_forts') or [])}.\n"
                f"  - Matières : {', '.join(a.get('matieres_principales') or [])}.\n"
                f"- **{b['code']}** ({b.get('mention')}) : {b.get('nom')}.\n"
                f"  - Axes : {', '.join(b.get('axes_forts') or [])}.\n"
                f"  - Matières : {', '.join(b.get('matieres_principales') or [])}."
            )
        elif tool == "analyser_profil_ml" and out.get("ok"):
            ranking = out.get("top_k") or []
            lines = ", ".join(f"{r['parcours']} ({r['score']:.0%})" for r in ranking)
            parts.append(
                f"**Recommandation du modèle ML** (`{out.get('model_name')}`) : "
                f"**{out.get('prediction')}**.\n\n"
                f"Top propositions : {lines}.\n\n"
                "Cette sortie provient du modèle statistique, distincte des documents."
            )
            completeness, missing = profile_completeness(profile)
            if completeness < 0.5:
                parts.append(
                    "Profil encore partiel — pour fiabiliser, précisez vos affinités en "
                    + ", ".join(missing[:4])
                    + "."
                )
        elif tool == "analyser_profil_ml" and not out.get("ok"):
            parts.append(out.get("message") or "Analyse ML impossible.")
        elif tool == "identifier_debouches" and out.get("ok"):
            code = out.get("parcours")
            nom = out.get("nom") or code
            deb = out.get("debouches") or []
            comps = out.get("competences") or []
            categories = out.get("categories") or {}
            block = [
                f"Pour le parcours **{code}** ({nom})"
                + (f", mention {out.get('mention')}" if out.get("mention") else "")
                + ", voici les débouchés documentés :"
            ]
            if categories:
                for cat, jobs in list(categories.items())[:8]:
                    sample = ", ".join(jobs[:8])
                    extra = f" (+{len(jobs) - 8} autres)" if len(jobs) > 8 else ""
                    block.append(f"\n**{cat}** : {sample}{extra}")
            elif deb:
                block.append("\n**Débouchés / métiers cités :**")
                block.extend(f"- {d}" for d in deb[:15])
            if comps:
                block.append("\n**Compétences associées (extrait) :**")
                block.extend(f"- {c}" for c in comps[:8])
            if not deb and not categories:
                block.append(
                    "\nLe corpus ne contient pas assez de détail sur les débouchés de ce parcours. "
                    "Je ne peux pas inventer de métiers."
                )
            if out.get("incertitude"):
                block.append(f"\n_Limite corpus :_ {out['incertitude']}")
            parts.append("\n".join(block))
        elif tool == "identifier_debouches" and not out.get("ok"):
            parts.append(out.get("message") or "Impossible d'identifier les débouchés.")
        elif tool == "rechercher_formation":
            if not out.get("found"):
                parts.append(out.get("message") or "Information absente du corpus.")
            else:
                results = out.get("results") or []
                intro = []
                details = []
                for r in results[:5]:
                    summary = _summarize_parcours_hit(r)
                    if str(r.get("chunk_id", "")).startswith("parcours_") or r.get("type") == "parcours":
                        intro.append(summary)
                    else:
                        details.append(summary)
                block = ["Voici une synthèse à partir du corpus pédagogique :"]
                if intro:
                    block.append("\n" + "\n".join(intro))
                if details:
                    block.append("\n**Points complémentaires :**")
                    block.extend(f"- {d}" for d in details[:4])
                parts.append("\n".join(block))
        elif out.get("message"):
            parts.append(out["message"])

    if intent == "meta":
        parts.append(
            "Pour une recommandation plus fiable, j'ai besoin d'affinités déclarées "
            "(maths, programmation, stats, design, électronique, gestion, etc.) et de vos contraintes. "
            "Je distingue toujours : résultats ML / documents / règles."
        )

    if not parts:
        parts.append(
            "Je n'ai pas assez d'information dans le corpus pour conclure. "
            "Reformulez ou précisez un parcours ISPM connu (ex. IGGLIA, ISAIA)."
        )

    cites = _format_citations(tool_outputs)
    # Affichage utilisateur : toujours la source officielle ISPM (pas de dump de chunks)
    has_doc_source = any(
        (out.get("source") == "documents")
        or out.get("tool")
        in {"rechercher_formation", "comparer_parcours", "identifier_debouches"}
        for out in tool_outputs
    )
    if has_doc_source or cites:
        parts.append(f"**Source :** {OFFICIAL_SOURCE_URL}")

    parts.append(DISCLAIMER)
    return "\n\n".join(parts)


def run_agent(
    message: str,
    profile: dict | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Exécute un tour d'agent avec outils + trace."""
    t0 = time.perf_counter()
    ensure_dirs()
    # s'assurer que l'index existe
    load_index()

    session_id = session_id or str(uuid.uuid4())
    profile = extract_profile_hints(message, profile or {})
    security = security_check(message)
    tools_called: list[dict] = []
    intent = "blocked"

    if security:
        answer = security["answer"]
        refusal = security["reason"]
    else:
        refusal = None
        intent = detect_intent(message)
        codes = extract_parcours_codes(message)

        if intent == "provenance":
            tools_called.append(
                {
                    "tool": "meta_provenance",
                    "ok": True,
                    "source": "systeme",
                    "message": (
                        "Le modèle ML d'ORIENT'IA est entraîné principalement sur des **données synthétiques** "
                        "générées à partir de règles alignées sur le corpus ISPM "
                        "(`data/processed/profiles_synthetic.csv`). "
                        "Les réponses d'**enquête réelle** anonymisées, lorsqu'elles sont disponibles, "
                        "servent de validation/test (`data/processed/profiles_survey.csv`). "
                        "Les informations sur les formations et débouchés proviennent du corpus documentaire "
                        f"({OFFICIAL_SOURCE_URL}). "
                        "Une recommandation n'est donc pas une décision administrative."
                    ),
                }
            )
        elif intent == "absent":
            tools_called.append(
                {
                    "tool": "meta_absent",
                    "ok": True,
                    "source": "systeme",
                    "message": (
                        "Cette information précise (frais exacts, dates de concours, filière non catalogue) "
                        "est **absente du corpus** actuel. Je ne peux pas l'inventer. "
                        "Adressez-vous à l'administration ISPM pour une validation officielle. "
                        f"Source de référence pédagogique : {OFFICIAL_SOURCE_URL}."
                    ),
                }
            )
        elif intent == "compare_debouches" and len(codes) >= 2:
            tools_called.append(comparer_parcours(codes[0], codes[1]))
            tools_called.append(identifier_debouches(codes[1]))
        elif intent == "compare" and len(codes) >= 2:
            out = comparer_parcours(codes[0], codes[1])
            tools_called.append(out)
        elif intent == "debouches":
            code = codes[0] if codes else "IGGLIA"
            tools_called.append(identifier_debouches(code))
        elif intent == "recommend_debouches":
            completeness, _ = profile_completeness(profile)
            if completeness < 0.25:
                tools_called.append(
                    {
                        "tool": "analyser_profil_ml",
                        "ok": False,
                        "message": (
                            "Profil trop incomplet pour une recommandation ML fiable. "
                            "Indiquez par exemple: « j'aime les maths, la programmation et peu le design »."
                        ),
                        "source": "ml_model",
                    }
                )
            else:
                ml = analyser_profil_ml(profile, top_k=3)
                tools_called.append(ml)
                top = ml.get("prediction")
                if top:
                    tools_called.append(identifier_debouches(top))
        elif intent == "recommend":
            completeness, _ = profile_completeness(profile)
            if completeness < 0.25:
                tools_called.append(
                    {
                        "tool": "analyser_profil_ml",
                        "ok": False,
                        "message": (
                            "Profil trop incomplet pour une recommandation ML fiable. "
                            "Indiquez par exemple: « j'aime les maths, la programmation et peu le design »."
                        ),
                        "source": "ml_model",
                    }
                )
                tools_called.append(rechercher_formation(message, top_k=4))
            else:
                tools_called.append(analyser_profil_ml(profile, top_k=3))
                top = tools_called[-1].get("prediction")
                if top:
                    tools_called.append(rechercher_formation(f"parcours {top}", top_k=3))
        elif intent == "meta":
            tools_called.append(
                {
                    "tool": "meta_clarification",
                    "ok": True,
                    "source": "systeme",
                    "message": (
                        "Votre demande est trop vague ou le profil est incomplet. "
                        "Précisez un parcours ISPM (ex. IGGLIA, ISAIA) ou des affinités "
                        "(maths, programmation, stats, design, électronique, gestion…). "
                        "Exemple : « j'aime les maths et la data, peu le design »."
                    ),
                }
            )
        else:
            tools_called.append(rechercher_formation(message, top_k=5))
            if len(codes) >= 2:
                tools_called.append(comparer_parcours(codes[0], codes[1]))

        answer = synthesize(intent, message, tools_called, profile)

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    citations_internal = _format_citations(tools_called)
    public_citations = (
        [{"url": OFFICIAL_SOURCE_URL, "label": "ISPM — Publications"}]
        if any(
            t.get("source") == "documents"
            or t.get("tool")
            in {"rechercher_formation", "comparer_parcours", "identifier_debouches"}
            for t in tools_called
        )
        else []
    )
    trace = {
        "trace_id": str(uuid.uuid4()),
        "session_id": session_id,
        "timestamp": _utcnow(),
        "question": message,
        "intent": intent,
        "refusal": refusal,
        "profile": {k: v for k, v in profile.items() if k.startswith(("score_", "comp_", "interet_", "pref_"))},
        "tools_called": [
            {
                "tool": t.get("tool"),
                "source": t.get("source"),
                "ok": t.get("ok", t.get("found")),
                "summary": {
                    k: t.get(k)
                    for k in ("prediction", "parcours", "message")
                    if k in t
                },
            }
            for t in tools_called
        ],
        "tool_outputs": tools_called,
        "citations_internal": citations_internal,
        "citations": public_citations,
        "answer": answer,
        "elapsed_ms": elapsed_ms,
    }
    append_trace(trace)

    return {
        "answer": answer,
        "intent": intent,
        "profile": profile,
        "tools_called": [t.get("tool") for t in tools_called],
        "tool_outputs": tools_called,
        "citations": public_citations,
        "refusal": refusal,
        "elapsed_ms": elapsed_ms,
        "session_id": session_id,
        "disclaimer": DISCLAIMER,
        "trace_id": trace["trace_id"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent Orient'IA (CLI)")
    parser.add_argument("--build-index", action="store_true")
    parser.add_argument("--message", type=str, default="")
    parser.add_argument("--list-tools", action="store_true")
    parser.add_argument("--profile", type=str, default="")
    args = parser.parse_args()

    if args.build_index:
        print(json.dumps(build_index(), indent=2, ensure_ascii=False))
    if args.list_tools:
        print(json.dumps(list_tools(), indent=2, ensure_ascii=False))
    if args.message:
        profile = json.loads(args.profile) if args.profile else None
        result = run_agent(args.message, profile=profile)
        print(json.dumps(
            {
                "answer": result["answer"],
                "intent": result["intent"],
                "tools_called": result["tools_called"],
                "citations": result["citations"],
                "elapsed_ms": result["elapsed_ms"],
                "trace_id": result["trace_id"],
            },
            indent=2,
            ensure_ascii=False,
        ))


if __name__ == "__main__":
    main()
