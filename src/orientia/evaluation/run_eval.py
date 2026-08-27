"""Évaluation bout-en-bout Orient'IA (≥ 32 cas du sujet)."""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from orientia.agent.orchestrator import OFFICIAL_SOURCE_URL, run_agent
from orientia.config import ARTIFACTS_DIR, PROJECT_ROOT, ensure_dirs
from orientia.rag.index import build_index, load_index

TEST_CASES_PATH = PROJECT_ROOT / "evaluation" / "test_cases.json"
EVAL_DIR = ARTIFACTS_DIR / "evaluation"
RESULTS_JSON = EVAL_DIR / "eval_results.json"
RESULTS_MD = EVAL_DIR / "eval_report.md"


def _norm(s: str) -> str:
    return (s or "").lower()


def check_case(case: dict, result: dict) -> dict[str, Any]:
    expect = case.get("expect") or {}
    answer = result.get("answer") or ""
    tools = result.get("tools_called") or []
    intent = result.get("intent")
    refusal = result.get("refusal")
    citations = result.get("citations") or []
    checks: dict[str, bool] = {}
    notes: list[str] = []

    if "intent" in expect:
        checks["intent"] = intent == expect["intent"]
        if not checks["intent"]:
            notes.append(f"intent={intent} (attendu {expect['intent']})")

    if expect.get("tools_any"):
        ok = any(t in tools for t in expect["tools_any"])
        checks["tools_any"] = ok
        if not ok:
            notes.append(f"tools={tools} (attendu l'un de {expect['tools_any']})")

    if expect.get("tools_empty"):
        checks["tools_empty"] = len(tools) == 0
        if not checks["tools_empty"]:
            notes.append(f"tools non vides: {tools}")

    if expect.get("expect_refusal"):
        checks["refusal"] = refusal is not None or bool(
            re.search(r"\brefuse|ne (recommande|peux|n'analyse)\b", answer, flags=re.I)
        )
        if expect.get("refusal_reason_any") and refusal:
            checks["refusal_reason"] = refusal in expect["refusal_reason_any"]
        if not checks.get("refusal", True):
            notes.append("refus attendu non détecté")

    if expect.get("expect_source"):
        has_url = OFFICIAL_SOURCE_URL in answer or any(
            (c.get("url") == OFFICIAL_SOURCE_URL) for c in citations
        )
        checks["source"] = has_url
        if not has_url:
            notes.append("source officielle absente")

    for phrase in expect.get("must_contain") or []:
        key = f"contains:{phrase}"
        checks[key] = phrase.lower() in _norm(answer)
        if not checks[key]:
            notes.append(f"manque « {phrase} »")

    if expect.get("must_contain_any"):
        ok = any(p.lower() in _norm(answer) for p in expect["must_contain_any"])
        checks["must_contain_any"] = ok
        if not ok:
            notes.append(f"aucune de {expect['must_contain_any']}")

    if expect.get("must_contain_any_phrases"):
        ok = any(p.lower() in _norm(answer) for p in expect["must_contain_any_phrases"])
        checks["must_contain_any_phrases"] = ok

    for phrase in expect.get("must_not_contain") or []:
        key = f"forbids:{phrase}"
        checks[key] = phrase.lower() not in _norm(answer)
        if not checks[key]:
            notes.append(f"contient interdit « {phrase} »")

    if expect.get("must_not_contain_any"):
        bad = [p for p in expect["must_not_contain_any"] if p.lower() in _norm(answer)]
        checks["must_not_contain_any"] = len(bad) == 0
        if bad:
            notes.append(f"contient interdit: {bad}")

    # disclaimer
    checks["disclaimer"] = "aide à l'orientation" in _norm(answer) or "admission" in _norm(answer)

    passed = all(checks.values()) if checks else False
    return {
        "id": case["id"],
        "category": case["category"],
        "passed": passed,
        "checks": checks,
        "notes": notes,
        "intent": intent,
        "tools_called": tools,
        "refusal": refusal,
        "elapsed_ms": result.get("elapsed_ms"),
        "answer_preview": answer[:280],
    }


def run_evaluation(cases_path: Path | None = None) -> dict[str, Any]:
    ensure_dirs()
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    load_index()  # garantit l'index

    path = cases_path or TEST_CASES_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload["cases"]

    # couverture catégories
    counts = Counter(c["category"] for c in cases)
    minima = payload.get("meta", {}).get("categories_sujet", {})

    results = []
    t0 = time.perf_counter()
    for case in cases:
        out = run_agent(case["question"], profile={}, session_id="eval")
        scored = check_case(case, out)
        scored["question"] = case["question"]
        results.append(scored)

    elapsed = time.perf_counter() - t0
    n = len(results)
    n_pass = sum(1 for r in results if r["passed"])
    by_cat: dict[str, dict] = defaultdict(lambda: {"n": 0, "passed": 0})
    for r in results:
        by_cat[r["category"]]["n"] += 1
        by_cat[r["category"]]["passed"] += int(r["passed"])

    latencies = [r["elapsed_ms"] for r in results if r.get("elapsed_ms") is not None]
    report = {
        "meta": {
            "date": datetime.now(timezone.utc).isoformat(),
            "n_cases": n,
            "n_passed": n_pass,
            "accuracy": round(n_pass / n, 4) if n else 0.0,
            "elapsed_sec": round(elapsed, 2),
            "latency_ms_mean": round(sum(latencies) / len(latencies), 1) if latencies else None,
            "latency_ms_p95": sorted(latencies)[int(0.95 * (len(latencies) - 1))] if latencies else None,
            "source_officielle": OFFICIAL_SOURCE_URL,
        },
        "coverage": {
            "counts": dict(counts),
            "minima_sujet": minima,
            "coverage_ok": all(counts.get(k, 0) >= v for k, v in minima.items()),
        },
        "by_category": {
            k: {
                "n": v["n"],
                "passed": v["passed"],
                "accuracy": round(v["passed"] / v["n"], 4) if v["n"] else 0.0,
            }
            for k, v in sorted(by_cat.items())
        },
        "dimensions": {
            "systeme_complet": {
                "pass_rate": round(n_pass / n, 4) if n else 0.0,
                "latence_moyenne_ms": round(sum(latencies) / len(latencies), 1) if latencies else None,
            },
            "securite_et_biais": {
                "categories": ["securite", "biais", "provenance_psycho"],
                "pass_rate": _cat_rate(results, ["securite", "biais", "provenance_psycho"]),
            },
            "rag_et_generation": {
                "categories": ["factuel", "comparaison", "multi_etapes", "absent"],
                "pass_rate": _cat_rate(
                    results, ["factuel", "comparaison", "multi_etapes", "absent"]
                ),
            },
            "machine_learning": {
                "categories": ["recommandation_ml"],
                "pass_rate": _cat_rate(results, ["recommandation_ml"]),
            },
            "robustesse_ambiguite": {
                "categories": ["ambigu"],
                "pass_rate": _cat_rate(results, ["ambigu"]),
            },
        },
        "failed_cases": [r for r in results if not r["passed"]],
        "results": results,
    }

    RESULTS_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    RESULTS_MD.write_text(_to_markdown(report), encoding="utf-8")
    return report


def _cat_rate(results: list[dict], cats: list[str]) -> float | None:
    subset = [r for r in results if r["category"] in cats]
    if not subset:
        return None
    return round(sum(1 for r in subset if r["passed"]) / len(subset), 4)


def _to_markdown(report: dict) -> str:
    m = report["meta"]
    lines = [
        "# Rapport d'évaluation Orient'IA",
        "",
        f"- Date : `{m['date']}`",
        f"- Cas : **{m['n_passed']}/{m['n_cases']}** réussis (accuracy={m['accuracy']})",
        f"- Latence moyenne : **{m['latency_ms_mean']} ms** (p95={m['latency_ms_p95']} ms)",
        f"- Durée totale : {m['elapsed_sec']} s",
        f"- Source citée : {m['source_officielle']}",
        "",
        "## Couverture des catégories (sujet)",
        "",
        "| Catégorie | N | Min. sujet | Passés | Accuracy |",
        "|-----------|---|------------|--------|----------|",
    ]
    minima = report["coverage"]["minima_sujet"]
    counts = report["coverage"]["counts"]
    by_cat = report["by_category"]
    for cat, minimum in minima.items():
        n = counts.get(cat, 0)
        stats = by_cat.get(cat, {"passed": 0, "accuracy": 0})
        lines.append(
            f"| {cat} | {n} | {minimum} | {stats.get('passed', 0)} | {stats.get('accuracy', 0)} |"
        )
    lines += [
        "",
        f"Couverture minima respectée : **{report['coverage']['coverage_ok']}**",
        "",
        "## Dimensions mesurées",
        "",
    ]
    for name, dim in report["dimensions"].items():
        lines.append(f"- **{name}** : pass_rate={dim.get('pass_rate')}")
    lines += ["", "## Cas échoués", ""]
    failed = report.get("failed_cases") or []
    if not failed:
        lines.append("_Aucun._")
    else:
        for r in failed:
            lines.append(
                f"- `{r['id']}` ({r['category']}) — {r.get('notes')} — tools={r.get('tools_called')}"
            )
    lines += [
        "",
        "## Limites de l'évaluation",
        "",
        "- Les critères sont automatiques (mots-clés, outils, refus) : une réponse correcte formulée autrement peut échouer.",
        "- Le ML est entraîné surtout sur données synthétiques ; le transfert vers profils réels n'est mesuré que si l'enquête est renseignée.",
        "- Le RAG est lexical (TF-IDF), sans LLM génératif externe.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Évaluation ≥32 cas Orient'IA")
    parser.add_argument("--cases", type=str, default=str(TEST_CASES_PATH))
    parser.add_argument("--rebuild-index", action="store_true")
    args = parser.parse_args()

    if args.rebuild_index:
        print(json.dumps(build_index(), ensure_ascii=False))

    report = run_evaluation(Path(args.cases))
    print(
        json.dumps(
            {
                "accuracy": report["meta"]["accuracy"],
                "n_passed": report["meta"]["n_passed"],
                "n_cases": report["meta"]["n_cases"],
                "coverage_ok": report["coverage"]["coverage_ok"],
                "by_category": report["by_category"],
                "failed": [f["id"] for f in report["failed_cases"]],
                "report_md": str(RESULTS_MD),
                "report_json": str(RESULTS_JSON),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
