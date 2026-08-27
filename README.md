# Orient'IA — Assistant d'orientation pédagogique ISPM

## Prérequis

- Python 3.9+

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate    
pip install -U pip setuptools
pip install -e ".[dev]"
```

## Exécution rapide (reproductible)

```bash
# 1. Corpus + débouchés + index RAG
python -m orientia.data.process_corpus
python -m orientia.data.process_debouches
python -m orientia.rag.index --build

# 2. Données synthétiques + entraînement ML (baseline + 2 approches)
python -m orientia.data.generate_synthetic --n 3000 --seed 42
python -m orientia.ml.train

# 3. Agent (CLI)
python -m orientia.agent.orchestrator --message "Compare ISAIA et IGGLIA"

# 4. Interface
streamlit run src/orientia/app/streamlit_app.py

# 5. Évaluation ≥ 32 cas
python -m orientia.evaluation.run_eval
```

## Outils de l'agent

| Outil | Rôle |
|-------|------|
| `rechercher_formation` | RAG documentaire |
| `comparer_parcours` | Comparaison structurée |
| `analyser_profil_ml` | Recommandation ML |
| `identifier_debouches` | Débouchés par parcours |

## Livrables (sujet)

| Élément | Emplacement |
|---------|-------------|
| Code source | `src/orientia/` |
| README | `README.md` (ce fichier) |
| Corpus / collecte | `data/raw/corpus/`, `data/corpus/` |
| Registre des sources | `data/metadata/registre_sources.json` |
| Jeu ML | `data/processed/profiles_synthetic.csv` |
| Enquête | `data/raw/survey/` |
| Notebooks | `notebooks/01_eda_and_training.ipynb` |
| Modèle | `artifacts/models/best_model.joblib` |
| Jeu d'évaluation | `data/evaluation/test_cases.json` |
| Résultats d'évaluation | `artifacts/evaluation/eval_report.md` |
| Schéma d'architecture | [`docs/architecture.md`](docs/architecture.md) |
| Limites / biais / risques | `data/metadata/limites_biais_risques.md` |
| Traces | `artifacts/traces/agent_traces.jsonl` |

## Machine Learning

- **Problème** : classification multi-classes (16 parcours ISPM)
- **Baseline** : régression logistique
- **Approches** : Random Forest, HistGradientBoosting
- **Sélection** : meilleur macro-F1 → `best_model.joblib`
- **Métriques** : `artifacts/metrics/comparison_report.json`

## Évaluation expérimentale

Le jeu contient **≥ 32 cas** répartis selon le sujet :

- questions factuelles, comparaisons, recommandations ML
- multi-étapes, infos absentes, profils ambigus
- sécurité / prompt injection, biais, provenance / refus profilage

```bash
python -m orientia.evaluation.run_eval
```

Rapports : `artifacts/evaluation/eval_results.json` et `eval_report.md`.

## Structure du dépôt

```
Orient'IA/
  data/
    raw/{corpus,survey}/
    processed/
    corpus/
    evaluation/
    metadata/
  src/orientia/
    data/ ml/ rag/ agent/ app/ evaluation/
  artifacts/{models,metrics,rag,traces,evaluation}/
  notebooks/
```
