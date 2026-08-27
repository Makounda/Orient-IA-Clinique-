# Schéma d’architecture — Orient’IA

Assistant intelligent d’orientation pédagogique (ISPM).  
Prototype — Data Science · ML · RAG · Agents.

---

## 1. Vue d’ensemble

Deux chaînes d’acquisition convergent vers un **agent unique**, dont chaque décision reste traçable.

```mermaid
flowchart TB
  subgraph acquisition [Acquisition]
    Site[Sources ISPM / publications]
    Notes[corpusISPMNote + corpusDebauche]
    Enquete[Enquête anonymisée]
    Synth[Génération profils synthétiques]
  end

  subgraph preparation [Préparation]
    Corpus[Corpus structuré + chunks]
    Registre[Registre des sources]
    DatasetML[Jeu de données ML]
  end

  subgraph modeles [Modèles et index]
    RAG[Index RAG TF-IDF]
    ML[Modèle ML entraîné]
  end

  subgraph runtime [Runtime]
    Agent[Agent conversationnel]
    Outils[Outils techniques]
    UI[Interface Streamlit / CLI]
  end

  subgraph sortie [Sortie]
    Reco[Recommandation argumentée]
    Source[Source citée]
    Traces[Traces / observabilité]
  end

  Site --> Notes
  Notes --> Corpus
  Notes --> Registre
  Enquete --> DatasetML
  Synth --> DatasetML
  Corpus --> RAG
  DatasetML --> ML
  RAG --> Outils
  ML --> Outils
  Outils --> Agent
  UI --> Agent
  Agent --> Reco
  Agent --> Source
  Agent --> Traces
```

---

## 2. Chaînes de données

### 2.1 Corpus pédagogique (documents)

```mermaid
flowchart LR
  A[corpusISPMNote] --> P1[process_corpus]
  B[corpusDebauche] --> P2[process_debouches]
  P1 --> C[corpus_structured.json]
  P1 --> D[corpus_chunks.jsonl]
  P2 --> E[debouches_structured.json]
  P2 --> D
  D --> I[build_index TF-IDF]
  I --> J[artifacts/rag/tfidf_index.joblib]
  P1 --> R[registre_sources.json]
  P2 --> R
```

| Étape | Module | Sortie |
|-------|--------|--------|
| Parse notes formations | `orientia.data.process_corpus` | chunks parcours / matières |
| Parse débouchés | `orientia.data.process_debouches` | chunks `debouches_*` |
| Indexation | `orientia.rag.index` | index TF-IDF persistant |

Source publique citée côté utilisateur : `https://ispm-edu.com/publications.php`

### 2.2 Données Machine Learning (profils)

```mermaid
flowchart LR
  Rules[Centroïdes / règles corpus] --> Gen[generate_synthetic]
  Gen --> TrainCSV[profiles_synthetic.csv]
  Survey[responses_anonymized.csv] --> Loader[survey_loader]
  Loader --> SurveyCSV[profiles_survey.csv]
  TrainCSV --> Train[ml.train]
  SurveyCSV -.->|test si N≥10| Train
  Train --> Model[best_model.joblib]
  Train --> Metrics[comparison_report.json]
```

| Montage recommandé (sujet) | Usage |
|----------------------------|--------|
| Synthétique | entraînement |
| Enquête réelle | validation / test (si disponible) |
| Sinon | hold-out 20 % synthétique |

---

## 3. Machine Learning

```mermaid
flowchart TB
  X[Profil : scores, compétences, intérêts, préférences] --> Prep[ColumnTransformer]
  Prep --> M1[Baseline : LogisticRegression]
  Prep --> M2[Random Forest]
  Prep --> M3[HistGradientBoosting]
  M1 --> Sel[Sélection macro-F1]
  M2 --> Sel
  M3 --> Sel
  Sel --> Best[best_model.joblib]
  Best --> ToolML[outil analyser_profil_ml]
```

- **Problème** : classification multi-classes → 16 parcours ISPM  
- **Features** : affinités 0–10, binaires compétences/intérêts, préférences catégorielles  
- **Pas de features sensibles** (sexe, âge, origine, etc.)

---

## 4. Agent conversationnel et outils

```mermaid
flowchart TB
  User[Message utilisateur] --> Sec[Contrôles sécurité]
  Sec -->|injection / biais / psycho| Refuse[Refus + disclaimer]
  Sec -->|OK| Intent[Détection d'intention]
  Intent --> Router{Routage}

  Router -->|factuel| T1[rechercher_formation]
  Router -->|comparaison| T2[comparer_parcours]
  Router -->|recommandation| T3[analyser_profil_ml]
  Router -->|débouchés| T4[identifier_debouches]
  Router -->|multi-étapes| T2
  Router -->|multi-étapes| T4
  Router -->|absent / provenance| Meta[Réponse système]

  T1 --> RAG[(Index RAG)]
  T2 --> Cat[(Catalogue parcours)]
  T2 --> RAG
  T3 --> ML[(Modèle ML)]
  T4 --> Deb[(debouches_structured + RAG)]

  T1 --> Synth[Synthèse argumentée]
  T2 --> Synth
  T3 --> Synth
  T4 --> Synth
  Meta --> Synth
  Synth --> Out[Réponse + Source URL + Disclaimer]
  Synth --> Log[Trace JSONL]
```

### Outils (≥ 3, ici 4)

| Outil | Type d’opération | Entrée | Sortie |
|-------|------------------|--------|--------|
| `rechercher_formation` | RAG lexical | requête texte | passages + score |
| `comparer_parcours` | catalogue + RAG | 2 codes parcours | axes / matières / citations |
| `analyser_profil_ml` | inférence ML | profil structuré | top-k parcours + scores |
| `identifier_debouches` | structuré + RAG | code parcours | métiers par catégorie |

Distinction affichée dans la réponse :
- résultats **ML**
- informations **documents**
- messages **système** (refus, absence, provenance)

---

## 5. Couches logicielles (dépôt)

```text
Orient'IA/
├── data/
│   ├── raw/corpus/          # notes brutes
│   ├── raw/survey/          # enquête
│   ├── processed/           # CSV profils
│   ├── corpus/              # JSON / JSONL structurés
│   ├── evaluation/          # ≥ 32 cas de test
│   └── metadata/            # registre, limites, labels
├── src/orientia/
│   ├── data/                # acquisition & préparation
│   ├── ml/                  # entraînement / predict
│   ├── rag/                 # index & search
│   ├── agent/               # tools + orchestrateur
│   ├── app/                 # Streamlit
│   └── evaluation/          # protocole d’évaluation
└── artifacts/
    ├── models/              # best_model.joblib
    ├── metrics/             # rapports ML
    ├── rag/                 # index TF-IDF
    ├── traces/              # observabilité agent
    └── evaluation/          # résultats des 32+ cas
```

---

## 6. Flux d’une requête (séquence)

```mermaid
sequenceDiagram
  participant U as Utilisateur
  participant UI as Streamlit / CLI
  participant A as Agent
  participant T as Outils
  participant R as RAG / ML
  participant L as Traces

  U->>UI: question
  UI->>A: run_agent(message, profil)
  A->>A: sécurité + intention
  alt refus
    A-->>UI: message de refus + disclaimer
  else ok
    A->>T: appel outil(s)
    T->>R: retrieval / predict
    R-->>T: résultats
    T-->>A: sorties structurées
    A->>A: synthèse + source URL
    A->>L: append JSONL
    A-->>UI: réponse argumentée
  end
  UI-->>U: affichage (+ traces UI)
```

---

## 7. Observabilité et sécurité

### Traces (`artifacts/traces/agent_traces.jsonl`)

Pour chaque tour : question, intention, profil extrait, outils appelés, sorties, réponse, latence, refus éventuel.

### Garde-fous

- injections de prompt / invention de filière → refus  
- recommandation par sexe / âge → refus  
- profilage psychologique → refus  
- info absente du corpus → non-invention + renvoi administration  
- disclaimer obligatoire dans l’interface et les réponses  

---

## 8. Évaluation

```mermaid
flowchart LR
  Cases[test_cases.json ≥32] --> Runner[evaluation.run_eval]
  Runner --> Agent[Agent + outils]
  Agent --> Report[eval_report.md / eval_results.json]
```

Catégories mesurées (sujet) : factuel, comparaison, ML, multi-étapes, absent, ambigu, sécurité, biais, provenance / psycho.

---

## 9. Principes directeurs

1. **Traçabilité** des sources et des appels d’outils  
2. **Séparation** ML / documents / règles  
3. **Prudence** : incertitude déclarée, pas de décision administrative  
4. **Mesure** : preuves expérimentales (ML + 32 cas) plutôt qu’affirmations  
