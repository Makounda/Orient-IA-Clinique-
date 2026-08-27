# Rapport d'évaluation Orient'IA

- Cas : **35/35** réussis (accuracy=1.0)
- Latence moyenne : **108.5 ms** (p95=363 ms)
- Durée totale : 3.84 s
- Source citée : https://ispm-edu.com/publications.php

## Couverture des catégories (sujet)

| Catégorie | N | Min. sujet | Passés | Accuracy |
|-----------|---|------------|--------|----------|
| factuel | 6 | 5 | 6 | 1.0 |
| comparaison | 4 | 4 | 4 | 1.0 |
| recommandation_ml | 7 | 6 | 7 | 1.0 |
| multi_etapes | 5 | 4 | 5 | 1.0 |
| absent | 3 | 3 | 3 | 1.0 |
| ambigu | 3 | 3 | 3 | 1.0 |
| securite | 3 | 3 | 3 | 1.0 |
| biais | 2 | 2 | 2 | 1.0 |
| provenance_psycho | 2 | 2 | 2 | 1.0 |

Couverture minima respectée : **True**

## Dimensions mesurées

- **systeme_complet** : pass_rate=1.0
- **securite_et_biais** : pass_rate=1.0
- **rag_et_generation** : pass_rate=1.0
- **machine_learning** : pass_rate=1.0
- **robustesse_ambiguite** : pass_rate=1.0

## Cas échoués

_Aucun._

## Limites de l'évaluation

- Les critères sont automatiques (mots-clés, outils, refus) : une réponse correcte formulée autrement peut échouer.
- Le ML est entraîné surtout sur données synthétiques ; le transfert vers profils réels n'est mesuré que si l'enquête est renseignée.
- Le RAG est lexical (TF-IDF), sans LLM génératif externe.
