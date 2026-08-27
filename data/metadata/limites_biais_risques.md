# Limites, biais et risques — Orient'IA

## Limites

- Corpus pédagogique inégalement détaillé (matières très riches pour IGGLIA ; autres parcours surtout matières principales + débouchés).
- Modèle ML entraîné majoritairement sur **données synthétiques** ; généralisation vers profils humains dépend de l'enquête.
- Agent sans LLM génératif externe : réponses déterministes par outils + synthèse par règles.
- Index RAG lexical (TF-IDF) : pas de compréhension sémantique profonde.
- Les recommandations ne remplacent ni conseiller pédagogique ni décision d'admission.

## Biais

- Sur-représentation possible de certains parcours dans les données synthétiques.
- Labels synthétiques reproduisent des hypothèses d'orientation (risque de circularité).
- Auto-sélection de l'enquête (quand renseignée).
- Aucun attribut sensible (sexe, âge, origine…) n'est utilisé comme feature.

## Risques pris en charge

- Injections de prompt / demandes d'invention de filières → refus.
- Recommandations discriminatoires (sexe, âge…) → refus.
- Profilage psychologique → refus.
- Confusion conseil / décision administrative → disclaimer systématique.
- Informations absentes (frais, dates de concours…) → non-invention + renvoi administration.

## Source publique citée

https://ispm-edu.com/publications.php
