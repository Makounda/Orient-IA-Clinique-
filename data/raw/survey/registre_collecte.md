# Registre de collecte — Enquête Orient'IA

À remplir et livrer avec le jeu de données (exigence du sujet).

## Questionnaire

- Fichier version diffusée : `data/raw/survey/questionnaire.json`
- Version : 1.0
- Date de la version : 2026-08-27
- Texte de consentement : `data/raw/survey/consentement.txt`

## Populations visées et mode de diffusion

| Population | Mode de diffusion | Canal | Date début | Date fin |
|------------|-------------------|-------|------------|----------|
| Étudiants | *à compléter* | *lien / QR / présentiel* | | |
| Professionnels | *à compléter* | *réseaux / alumni* | | |

## Période et volumes

- Période de collecte : *du … au …* (gel recommandé fin J1 du hackathon)
- Nombre de réponses reçues : _
- Nombre retenues : _
- Nombre écartées : _
- Motifs d'exclusion : _

## Répartition

| Population | N retenues |
|------------|------------|
| student | |
| professional | |
| **Total** | |

## Anonymisation

- Procédure : attribution d'un `anonymized_id` ; pas de champs nominatifs ; pas de données sensibles.
- Fichier livré : `data/raw/survey/responses_anonymized.csv`
- Fichier normalisé (schéma ML) : `data/processed/profiles_survey.csv`

## Traitements postérieurs

| Traitement | Justification |
|------------|---------------|
| Nettoyage bornes scores / notes | Cohérence schéma |
| Exclusion sans consentement / sans label | Recevabilité |
| Recodages éventuels | *à documenter* |

## Biais d'échantillonnage constatés

- Volume limité → intervalles de confiance larges.
- Auto-sélection possible (parcours / profils sur-représentés).
- Étudiants : le parcours *choisi* ≠ parcours *optimal*.
- Professionnels : biais de reconstruction temporelle.

## Commentaire libre

_
