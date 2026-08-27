# Méthode de génération des données synthétiques (v2 — corpus ISPM)

## Objectif

Classification multi-classes vers **16 parcours ISPM** issus de `corpusISPMNote` / `parcours_labels.json`.

## Alignement corpus

Les centroïdes de génération reprennent les **matières principales** du corpus :

| Parcours | Axes corpus |
|----------|-------------|
| IGGLIA | prog, BD, génie logiciel, gestion, IA |
| ESIIA | électronique, systèmes, réseaux, IA |
| IMTICIA | multimédia, TIC, design |
| ISAIA | maths, stats, data, IA |
| EMII | physique, électromécanique, automatisme |
| ICMP | chimie, procédés industriels |
| GCA | physique, construction, architecture |
| CAA/EMP/FIC/DTJA | gestion, économie, droit, finance |
| IAA/AEE/PIP | biologie, chimie, agro/pharma |
| TEE/TEH | langues, tourisme, hôtellerie |

Features ajoutées : `score_physique`, `score_sciences_vie`, `score_langues`, `score_droit_eco`, intérêts métier hors informatique.

## Méthode

1. Tirage du label selon poids (Informatique légèrement sur-représentée).
2. Bruit gaussien sur scores + 12 % de préférences hors centroid.
3. ~5 % de bruit d'étiquette.
4. Contrôles de cohérence (bornes, labels catalogue).

Script : `python -m orientia.data.generate_synthetic --n 3000 --seed 42`

## Hypothèses / biais / contrôles

Voir aussi le registre `data/metadata/registre_sources.json` pour le corpus.

- Detail inégal : IGGLIA très documenté, ICMP peu documenté → centroïde plus grossier.
- Données synthétiques ≠ choix réels ; l'enquête reste le test de généralisation.
- Aucun attribut sensible (genre, âge, etc.).
