# Synthèse automatique — étude principale BIRD Mini-Dev

## Résultats d'exécution (moyenne ± écart-type, 3 runs)

| Pipeline | Execution Accuracy |
| --- | --- |
| A - Sans RAG | 0.00 % ± 0.00 % |
| B - RAG schema seul | 6.45 % ± 0.00 % |
| C - RAG metier seul | 1.34 % ± 0.10 % |
| D - RAG hybride | 22.85 % ± 0.19 % |

## H1 — comparaison appariée B vs D

| Run | Erreurs sémantiques B | Erreurs sémantiques D | p sémantique | p structurelle | H1 selon le protocole par run |
| --- | --- | --- | --- | --- | --- |
| run_00 | 53.6 % | 26.1 % | 3.24e-08 | 0.0895 | True |
| run_01 | 53.7 % | 27.2 % | 1.01e-07 | 0.163 | True |
| run_02 | 53.2 % | 26.6 % | 5.73e-08 | 0.121 | True |

## Limite d'interprétation

Les trois runs montrent une réduction statistiquement significative des erreurs sémantiques de D. Cependant, le résumé pooled des paires question-run indique une hausse significative des erreurs structurelles de D ; il est descriptif car les mêmes questions sont répétées entre seeds. La conclusion finale doit donc présenter les trois tests par run et ne pas affirmer une amélioration structurelle.

## Annotation humaine

60 cas ont été annotés. L'accord brut est de 100.0 % et Cohen's kappa est de 1.000. Aucun désaccord n'a été enregistré.

Cette valeur n'est interprétable comme un accord inter-annotateurs que si les deux colonnes ont effectivement été remplies de manière indépendante.
