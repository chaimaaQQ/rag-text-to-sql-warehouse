# Plan de soutenance - 10 diapositives

1. Titre, auteurs, encadrant et objectif.
2. Probleme BI : schema technique et connaissances metier manquantes.
3. Etat de l'art : BIRD, RAG, prompting/fine-tuning Text-to-SQL.
4. Question de recherche et hypothese H1.
5. Protocole BIRD/evidence : ce qui est injecte et ce qui est reellement recupere.
6. Architecture : schema (tables, colonnes, relations), connaissances metier, LLM, validation/execution.
7. Plan factoriel : pipelines A/B/C/D, retrieval fige, trois repetitions, auto-correction separee.
8. Ablations : Table/Column Recall@5 et Knowledge Recall@5 sur les deux corpus documentaires.
9. Resultats : Execution Accuracy avec barres d'erreur, taux d'erreurs et test H1 apparié.
10. Limites, enseignements, travaux futurs et demonstration/reproductibilite.

Chaque chiffre doit pointer vers un JSON de `results/`; chaque graphique doit
indiquer le corpus, la taille d'echantillon et le nombre de repetitions.
