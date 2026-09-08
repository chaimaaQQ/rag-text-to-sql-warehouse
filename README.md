# RAG Text-to-SQL pour entrepots de donnees

Prototype de recherche appliquee pour le PFA Master 1. Le systeme compare
quatre pipelines Text-to-SQL sur BIRD Mini-Dev :

| Pipeline | Contexte donne au LLM |
| --- | --- |
| A | Aucun contexte externe |
| B | Schema recupere (BM25, embeddings ou hybride) |
| C | Connaissance metier BIRD (`evidence`) |
| D | Schema recupere + connaissance metier BIRD |

Le protocole est volontairement strict : sur BIRD, `evidence` est une
connaissance liee a la question et elle est injectee telle quelle dans C/D.
Le retrieval documentaire est donc evalue separement sur TPC-DS et
BIRD-Evidence-Corpus, jamais presente comme une metrique BIRD directe.

## Resultats reproduits

L'etude principale finalisee est stockee dans `results/main_study_final/` :
trois repetitions (seeds 0, 1 et 2) pour les quatre pipelines. Les moyennes
d'Execution Accuracy sont A = 0.00 %, B = 6.45 %, C = 1.34 % et D = 22.85 %.
La synthese, les tests H1, les figures et le brouillon de rapport sont dans
`results/main_study_final/analysis/`. Le manifeste fige la configuration
exacte de l'etude.

## Prerequis

- Python 3.10+ ;
- Ollama en cours d'execution, avec `qwen2.5:7b` (ou le modele fige dans
  `config.yaml`) ;
- BIRD Mini-Dev : bases SQLite placees sous
  `data/raw/bird/dev_databases/<db_id>/<db_id>.sqlite` ;
- dependances : `pip install -r requirements.txt`.

Les bases BIRD ne sont pas versionnees dans Git : elles sont volumineuses et
souvent soumises aux conditions de distribution du benchmark. Le chemin exact
et les commandes ci-dessous rendent neanmoins l'etude reproductible.

## Preparation des donnees

Depuis la racine du depot :

```powershell
# Necessaire seulement si vous voulez regenerer les 500 questions traitees
# depuis le fichier BIRD dev.json brut :
python src/data_loader.py --input <chemin-vers-dev.json> --output data/processed/questions.json
python src/normalize_questions.py --input data/processed/questions.json --output data/processed/questions_unique.json --report data/processed/questions_normalization_report.json
python src/schema_parser.py --databases_dir data/raw/bird/dev_databases --output_dir data/schemas
python src/retriever_schema.py build --schemas_dir data/schemas --output_dir data/index
python src/knowledge_base.py
python src/evaluator.py ground-truth --questions data/processed/questions_unique.json --schemas_dir data/schemas --output data/evaluation/ground_truth.json
```

Reconstruire l'index apres toute modification de `data/schemas` : les
documents de schema incluent les colonnes et les relations de cle etrangere.

## Ablations de retrieval

```powershell
python src/run_schema_ablation.py --index_dir data/index --ground_truth data/evaluation/ground_truth.json --questions data/processed/questions.json --output results/retrieval_ablation/schema_ablation.json --k 5
python src/run_business_retrieval_ablation.py --questions data/evaluation/tpcds_eval_questions_validated.json --knowledge_base data/knowledge_base/knowledge_base.json --output results/retrieval_ablation/business_ablation.json --k 5
python src/run_bird_evidence_retrieval_ablation.py --questions data/processed/questions.json --knowledge_base data/knowledge_base/knowledge_base.json --output results/retrieval_ablation/bird_evidence_ablation.json --k 5 --sample_size 150
```

La methode retenue pour le schema est celle qui maximise Table Recall@5. La
methode documentaire est choisie sur son corpus secondaire, puis documentee
separement dans le rapport.

## Etude principale obligatoire

Cette commande fixe le modele, la temperature, `top_k`, la methode de
retrieval et l'auto-correction desactivee. Elle execute A/B/C/D avec trois
seeds, valide chaque SQL, execute SQL et gold SQL en lecture seule, et produit
les moyennes/ecarts-types.

```powershell
python src/run_main_study.py `
  --questions data/processed/questions.json `
  --schemas_dir data/schemas `
  --index_dir data/index `
  --databases_dir data/raw/bird/dev_databases `
  --output_dir results/main_study_reproduction `
  --model qwen2.5:7b --temperature 0 --top_k 5 `
  --schema_method hybrid --runs 3
```

`results/main_study_reproduction/main_study_manifest.json` conserve les parametres
figes. Chaque sortie `pipeline_X/run_YY/` contient generation, validation,
execution, categories d'erreurs et provenance. Le lanceur refuse d'ecraser un
resultat existant sans `--overwrite`.

### Finalisation de runs repris individuellement

L'etude versionnee dans `results/main_study_final/` a ete executee pipeline par
pipeline avec `--resume`, afin de sauvegarder chaque question. Une fois les
douze fichiers `generated_sql.json` et `execution_accuracy.json` presents, la
commande suivante regenere les validations structurelles, les categories,
la synthese et les tests H1 :

```powershell
python src/finalize_main_study.py `
  --results_dir results/main_study_final `
  --schemas_dir data/schemas `
  --output_dir results/main_study_final/analysis
```

## H1 et annotation humaine

Le test principal compare B et D uniquement sur les questions tentees par les
deux pipelines. Utiliser McNemar exact, pas Fisher, car les observations sont
appariees. Pour les resultats au format historique :

```powershell
python src/hypothesis_h1_test_paired.py `
  --error_classification_dir results/error_classification `
  --output results/hypothesis_H1/h1_test_result_paired.json
```

Les categories fondees sur l'execution ne remplacent pas la validation
humaine des erreurs semantiques et jointures executables. Creer puis faire
remplir la feuille par deux annotateurs independants :

```powershell
python src/error_annotation.py sample --questions data/processed/questions.json `
  --generated_b results/pipelines_A_B_C_D/pipeline_B/generated_sql.json `
  --execution_b results/pipelines_A_B_C_D/pipeline_B/execution_accuracy.json `
  --generated_d results/pipelines_A_B_C_D/pipeline_D/generated_sql.json `
  --execution_d results/pipelines_A_B_C_D/pipeline_D/execution_accuracy.json `
  --output results/error_classification/manual_annotation.csv --n 60

python src/error_annotation.py score --sheet results/error_classification/manual_annotation.csv `
  --output results/error_classification/manual_annotation_agreement.json
```

Rapporter le taux d'accord, Cohen kappa et tous les desaccords. Aucun label ne
doit etre rempli automatiquement.

## Matrice et figures

```powershell
python src/build_comparative_matrix.py `
  --results_dir results/pipelines_A_B_C_D `
  --evaluation_dir data/evaluation `
  --error_classification_dir results/error_classification `
  --schema_ablation_file results/retrieval_ablation/schema_ablation.json `
  --business_ablation_file results/retrieval_ablation/business_ablation.json `
  --bird_ablation_file results/retrieval_ablation/bird_evidence_ablation.json `
  --output_dir results/figures

python src/generate_figures.py --matrix results/figures/comparative_matrix.json `
  --schema_ablation results/retrieval_ablation/schema_ablation.json `
  --h1_paired results/hypothesis_H1/h1_test_result_paired.json `
  --output_dir results/figures
```

Les fichiers de `report/` structurent le rapport, la synthese article et la
soutenance. Ne pas annoncer H1 comme confirmee avant les trois repetitions,
l'annotation A/B et le test apparié final.

## Limites connues et perimetre

- Le projet est un prototype RAG par prompting, non un modele Text-to-SQL
  fine-tune ;
- les mesures de cout/latence sont produites par l'etude principale ;
- l'auto-correction est une ablation secondaire distincte et ne doit jamais
  etre melangee aux chiffres de H1 ;
- les resultats historiques du depot doivent etre regeneres avec les scripts
  actuels avant d'etre cites dans le rapport final.
