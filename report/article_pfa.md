# Retrieval conjoint de schema et de connaissance metier pour le Text-to-SQL analytique

## Resume

Ce travail evalue si l'ajout de connaissance metier a un retrieval de schema
ameliore la generation SQL analytique. Le systeme compare quatre pipelines :
sans RAG, schema seul, metier seul et hybride. Le protocole distingue le
retrieval de schema sur BIRD Mini-Dev du retrieval documentaire, evalue sur
TPC-DS et BIRD-Evidence-Corpus. Les resultats definitifs sont rapportes sur
trois repetitions du meme modele, avec validation SQL, execution SQLite et
test apparié de l'hypothese H1.

## Problematique et hypothese

**H1.** Le pipeline hybride diminue significativement les erreurs semantiques
par rapport au pipeline schema seul, sans changement significatif des erreurs
structurelles. La comparaison B/D est appariée : seules les questions tentees
par les deux pipelines sont retenues, puis chaque categorie est testee par
McNemar exact.

## Methode

Le schema est transforme en documents table/colonnes/relations et recupere par
BM25, embeddings MiniLM ou combinaison hybride. Sur BIRD, `evidence` est une
connaissance associee a chaque question et ne constitue donc pas une recherche
documentaire. Le corpus TPC-DS et BIRD-Evidence-Corpus servent exclusivement a
mesurer Knowledge Recall@k. La generation est faite avec un LLM fige, a
temperature fixe, trois seeds et auto-correction desactivee pendant l'etude
principale.

## Resultats (trois repetitions)

| Pipeline | Execution Accuracy, moyenne +/- ecart-type | Valid SQL Rate | Refus |
| --- | --- | --- | --- |
| A | 0,00 % +/- 0,00 % | 0,00 % | 79,80 % |
| B | 6,45 % +/- 0,00 % | 19,53 % | 71,07 % |
| C | 1,34 % +/- 0,10 % | 2,53 % | 42,87 % |
| D | 22,85 % +/- 0,19 % | 50,80 % | 22,87 % |

Le retrieval hybride est retenu pour le schema : Table Recall@5 = 93,75 % et
Column Recall@5 = 40,86 %. Sur TPC-DS (27 questions), BM25, embeddings et
hybride atteignent tous Recall@5 = 100 % ; cette evaluation documentaire est
separee des mesures BIRD de bout en bout.

## Test H1

Rapporter : taille appariée, table de contingence, taux B/D, p-value McNemar,
effet et intervalle de confiance. La conclusion doit etre l'une des deux :

- « H1 est soutenue sur l'echantillon apparié » ;
- « H1 n'est pas soutenue ; [condition qui echoue] ».

Ne pas conclure a partir d'une comparaison de populations differentes causee
par des taux de refus B/D distincts. Sur les trois runs apparies, D reduit les
erreurs semantiques de 53,2--53,7 % a 26,1--27,2 % (McNemar exact,
$p < 1,1\times10^{-7}$). Les differences structurelles ne sont pas
significatives run par run ; le rapport ne revendique donc pas une amelioration
structurelle.

## Limites

Les erreurs executees mais fausses ne sont pas automatiquement des erreurs
metier : une jointure executable mais incorrecte exige une annotation humaine.
Soixante sorties B/D sont annotees ; l'accord brut rapporte est de 100 % et
Cohen kappa = 1,000, sans desaccord. Ce resultat suppose que les deux
annotateurs ont travaille independamment. Le systeme est un prototype RAG par
prompting, non un modele fine-tune et non une revendication de performance
SOTA.

## References essentielles

- Lewis et al. (2020), *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*.
- Li et al. (2023), *BIRD: A BIg Bench for Large-Scale Database Grounded Text-to-SQLs*.
- Shi et al. (2025), *A Survey on Employing Large Language Models for Text-to-SQL Tasks*.
