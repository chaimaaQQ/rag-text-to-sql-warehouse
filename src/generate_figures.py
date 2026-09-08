"""Generate the required, publication-ready result visualisations.

The script reads saved JSON only; it never recomputes nor modifies an
experiment. It is therefore safe to rerun after an analysis update.
"""

import argparse
import json
from pathlib import Path


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def save_execution_figure(matrix: list[dict], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    names = [row["pipeline"].split(" - ", 1)[0] for row in matrix]
    values = [100 * (row.get("execution_accuracy") or 0) for row in matrix]
    fig, axis = plt.subplots(figsize=(7, 4.2))
    bars = axis.bar(names, values, color=("#8E9AAF", "#4C78A8", "#F58518", "#54A24B"))
    axis.set_ylabel("Execution Accuracy (%)")
    axis.set_ylim(0, max(5, max(values) * 1.25))
    axis.set_title("Etude principale BIRD Mini-Dev")
    axis.bar_label(bars, labels=[f"{value:.1f}%" for value in values], padding=3)
    fig.tight_layout()
    fig.savefig(output_dir / "execution_accuracy_by_pipeline.png", dpi=220)
    plt.close(fig)


def save_schema_ablation_figure(ablation: dict, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    methods = list(ablation["results"]["table"])
    table_values = [100 * ablation["results"]["table"][method]["recall@5"] for method in methods]
    column_values = [100 * ablation["results"]["column"][method]["recall@5"] for method in methods]
    positions = range(len(methods))
    fig, axis = plt.subplots(figsize=(7, 4.2))
    axis.bar([position - 0.18 for position in positions], table_values, width=0.36, label="Tables")
    axis.bar([position + 0.18 for position in positions], column_values, width=0.36, label="Colonnes")
    axis.set_xticks(list(positions), [method.upper() for method in methods])
    axis.set_ylabel("Recall@5 (%)")
    axis.set_ylim(0, 100)
    axis.set_title("Ablation du retrieval de schema")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "schema_retrieval_ablation.png", dpi=220)
    plt.close(fig)


def save_h1_figure(h1: dict, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    structural = h1["structural_test_paired"]
    semantic = h1["semantic_test_paired"]
    labels = ("Structurelles", "Semantiques")
    b_values = (100 * structural["rate_B"], 100 * semantic["rate_B"])
    d_values = (100 * structural["rate_D"], 100 * semantic["rate_D"])
    positions = range(len(labels))
    fig, axis = plt.subplots(figsize=(7, 4.2))
    axis.bar([position - 0.18 for position in positions], b_values, width=0.36, label="B - schema seul")
    axis.bar([position + 0.18 for position in positions], d_values, width=0.36, label="D - hybride")
    axis.set_xticks(list(positions), labels)
    axis.set_ylabel("Taux d'erreur (%)")
    axis.set_ylim(0, 100)
    axis.set_title("Test H1 : echantillon apparié")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "h1_paired_error_rates.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Produit les figures de la matrice et de H1.")
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--schema_ablation", required=True)
    parser.add_argument("--h1_paired", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_execution_figure(load_json(args.matrix), output_dir)
    save_schema_ablation_figure(load_json(args.schema_ablation), output_dir)
    save_h1_figure(load_json(args.h1_paired), output_dir)
    print(f"Figures ecrites dans {output_dir}")


if __name__ == "__main__":
    main()
