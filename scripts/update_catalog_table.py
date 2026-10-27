"""Update generated licensing columns in the Markdown model catalog."""

from __future__ import annotations

from pathlib import Path

from imagehub.catalog import MODEL_CATALOG


def main() -> None:
    path = Path(__file__).resolve().parents[1] / "docs" / "catalog.md"
    output: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("| Model id |"):
            output.append(
                "| Model id | Family | Source | Revision | Origin | Validation | Gated | "
                "Weight license | Code license | Code commercial use | Awesome catalog |"
            )
            continue
        if line.startswith("| --- |"):
            output.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
            continue
        if not line.startswith("| `"):
            output.append(line)
            continue
        columns = [column.strip() for column in line.strip().strip("|").split("|")]
        model_id = columns[0].strip("`")
        spec = MODEL_CATALOG[model_id]
        original = columns[:6]
        awesome_catalog = columns[-1]
        if model_id == "qwen-image":
            awesome_catalog = f"`{spec.catalog_id}`"
        output.append(
            "| "
            + " | ".join(
                [
                    *original,
                    "yes" if spec.gated else "no",
                    spec.license,
                    spec.code_license,
                    "yes" if spec.code_commercial_use else "no",
                    awesome_catalog,
                ]
            )
            + " |"
        )
    path.write_text("\n".join(output) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
