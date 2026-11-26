# Generate compact JSON for report generation

`generate_minimal_json_for_llm.py` converts raw classifier and retrieval evidence into the compact v5 JSON used by the report generation pipeline. It processes one input file at a time and prints the output size and percentage reduction.

## Command line

Run from the repository root after generating raw evidence. [One-image workflow](../README.md#one-image-workflow)

```bash
python scripts/generate_minimal_json_for_llm.py \
  --input outputs/my_case/evidence.json \
  --output outputs/my_case/evidence_compact.json
```

| Option | Default | Description |
| --- | --- | --- |
| `--input`, `-i` | `data/classifier_retriever_evidence.json` | Raw evidence JSON |
| `--output`, `-o` | `data/classifier_retriever_evidence_minimal.json` | Destination JSON |
| `--top-k-common` | `1` | Set to `0` to omit the fallback. A positive value permits the single highest-ranked common reference when no selected label has an evidence entry. |
| `--top-k-label` | `6` | Maximum number of selected labels to retain, capped at six |
| `--thresholds-json` | Evidence threshold snapshot | Threshold artifact from the original selection run, required for legacy evidence without a snapshot |
| `--include-all-classifier-probabilities` | Off | Add full-precision scores, thresholds, and selection and retrieval flags for all six target pathologies |

## Python API

```python
from scripts.generate_minimal_json_for_llm import generate_minimal_json

evidence = generate_minimal_json(
    input_file="outputs/my_case/evidence.json",
    output_file="outputs/my_case/evidence_compact.json",
)
```

## Output fields

| Field | Meaning |
| --- | --- |
| `v` | Format version, currently `5` |
| `f` | Selected-label entries ordered by evidence status and classifier probability |
| `g` | Optional common reference when `f` is empty |
| `clf` | Optional classifier entries for all six target pathologies |

Each entry in `f` includes the label (`l`), rounded classifier probability (`p`) and selection threshold (`t`), evidence status (`s`), M@5 margin (`m`), evidence strength (`e`), and stability (`c`). A supporting reference (`a`) or matched non-supporting reference (`n`) appears when available. The entry also records an alternate classifier score source (`o`) when applicable.

References include a rounded similarity score (`q`) and a short evidence excerpt (`f`). Label-specific references also record label polarity (`z`). A supporting reference can include a label support score (`u`). The excerpt comes from the local report text or the reference's MeSH terms and problem labels, depending on availability.
