# Experiments

## Overview

Experiments allow different modelling approaches, datasets and parameters to be compared using a consistent evaluation framework.

Experiments are configured entirely through YAML.

---

## Experiment Configuration

Example:

```yaml
ws_dataset: "workstream_objectives_v8.csv"
ev_dataset: "evidence_library_records_v1.csv"
evaluation_dataset: "manual_mapping_corrected.csv"

bi_encoder: "all-mpnet-base-v2"
llm_id: "eu.anthropic.claude-haiku-4-5-20251001-v1:0"

pipeline:
  - name: hardcode

  - name: bi_encoder
    top_k: 5

  - name: llm
    top_k: 5
```

Common settings:

| Setting | Purpose |
|----------|----------|
| ev_dataset | Evidence records |
| ws_dataset | Workstream objectives |
| evaluation_dataset | Ground truth mappings |
| pipeline | Pipeline stages |
| include_non_odp | Include non-ODP workstreams |
| save_experiment | Save outputs and metadata |

---

## Running an Experiment
Running the code below:
```bash
python -m src.main
```

runs an experiment using the default `experiment_config.yaml`, `new_mapping.yaml`, and `workstreams.yaml` configuration files.

---



## Evaluation

The framework evaluates ranked predictions against manually labelled mappings.

Metrics include:

- Top-1 Precision
- Top-1 Recall
- Top-1 F1
- Recall@K
- Confusion Matrix

---

## Interpreting Results

High Recall@K but low Top-1 performance suggests:

- retrieval is successful
- ranking requires improvement

Low Recall@K suggests:

- candidate generation is failing
- correct workstreams are not reaching later stages

---

## Saved Outputs

When:

```yaml
save_experiment: true
```

the framework saves the following outputs to the One Snapshot shared folder on Data Workspace:

- Experiment summary
- Input configuration files
- Predictions
- Evaluation dataframe
- Confusion matrix

These outputs allow experiments to be reproduced and compared consistently.

### Experiment Naming

Experiment names are generated automatically from the experiment configuration to ensure that the key properties of an experiment can be understood directly from its name.

Experiment names follow the convention:

```text
ws-{workstream_version}_ev-{evidence_version}_{pipeline}_scope-{scope}
```

where:

| Component | Description |
|------------|------------|
| `workstream_version` | Version of the workstream objectives dataset. |
| `evidence_version` | Version of the evidence library dataset. |
| `pipeline` | Summary of the configured pipeline stages, models and top-k values. |
| `scope` | Whether predictions are generated over ODP workstreams only or all workstreams. |

Pipeline stages are represented using shortened identifiers:

| Stage | Abbreviation |
|---------|---------|
| Hardcode | `hc` |
| Bi-encoder | `be` |
| Reranker | `rr` |
| NLI | `nli` |
| LLM | `llm` |

Model names are converted to short forms using the mappings defined in `get_models.py`.

### Example

```text
ws-v8_ev-v1_hc_be-mpnetB-k5_llm-haiku-k5_scope-all
```

This indicates:

- Workstream objectives dataset version 8
- Evidence library dataset version 1
- Hardcoded workstream filtering
- MPNet bi-encoder retrieval with top 5 candidates
- Claude Haiku ranking over the top 5 candidates
- Predictions generated across all workstreams (including non-ODP workstreams)

