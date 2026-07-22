# OneSnapshot NLP

NLP experimentation framework for mapping evidence library records to DBT workstreams.

The project supports multiple approaches to workstream classification, including:

- Hardcoded directorate and ALB mappings
- Sentence-transformer bi-encoders
- Cross-encoder rerankers
- Natural Language Inference (NLI) models
- AWS Bedrock LLMs

Experiments are configuration-driven and can be reproduced from saved experiment metadata.


## Public Repository Version

This repository contains synthetic example configuration files.

The original project used organisation-specific workstreams,
departments and mappings which have been replaced with fictional
examples while preserving the structure and behaviour of the system.

This allows the architecture and implementation to be shared
without exposing internal business information.


## Project Goal

Given an evidence library record and a set of DBT workstreams, the pipeline generates ranked workstream predictions and evaluates them against manually labelled mappings.

The framework is designed for:

- experimentation and model comparison
- reproducibility
- evaluation of retrieval and ranking approaches
- robust handling of LLM outputs

---

## Project Status

This project is actively maintained and continues to be used for NLP experimentation and evaluation.

While the core experimentation framework, evaluation pipeline and LLM validation workflows are well established, development is ongoing to further improve production-readiness. Current areas of focus include expanding automated test coverage, increasing robustness through additional validation and error handling, and introducing more comprehensive integration testing.

These improvements are intended to complement the existing experimentation framework while maintaining reproducibility and ease of extension.

---

## Repository Structure

```text
onesnapshot-nlp/
│
├── src/
│   ├── main.py
│   ├── methods.py
│   ├── methods_utils.py
│   ├── experiment.py
│   ├── evaluate_functions.py
│   ├── utility_functions.py
│   ├── recreate_experiment.py
│   └── locations.py
│
├── configs/
│
├── test/
│
├── docs/
│   ├── architecture.md
│   ├── experiments.md
│   └── testing.md
│
├── pyproject.toml
├── setup.sh
└── README.md
```

---

## Installation

From the project root:

```bash
. setup.sh
```

To recreate the environment after dependency changes:

```bash
. setup.sh recreate
```

---

## Running an Experiment

```bash
python -m src.main
```

runs an experiment using the default `experiment_config.yaml`, `new_mapping.yaml`, and `workstreams.yaml` configuration files.


Optionally specify alternative mapping and workstream files:

```bash
python -m src.main \
    -c configs/{some_experiment_config.yaml} \
    -m configs/{some_mapping.yaml} \
    -w configs/{some_workstreams.yaml}
```

---

## Running Tests
Run tests with:

```bash
pytest
```

which should also produce a coverage report due to the default settings in the `pyproject.toml` file.

---

## Documentation

Additional documentation is available in:

| Document | Purpose |
|-----------|-----------|
| architecture.md | Pipeline design, modelling approaches and design rationale |
| experiments.md | Configuration, experiment execution, evaluation and reproducibility |

---
