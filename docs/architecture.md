# Architecture

## Overview

The project maps evidence library records to DBT workstreams.

Rather than implementing a single fixed model, the codebase provides a configurable experimentation framework that allows different retrieval and ranking methods to be combined and evaluated.

The experiment runner is implemented in:

```text
src/main.py
```

The workflow is driven by YAML configuration files.

---

## Pipeline Overview

Typical pipeline:

```text
Evidence Record
       │
       ▼
Hardcoded Filtering
       │
       ▼
Bi-Encoder Retrieval
       │
       ▼
Reranker / NLI / LLM
       │
       ▼
Evaluation
```

Not every stage is required.

For example:

```yaml
pipeline:
  - name: hardcode

  - name: bi_encoder
    top_k: 5

  - name: llm
    top_k: 5
```

---

## Core Modules

### main.py

Responsible for:

- reading configuration files
- reading datasets
- loading models
- applying pipeline stages
- evaluation
- experiment saving

---

### methods.py

Contains modelling methods:

- hardcode
- bi_encoder
- reranker
- nli
- llm

Each method can be used as a pipeline step.

---

### methods_utils.py

Contains helper functions including:

- candidate workstream generation
- prompt building
- LLM response parsing
- LLM response validation
- repair-prompt generation

---

### experiment.py

Defines the experiment configuration schema and validation logic used by the framework.

Responsible for:

- validating experiment configuration files
- validating pipeline stages and required model settings
- converting pipeline configuration into structured objects
- generating standardised experiment names
- enforcing configuration consistency before experiments are executed

The module provides the `ExperimentConfig` and `PipelineStepConfig`
dataclasses used throughout the framework.

---

### evaluate_functions.py

Responsible for:

- precision
- recall
- f1
- recall@k
- confusion matrices

---

## Pipeline Stages

### Hardcode

Uses ALB and directorate mappings to narrow candidate workstreams.

Benefits:

- cheap
- deterministic
- can improve downstream precision

---

### Bi-Encoder

Embeds evidence records and workstream objectives into a shared vector space.

Used for:

- efficient retrieval
- narrowing candidate workstreams


---

### Reranker

Uses a cross-encoder to rescore candidates produced by retrieval.

Useful when:

- retrieval finds reasonable candidates
- ranking quality matters

---

### NLI

Treats workstream assignment as an entailment task.

Useful for comparing retrieval-based approaches against sequence-classification approaches.

---

### LLM

Uses AWS Bedrock models to compare candidate workstreams directly.

The LLM stage is intended to operate on a small candidate set rather than the full workstream space.

---

## LLM Validation

LLM outputs are validated before use.

Checks include:

- valid JSON
- required keys present
- valid workstream names
- sequential rankings
- complete rankings

If validation fails:

1. Errors are identified
2. A repair prompt is generated
3. The model is reprompted

This improves robustness against non-deterministic LLM outputs.

---

## Design Decisions

### Why use retrieval before LLM?

The complete workstream set is too large to efficiently compare within a single prompt.

Retrieval narrows the search space before the LLM is used.

Benefits:

- lower cost
- lower latency
- smaller prompts

---

### Why validate LLM outputs?

LLMs occasionally produce:

- malformed JSON
- extra explanatory text
- invalid workstream names

Validation prevents these outputs from corrupting downstream results.

---

### Why save experiment metadata?

Reproducibility is a key project objective.

Each experiment can be recreated from:

- configuration files
- experiment metadata
- dataset versions