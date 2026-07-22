import re
from dataclasses import dataclass, field

from src.get_models import MODEL_SHORT_NAMES


@dataclass
class PipelineStepConfig:
    name: str
    top_k: int | None = None

    def __post_init__(self):
        VALID_STEPS = {"hardcode", "bi_encoder", "reranker", "nli", "llm"}

        if self.name not in VALID_STEPS:
            raise ValueError(
                f"Invalid pipeline step '{self.name}'. "
                f"Valid steps are: {sorted(VALID_STEPS)}"
            )


@dataclass
class ExperimentConfig:
    # data configs
    ws_dataset: str
    ev_dataset: str
    evaluation_dataset: str

    # Evidence library columns to keep in any outputs
    ev_cols_to_keep: list

    # Model configs
    bi_encoder: str | None = None
    reranker: str | None = None
    nli: str | None = None
    llm_id: str | None = None
    llm_prompt_config_path: str | None = None
    llm_response_keys_required: list | None = None

    # Pipeline
    pipeline: list[dict] = field(default_factory=list)

    # Evaluation configs
    true_label_col: str = "manual_workstream_mapping"
    include_non_odp: bool = False
    predict_non_eval: bool = False  # whether to generate predictions for evidence records we don't have labels for

    save_experiment: bool = False

    # Set up experiment_name field but don't initialise
    experiment_name: str = field(init=False)
    pipeline_steps: list[PipelineStepConfig] = field(init=False)

    def __post_init__(self):
        # Convert raw pipeline dictionaries into PipelineStepConfig objects
        self.pipeline_steps = [PipelineStepConfig(**step) for step in self.pipeline]

        if not self.pipeline_steps:
            raise ValueError("Pipeline must contain at least one step.")

        step_names = [step.name for step in self.pipeline_steps]

        # Set some checks
        if "bi_encoder" in step_names:
            if self.bi_encoder is None:
                raise ValueError(
                    "Bi-encoder model must be provided when pipeline includes 'bi_encoder'."
                )

            if self.bi_encoder not in MODEL_SHORT_NAMES:
                raise ValueError(
                    "Bi-encoder model provided must exist in get_models.py."
                )

        if "reranker" in step_names:
            if self.reranker is None:
                raise ValueError(
                    "Reranker model must be provided when pipeline includes 'reranker'."
                )

            if self.reranker not in MODEL_SHORT_NAMES:
                raise ValueError("Reranker model provided must exist in get_models.py.")

        if "nli" in step_names:
            if self.nli is None:
                raise ValueError(
                    "NLI model must be provided when pipeline includes 'nli'."
                )

            if self.nli not in MODEL_SHORT_NAMES:
                raise ValueError("NLI model provided must exist in get_models.py.")

        if "llm" in step_names:
            if self.llm_id is None:
                raise ValueError(
                    "LLM must be provided when the pipeline includes 'llm'."
                )

            if self.llm_id not in MODEL_SHORT_NAMES:
                raise ValueError("LLM provided must exist in get_models.py.")

            if self.llm_prompt_config_path is None:
                raise ValueError(
                    "LLM prompt config path must be provided when the pipeline includes 'llm'."
                )

            if self.llm_response_keys_required is None:
                raise ValueError(
                    "LLM response keys required must be provided when the pipeline includes 'llm'."
                )

        # Extract datasets
        ws_version = re.search(r"v\d+(?=\.csv$)", self.ws_dataset).group()
        ev_version = re.search(r"v\d+(?=\.csv$)", self.ev_dataset).group()

        # Build compact pipeline name
        pipeline_name_parts = []

        for step in self.pipeline_steps:
            if step.name == "hardcode":
                pipeline_name_parts.append("hc")

            elif step.name == "bi_encoder":
                bi_enc = MODEL_SHORT_NAMES[self.bi_encoder]
                pipeline_name_parts.append(f"be-{bi_enc}-k{step.top_k}")

            elif step.name == "reranker":
                rr = MODEL_SHORT_NAMES[self.reranker]
                pipeline_name_parts.append(f"rr-{rr}-k{step.top_k}")

            elif step.name == "nli":
                nli = MODEL_SHORT_NAMES[self.nli]
                pipeline_name_parts.append(f"nli-{nli}-k{step.top_k}")

            elif step.name == "llm":
                llm_short = MODEL_SHORT_NAMES[self.llm_id]
                pipeline_name_parts.append(f"llm-{llm_short}-k{step.top_k}")

        pipeline_name = "_".join(pipeline_name_parts)

        if self.include_non_odp:
            scope = "all"
        else:
            scope = "odp_only"

        # Define the experiment name using various arguments
        self.experiment_name = (
            f"ws-{ws_version}_ev-{ev_version}_{pipeline_name}_scope-{scope}"
        )
