import argparse
import json
import os

import pandas as pd
import yaml
from dwutils import s3
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.evaluate_functions import evaluate_ranked_predictions
from src.experiments.experiment import ExperimentConfig
from src.locations import EXPERIMENTS_HOME, MODELS_HOME
from src.methods import bi_encoder, hardcode, llm, nli, reranker
from src.utility_functions import prepare_model_inputs


def apply_pipeline_step(
    evidence_df: pd.DataFrame,
    step,
    all_workstreams: list[str],
    alb_map: dict,
    directorate_map: dict,
    workstream_objectives: pd.DataFrame,
    evidence_embeddings=None,
    workstream_embeddings=None,
    reranker_model=None,
    nli_model=None,
    nli_tokenizer=None,
    llm_id=None,
    llm_prompt_config=None,
    llm_response_keys_required=None,
) -> pd.DataFrame:
    """
    Apply a single configured pipeline step. Each step updates the dataframe with candidate_workstreams and ranked
    workstream prediction columns.

    """

    step_name = step.name
    top_k = step.top_k

    if step_name == "hardcode":
        outputs = hardcode(
            evidence_df=evidence_df,
            all_workstreams=all_workstreams,
            alb_map=alb_map,
            directorate_map=directorate_map,
        )

    elif step_name == "bi_encoder":
        outputs = bi_encoder(
            evidence_df=evidence_df,
            all_workstreams=all_workstreams,
            evidence_embeddings=evidence_embeddings,
            workstream_embeddings=workstream_embeddings,
            workstream_objectives_df=workstream_objectives,
            top_k=top_k,
        )

    elif step_name == "reranker":
        outputs = reranker(
            evidence_df=evidence_df,
            all_workstreams=all_workstreams,
            workstream_objectives_df=workstream_objectives,
            reranker_model=reranker_model,
            top_k=top_k,
        )

    elif step_name == "nli":
        outputs = nli(
            evidence_df=evidence_df,
            all_workstreams=all_workstreams,
            workstream_objectives_df=workstream_objectives,
            nli_model=nli_model,
            nli_tokenizer=nli_tokenizer,
            top_k=top_k,
        )

    elif step_name == "llm":
        outputs = llm(
            evidence_df=evidence_df,
            all_workstreams=all_workstreams,
            workstream_objectives_df=workstream_objectives,
            llm_id=llm_id,
            prompt_config=llm_prompt_config,
            response_keys_required=llm_response_keys_required,
            top_k=top_k,
        )

    else:
        raise ValueError(f"Unknown pipeline step: '{step_name}'")

    return evidence_df.assign(**outputs)


def run_experiment(
    experiment_config_path: str = "configs/experiment_config.yaml",
    mappings_path: str = "configs/new_mapping.yaml",
    workstreams_path: str = "configs/workstreams.yaml",
) -> None:
    """
    Run the full NLP workstream classification pipeline.

    This function loads an experiment configuration file, reads the required mapping
    and workstream files, and retrieves the top-related workstreams to the evidence
    library records, using the models specified in the configuration. It evaluates the
    ranked predictions, and optionally saves the experiment outputs according to the
    experiment config.

    Args:
        experiment_config_path: str
            Path to the YAML file containing the experiment-specific configuration.
            This should define model choices, pipeline design, datasets used, various
            model settings, whether to save experiment outputs, and any other parameters.

        mappings_path: str = "configs/mapping.yaml"
            Path to the YAML file containing the ALB and directorate mapping dictionaries.
            Defaults to `"configs/workstreams.yaml"`.

        workstreams_path: str = "configs/workstreams.yaml"
            Path to the YAML file containing the workstreams evidence library records are
            mapped to by models. Defaults to `"configs/mapping.yaml"`.

    Returns:
        None. This function runs the experiment and writes output to disk when experiment
        saving is enabled in the config.
    """
    # Check that config files exist
    if not os.path.exists(experiment_config_path):
        raise ValueError(f"Experiment config not found at '{experiment_config_path}'")

    if not os.path.exists(mappings_path):
        raise ValueError(f"Mappings not found at '{mappings_path}'")

    if not os.path.exists(workstreams_path):
        raise ValueError(f"Workstreams not found at '{workstreams_path}'")

    # ------------------------- Load config files -------------------------
    with open(experiment_config_path) as f:
        config_dict = yaml.safe_load(f)

    config = ExperimentConfig(**config_dict)

    with open(mappings_path) as f:
        mapping_dict = yaml.safe_load(f)

    alb_map = mapping_dict["alb_map"]
    directorate_map = mapping_dict["directorate_map"]

    with open(workstreams_path) as f:
        workstream_dict = yaml.safe_load(f)

    print(f"Experiment name: {config.experiment_name}")

    print(
        "Pipeline:",
        " -> ".join(
            [f"{step.name}(top k={step.top_k})" for step in config.pipeline_steps]
        ),
    )

    # Set whether workstreams we are predicting over includes support workstreams or not
    core_workstreams = workstream_dict["core_workstreams"]
    support_workstreams = workstream_dict["support_workstreams"]

    if config.include_support:
        all_workstreams = core_workstreams + support_workstreams
    else:
        all_workstreams = core_workstreams

    # ------------------------- Read in data from Data Workspace shared folder-------------------------
    evidence_library_path = os.path.join("nlp/inputs", config.ev_dataset)
    workstream_objectives_path = os.path.join("nlp/inputs", config.ws_dataset)
    manual_mappings_path = os.path.join("nlp/inputs", config.evaluation_dataset)

    evidence_library_records = pd.read_csv(
        s3.read(team="one_snapshot", path=evidence_library_path)
    ).reset_index(drop=True)
    workstream_objectives = pd.read_csv(
        s3.read(team="one_snapshot", path=workstream_objectives_path)
    ).reset_index(drop=True)
    evaluation_mappings = pd.read_csv(
        s3.read(team="one_snapshot", path=manual_mappings_path)
    )

    # Filter out irrelevant columns from evidence_library_records
    evidence_library_records = evidence_library_records[config.ev_cols_to_keep]

    # Save evidence_library_records index as a column, to preserve access of bi-encoder embeddings even if we filtered evidence_library_records
    evidence_library_records["ev_df_index"] = evidence_library_records.index

    # Drops support workstreams if needed
    workstream_objectives = workstream_objectives[
        workstream_objectives["Workstream"].isin(all_workstreams)
    ].reset_index(drop=True)

    # Filter the data frame that we generate predictions for if we are not predicting non-eval evidence records
    if not config.predict_non_eval:
        evidence_library_records = evidence_library_records[
            evidence_library_records["ID"].isin(evaluation_mappings["ID"])
        ]

    # ------------------------- Load Models -------------------------
    step_names = [step.name for step in config.pipeline_steps]

    biencoder_model = None
    reranker_model = None
    nli_model = None
    nli_tokenizer = None
    llm_prompt_config = None
    llm_response_keys_required = None

    evidence_embeddings = None
    workstream_embeddings = None

    # ------------------------- Load bi-encoder if required -------------------------
    if "bi_encoder" in step_names:
        print(f"Loading bi-encoder model: {config.bi_encoder}")

        biencoder_model = SentenceTransformer(
            os.path.join(MODELS_HOME, config.bi_encoder)
        )

        evidence_inputs, workstream_inputs_dict = prepare_model_inputs(
            evidence_library_records,
            "Description",
            "Title",
            workstream_objectives,
            "Workstream",
            "Objective",
        )

        print("Encoding evidence records...")
        evidence_embeddings = biencoder_model.encode(
            evidence_inputs,
            convert_to_tensor=True,
            batch_size=64,
            show_progress_bar=True,
        )

        print("Encoding workstream objectives...")
        workstream_embeddings = biencoder_model.encode(
            list(workstream_inputs_dict.values()),
            convert_to_tensor=True,
            batch_size=64,
            show_progress_bar=True,
        )

    # ------------------------- Load reranker if required -------------------------
    if "reranker" in step_names:
        print(f"Loading reranker model: {config.reranker}")
        reranker_model = CrossEncoder(os.path.join(MODELS_HOME, config.reranker))

    # ------------------------- Load NLI model if required -------------------------
    if "nli" in step_names:
        print(f"Loading NLI model: {config.nli}")

        nli_tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(MODELS_HOME, config.nli)
        )

        nli_model = AutoModelForSequenceClassification.from_pretrained(
            os.path.join(MODELS_HOME, config.nli)
        )
        nli_model.eval()

    # ------------------------- Load LLM if required -------------------------
    if "llm" in step_names:
        print(f"Using the following LLM from Bedrock: {config.llm_id}")

        with open(config.llm_prompt_config_path) as f:
            llm_prompt_config = yaml.safe_load(f)

        llm_response_keys_required = set(config.llm_response_keys_required)

    # ------------------------- Run configured pipeline ------------------------
    prediction_df = evidence_library_records.copy()

    for step in config.pipeline_steps:
        print(f"\nRunning pipeline step: {step.name}, top_k={step.top_k}")

        prediction_df = apply_pipeline_step(
            evidence_df=prediction_df,
            step=step,
            all_workstreams=all_workstreams,
            alb_map=alb_map,
            directorate_map=directorate_map,
            workstream_objectives=workstream_objectives,
            evidence_embeddings=evidence_embeddings,
            workstream_embeddings=workstream_embeddings,
            reranker_model=reranker_model,
            nli_model=nli_model,
            nli_tokenizer=nli_tokenizer,
            llm_id=config.llm_id,
            llm_prompt_config=llm_prompt_config,
            llm_response_keys_required=llm_response_keys_required,
        )

        print(f"{step.name} output sample:")
        print(prediction_df.head())

    # ------------------------ Evaluation ------------------------
    final_step = step_names[-1]

    if final_step == "bi_encoder":
        prediction_cols_pattern = "ss_workstream_"
    elif final_step == "reranker":
        prediction_cols_pattern = "rr_workstream_"
    elif final_step == "nli":
        prediction_cols_pattern = "nli_workstream_"
    elif final_step == "llm":
        prediction_cols_pattern = "llm_workstream_"

    eval_df, summary_scores, confusion_matrix = evaluate_ranked_predictions(
        prediction_df=prediction_df,
        evaluation_df=evaluation_mappings,
        prediction_cols_pattern=prediction_cols_pattern,
        evidence_id_col="ID",
        true_label_col=config.true_label_col,
        evaluate_support=config.include_support,
        core_workstreams=core_workstreams,
        support_workstreams=support_workstreams,
        recall_at_k_ints=[1, 2, 3, 4, 5],
    )

    print("Summary Scores:\n", summary_scores)
    print("\nConfusion Matrix:\n", confusion_matrix)
    print("\nEvaluation Dataframe Sample:\n", eval_df)

    # ------------------------ Experiment Logging ------------------------
    if config.save_experiment:
        os.makedirs(EXPERIMENTS_HOME, exist_ok=True)

        # Load the experiment registry if it exists
        registry_path = os.path.join(EXPERIMENTS_HOME, "registry.json")

        if os.path.exists(registry_path):
            with open(registry_path) as f:
                registry = json.load(f)
        else:
            registry = {}  # create empty registry dict if registry doesn't exist

        # If an identical experiment has been created before, then we don't save experiment to registry
        if config.experiment_name in registry:
            print(
                f"Experiment {config.experiment_name} was not saved to registry as it already exists."
            )

        else:
            # Save the new experiment run only if there hasn't been an identical run before
            # Build pipeline metadata
            pipeline_metadata = []

            for step in config.pipeline_steps:
                if step.name == "hardcode":
                    pipeline_metadata.append({"step": "hardcode"})

                elif step.name == "bi_encoder":
                    pipeline_metadata.append(
                        {
                            "step": "bi_encoder",
                            "model": config.bi_encoder,
                            "top_k": step.top_k,
                        }
                    )

                elif step.name == "reranker":
                    pipeline_metadata.append(
                        {
                            "step": "reranker",
                            "model": config.reranker,
                            "top_k": step.top_k,
                        }
                    )

                elif step.name == "nli":
                    pipeline_metadata.append(
                        {
                            "step": "nli",
                            "model": config.nli,
                            "top_k": step.top_k,
                        }
                    )

                elif step.name == "llm":
                    pipeline_metadata.append(
                        {
                            "step": "llm",
                            "model": config.llm_id,
                            "top_k": step.top_k,
                        }
                    )

            experiment_number = len(registry) + 1

            experiment_dict = {
                "experiment_number": experiment_number,
                "workstream_descriptions": config.ws_dataset,
                "evidence_records": config.ev_dataset,
                "evaluation_dataset": config.evaluation_dataset,
                "include_support": config.include_support,
                "pipeline": pipeline_metadata,
                **summary_scores,
            }

            registry[config.experiment_name] = experiment_dict

            with open(registry_path, "w") as f:
                json.dump(registry, f, indent=2)

            print(f"Experiment {config.experiment_name} saved to registry.json file")

            # Create experiment run folder path
            run_dir = os.path.join(
                "nlp/experiment_runs",
                f"{experiment_number:03d}_{config.experiment_name}",
            )

            # Save experiment summary
            with s3.write(
                team="one_snapshot",
                path=f"{run_dir}/experiment_summary.json",
                mode="string",
            ) as f:
                json.dump(experiment_dict, f, indent=2)

            # Save input yaml files
            with s3.write(
                team="one_snapshot",
                path=f"{run_dir}/configs/experiment_config.yaml",
                mode="string",
            ) as f:
                yaml.safe_dump(config_dict, f)

            with s3.write(
                team="one_snapshot",
                path=f"{run_dir}/configs/mapping.yaml",
                mode="string",
            ) as f:
                yaml.safe_dump(mapping_dict, f)

            with s3.write(
                team="one_snapshot",
                path=f"{run_dir}/configs/workstreams.yaml",
                mode="string",
            ) as f:
                yaml.safe_dump(workstream_dict, f)

            if "llm" in step_names:
                with s3.write(
                    team="one_snapshot",
                    path=f"{run_dir}/configs/llm_prompt_config.yaml",
                    mode="string",
                ) as f:
                    yaml.safe_dump(llm_prompt_config, f)

            # Save outputs
            with s3.write(
                team="one_snapshot",
                path=f"{run_dir}/outputs/predictions.csv",
                mode="string",
            ) as f:
                prediction_df.to_csv(f, index=False)

            with s3.write(
                team="one_snapshot",
                path=f"{run_dir}/outputs/eval_df.csv",
                mode="string",
            ) as f:
                eval_df.to_csv(f, index=False)

            with s3.write(
                team="one_snapshot",
                path=f"{run_dir}/outputs/confusion_matrix.csv",
                mode="string",
            ) as f:
                confusion_matrix.to_csv(f, index=False)


# Create command-line interface (CLI) functionality
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run an NLP workstream classification experiment."
    )

    parser.add_argument(
        "-c",
        "--config_path",
        default="configs/experiment_config.yaml",
        help="Path to the experiment configuration YAML file.",
    )

    parser.add_argument(
        "-m",
        "--mappings_path",
        default="configs/new_mapping.yaml",
        help="Path to the mappings YAML file. Defaults to configs/mapping.yaml.",
    )

    parser.add_argument(
        "-w",
        "--workstreams_path",
        default="configs/workstreams.yaml",
        help=(
            "Path to the workstreams YAML file. Defaults to configs/workstreams.yaml."
        ),
    )

    args = parser.parse_args()

    run_experiment(
        experiment_config_path=args.config_path,
        mappings_path=args.mappings_path,
        workstreams_path=args.workstreams_path,
    )
