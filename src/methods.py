import numpy as np
import pandas as pd
import torch
from dwutils import bedrock
from sentence_transformers import util
from tqdm.auto import tqdm

from src import methods_utils


def hardcode(
    evidence_df,
    all_workstreams,
    alb_map,
    directorate_map,
    alb_col="Arms length body",
    directorate_col="Directorates",
):
    """
    Generate candidate workstreams using ALB and directorate hardcoding rules
    Vectorised hardcoding step. Vectorised implementation that operates on the entire
    DataFrame

    Parameters
    ----------
    evidence_df : pandas.DataFrame
        Evidence records data. Must contain the columns
        `Arms length body` and `Directorates`.

    all_workstreams : list[str]
        Complete list of available workstreams. Used as a fallback when
        no ALB or directorate mapping exists.

    alb_map : dict[str, list[str]]
        Mapping from ALB names to allowed workstreams.


    directorate_map : dict[str, list[str]]
        Mapping from directorate names to allowed workstreams.


    Returns
    -------
    dict

        Dictionary containing a single key:

        `candidate_workstreams` : pandas.Series
            A Series of lists where each element contains the candidate
            workstreams for the corresponding evidence record.
    """

    # Use alb and directorate mapping to hardcode evidence records
    candidate_workstreams = (
        evidence_df[alb_col]
        .map(alb_map)
        .combine_first(evidence_df[directorate_col].map(directorate_map))
    )

    # Fallback on all workstreams if no mapping present
    candidate_workstreams = candidate_workstreams.apply(
        lambda x: x if isinstance(x, list) else all_workstreams
    )

    return {"candidate_workstreams": candidate_workstreams}


def bi_encoder(
    evidence_df,
    all_workstreams,
    evidence_embeddings,
    workstream_embeddings,
    workstream_objectives_df,
    top_k=3,
):
    """
    Retrieves the top-k most relevant workstreams for each evidence record
    using cosine similarity between evidence and workstream embeddings.

    If a `candidate_workstreams` column exists, similarity scores are
    calculated only against those workstreams. Otherwise all workstreams
    are considered.

    Parameters
    ----------
    evidence_df : pd.DataFrame
        Evidence records data.

    all_workstreams : list
        List of all possible workstreams.

    evidence_embeddings : torch.Tensor
        Evidence embeddings with shape
        `(n_evidence_records, embedding_dim)`.

    workstream_embeddings : torch.Tensor
        Workstream embeddings with shape
        `(n_workstreams, embedding_dim)`.

    workstream_objectives_df : pd.DataFrame
        Workstream descriptions containing a `Workstream` column.

    top_k : int, default=3
        Number of workstreams to retain.

    Returns
    -------
    dict
        Dictionary containing:

        - `candidate_workstreams`:
          Top-k workstreams for each evidence record.
        - `ss_workstream_1` ... `ss_workstream_k`:
          Retrieved workstreams.
        - `similarity_score_1` ...
          `similarity_score_k`:
          Corresponding cosine similarity scores.
    """

    workstreams = workstream_objectives_df["Workstream"].tolist()
    ws_to_idx = {ws: i for i, ws in enumerate(workstreams)}

    n_rows = len(evidence_df)
    n_workstreams = len(workstreams)

    # Create list of candidate workstream lists
    if "candidate_workstreams" in evidence_df.columns:
        candidate_lists = (
            evidence_df["candidate_workstreams"]
            .apply(lambda x: x if isinstance(x, list) else all_workstreams)
            .tolist()
        )
    else:
        candidate_lists = [all_workstreams] * n_rows

    # Similarity matrix
    sim_matrix = util.cos_sim(evidence_embeddings, workstream_embeddings)

    # Mask invalid workstreams
    mask = torch.zeros(
        (n_rows, n_workstreams), dtype=torch.bool, device=sim_matrix.device
    )

    for row_idx, candidates in enumerate(candidate_lists):
        valid_indices = [ws_to_idx[ws] for ws in candidates]

        if valid_indices:
            mask[row_idx, valid_indices] = True
        else:
            mask[row_idx, :] = True

    masked_sim = sim_matrix.masked_fill(~mask, float("-inf"))

    top_scores, top_indices = torch.topk(masked_sim, k=min(top_k, n_workstreams), dim=1)

    top_scores = top_scores.cpu().numpy()
    top_indices = top_indices.cpu().numpy()

    outputs = {}

    for rank in range(top_scores.shape[1]):
        outputs[f"ss_workstream_{rank + 1}"] = [
            workstreams[idx] if np.isfinite(score) else None
            for score, idx in zip(top_scores[:, rank], top_indices[:, rank])
        ]

        outputs[f"similarity_score_{rank + 1}"] = [
            score if np.isfinite(score) else None for score in top_scores[:, rank]
        ]

    outputs["candidate_workstreams"] = []

    for row_scores, row_indices in zip(top_scores, top_indices):
        candidates = [
            workstreams[idx]
            for score, idx in zip(row_scores, row_indices)
            if np.isfinite(score)
        ]

        outputs["candidate_workstreams"].append(candidates)

    return outputs


def reranker(
    evidence_df,
    all_workstreams,
    workstream_objectives_df,
    reranker_model,
    top_k=3,
    batch_size=32,
):
    """
    Reranks candidate workstreams for each evidence record using a
    cross-encoder model.

    If a `candidate_workstreams` column exists, only those workstreams
    are scored. Otherwise all workstreams are considered. Records with
    a single candidate workstream are classified directly and are not
    passed through the reranker model.

    Parameters
    ----------
    evidence_df : pd.DataFrame
        Evidence records data. Must contain `Title`,
        `Description` and `Directorates` columns.

    all_workstreams : list
        List of all possible workstreams.

    workstream_objectives_df : pd.DataFrame
        Workstream descriptions containing `Workstream`
        and `Objective` columns.

    reranker_model
        Cross-encoder model supporting `predict()`.

    top_k : int, default=3
        Number of workstreams to retain.

    batch_size : int, default=32
        Batch size used for scoring pairs.

    Returns
    -------
    dict
        Dictionary containing:

        - `candidate_workstreams`
        - `rr_workstream_1` ... `rr_workstream_k`
        - `rr_score_1` ... `rr_score_k`
    """

    ws_objectives = dict(
        zip(
            workstream_objectives_df["Workstream"],
            workstream_objectives_df["Objective"],
        )
    )

    if "candidate_workstreams" in evidence_df.columns:
        candidate_lists = (
            evidence_df["candidate_workstreams"]
            .apply(lambda x: x if isinstance(x, list) else all_workstreams)
            .tolist()
        )
    else:
        candidate_lists = [all_workstreams] * len(evidence_df)

    single_idx = [i for i, ws in enumerate(candidate_lists) if len(ws) == 1]
    multi_idx = [i for i, ws in enumerate(candidate_lists) if len(ws) > 1]

    # Initialise outputs
    outputs = {"candidate_workstreams": [None] * len(evidence_df)}

    for rank in range(1, top_k + 1):
        outputs[f"rr_workstream_{rank}"] = [None] * len(evidence_df)
        outputs[f"rr_score_{rank}"] = [None] * len(evidence_df)

    # Populate rows with a single candidate workstream
    for row_idx in single_idx:
        ws = candidate_lists[row_idx][0]

        outputs["candidate_workstreams"][row_idx] = [ws]
        outputs["rr_workstream_1"][row_idx] = ws
        outputs["rr_score_1"][row_idx] = None

    if not multi_idx:
        return outputs

    pairs = []
    metadata = []

    # Only create pairs for rows with multiple candidate workstreams
    for row_idx in multi_idx:
        row = evidence_df.iloc[row_idx]

        evidence_input = f"{row.Title}. {row.Description}"

        candidates = candidate_lists[row_idx]

        for ws in candidates:
            workstream_input = f"{ws}, {ws_objectives[ws]}"

            pairs.append((evidence_input, workstream_input))

            metadata.append({"row_idx": row_idx, "workstream": ws})

    # Score all pairs in a single batched call
    scores = reranker_model.predict(
        pairs, batch_size=batch_size, show_progress_bar=True
    )

    scores_df = pd.DataFrame(metadata)
    scores_df["rr_score"] = scores

    # Rank workstreams within each evidence record
    scores_df = scores_df.sort_values(["row_idx", "rr_score"], ascending=[True, False])

    scores_df["rank"] = scores_df.groupby("row_idx").cumcount().add(1)

    scores_df = scores_df[scores_df["rank"] <= top_k]

    for rank in range(1, top_k + 1):
        rank_df = scores_df[scores_df["rank"] == rank].set_index("row_idx")

        for row_idx in multi_idx:
            outputs[f"rr_workstream_{rank}"][row_idx] = rank_df["workstream"].get(
                row_idx
            )

            outputs[f"rr_score_{rank}"][row_idx] = rank_df["rr_score"].get(row_idx)

    for row_idx in multi_idx:
        outputs["candidate_workstreams"][row_idx] = scores_df.loc[
            scores_df["row_idx"] == row_idx, "workstream"
        ].tolist()

    return outputs


def nli(
    evidence_df,
    all_workstreams,
    workstream_objectives_df,
    nli_model,
    nli_tokenizer,
    top_k=3,
    batch_size=32,
):
    """
    Reranks candidate workstreams for each evidence record using a
    Natural Language Inference (NLI) model.

    For every evidence-workstream pair, the evidence is treated as the
    premise and the workstream objective is converted into a hypothesis.
    The model's entailment probability is used as a relevance score.

    If a `candidate_workstreams` column exists, only those workstreams
    are scored. Otherwise all workstreams are considered.

    Parameters
    ----------
    evidence_df : pd.DataFrame
        Evidence records data. Must contain `Title` and `Description`
        columns.

    all_workstreams : list[str]
        Complete list of available workstreams.

    workstream_objectives_df : pd.DataFrame
        Workstream metadata containing `Workstream` and `Objective`
        columns.

    nli_model
        Hugging Face model trained for
        natural language inference.

    nli_tokenizer
        Tokenizer corresponding to `nli_model`.

    top_k : int, default=3
        Number of highest-scoring workstreams to retain for each
        evidence record.

    batch_size : int, default=32
        Number of evidence-workstream pairs processed per inference
        batch. Increasing this may improve throughput but will require
        more memory.

    Returns
    -------
    dict
        Dictionary containing:

        - `candidate_workstreams`
        - `nli_workstream_1` ... `nli_workstream_k`
        - `nli_score_1` ... `nli_score_k`
    """

    ws_objectives = dict(
        zip(
            workstream_objectives_df["Workstream"],
            workstream_objectives_df["Objective"],
        )
    )

    if "candidate_workstreams" in evidence_df.columns:
        candidate_lists = (
            evidence_df["candidate_workstreams"]
            .apply(lambda x: x if isinstance(x, list) else all_workstreams)
            .tolist()
        )
    else:
        candidate_lists = [all_workstreams] * len(evidence_df)

    single_idx = [i for i, ws in enumerate(candidate_lists) if len(ws) == 1]
    multi_idx = [i for i, ws in enumerate(candidate_lists) if len(ws) > 1]

    # Initialise outputs
    outputs = {"candidate_workstreams": [None] * len(evidence_df)}

    for rank in range(1, top_k + 1):
        outputs[f"nli_workstream_{rank}"] = [None] * len(evidence_df)
        outputs[f"nli_score_{rank}"] = [None] * len(evidence_df)

    # Populate rows with a single candidate workstream
    for row_idx in single_idx:
        ws = candidate_lists[row_idx][0]

        outputs["candidate_workstreams"][row_idx] = [ws]
        outputs["nli_workstream_1"][row_idx] = ws
        outputs["nli_score_1"][row_idx] = None

    if not multi_idx:
        return outputs

    # Construct NLI premise/hypothesis pairs for evidence-workstream combinations.
    # Only include records that have multiple candidate workstreams

    premises = []
    hypotheses = []
    metadata = []

    for row_idx in multi_idx:
        row = evidence_df.iloc[row_idx]

        premise = (
            f"Evidence Title: {row.Title}. Evidence Description: {row.Description}"
        )

        for ws in candidate_lists[row_idx]:
            hypothesis = (
                f"This evidence relates to the following workstream:{ws_objectives[ws]}"
            )

            premises.append(premise)
            hypotheses.append(hypothesis)

            metadata.append({"row_idx": row_idx, "workstream": ws})

    # Determine which output logit corresponds to entailment.
    entailment_idx = {
        label.lower(): idx for idx, label in nli_model.config.id2label.items()
    }["entailment"]

    device = next(nli_model.parameters()).device

    all_scores = []

    # Run NLI inference in batches to limit memory usage.
    # The premise/hypothesis pairs are tokenised and scored,
    # and the entailment probability is retained.

    for start in tqdm(range(0, len(premises), batch_size), desc="NLI scoring"):
        end = start + batch_size

        inputs = nli_tokenizer(
            premises[start:end],
            hypotheses[start:end],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            model_outputs = nli_model(**inputs)

        probs = torch.softmax(model_outputs.logits, dim=1)

        batch_scores = probs[:, entailment_idx].cpu().numpy()

        all_scores.extend(batch_scores)

    scores = np.array(all_scores)

    # Rank candidate workstreams within each evidence record
    scores_df = pd.DataFrame(metadata)
    scores_df["nli_score"] = scores

    scores_df = scores_df.sort_values(
        ["row_idx", "nli_score"],
        ascending=[True, False],
    )

    scores_df["rank"] = scores_df.groupby("row_idx").cumcount().add(1)

    scores_df = scores_df[scores_df["rank"] <= top_k]

    for rank in range(1, top_k + 1):
        rank_df = scores_df[scores_df["rank"] == rank].set_index("row_idx")

        for row_idx in multi_idx:
            outputs[f"nli_workstream_{rank}"][row_idx] = rank_df["workstream"].get(
                row_idx
            )

            outputs[f"nli_score_{rank}"][row_idx] = rank_df["nli_score"].get(row_idx)

    for row_idx in multi_idx:
        outputs["candidate_workstreams"][row_idx] = scores_df.loc[
            scores_df["row_idx"] == row_idx, "workstream"
        ].tolist()
    return outputs


def llm(
    evidence_df: pd.DataFrame,
    all_workstreams: list,
    workstream_objectives_df: pd.DataFrame,
    llm_id: str,
    prompt_config: dict,
    response_keys_required: set,
    top_k=3,
):
    """
    Ranks candidate workstreams for a particular evidence record using an LLM.
    Note: this function should strictly be used after a retrieval step that narrows the candidate workstreams set.
    top_k should be set to be quite restrictive, as unlike the NLI and reranker functions, all candidate
    workstreams are assessed together by being passed as a prompt to the LLM.

    Args:
       evidence_df: pd.DataFrame
            The DataFrame containing evidence records to classify.

        all_workstreams: list
            List of all possible workstreams.

        workstream_objectives_df: pd.DataFrame
            Data frame containing workstream names and objectives.
            Must contain 'Workstream' and 'Objective' columns.

        llm_id: str
            LLM used to assess which workstream the evidence belongs to.

        prompt_config: dict
           LLM prompt config dictionary, so that prompt_config["prompt_template"] contains the LLM prompt template.

        response_keys_required: set
            The required/expected keys to be returned by an LLM in its JSON output.

        top_k: int, default=3
            The top_k workstreams from candidate_workstreams are passed in the prompt to the LLM.
            If len(candidate_workstreams) > top_k, only the first top_k workstreams in candidate_workstreams are used in the LLM prompt.
            Note: This is different from the top_k used in nli and reranker functions, which determine the number of top_k workstreams to return.

    Returns:
        row: pd.Series
            The row with new columns to highlight the LLM's rankings and the LLM's scores for each evidence-workstream relevance.
    """
    # Initialise outputs
    outputs = {"candidate_workstreams": [None] * len(evidence_df)}

    for rank in range(1, top_k + 1):
        outputs[f"llm_workstream_{rank}"] = [None] * len(evidence_df)

    outputs["llm_error"] = [None] * len(evidence_df)
    outputs["llm_justification"] = [None] * len(evidence_df)
    outputs["llm_num_repairs"] = [None] * len(evidence_df)
    outputs["llm_error_history"] = [None] * len(evidence_df)

    # Workstream objective lookup
    workstreams_dict = dict(
        zip(
            workstream_objectives_df["Workstream"],
            workstream_objectives_df["Objective"],
        )
    )

    # Get candidate workstream lists
    if "candidate_workstreams" in evidence_df.columns:
        candidate_lists = (
            evidence_df["candidate_workstreams"]
            .apply(lambda x: x if isinstance(x, list) else all_workstreams)
            .tolist()
        )
    else:
        candidate_lists = [all_workstreams] * len(evidence_df)

    # Truncate candidate lists for LLM prompt
    candidate_lists = [ws_list[:top_k] for ws_list in candidate_lists]

    # Directly assign rows with a single candidate to reduce latency when using LLM
    single_idx = [i for i, ws in enumerate(candidate_lists) if len(ws) == 1]
    multi_idx = [i for i, ws in enumerate(candidate_lists) if len(ws) > 1]

    for row_idx in single_idx:
        ws = candidate_lists[row_idx][0]

        outputs["candidate_workstreams"][row_idx] = [ws]
        outputs["llm_workstream_1"][row_idx] = ws

    if not multi_idx:
        return outputs

    # Build the prompts
    prompts = []

    for row_idx in multi_idx:
        row = evidence_df.iloc[row_idx]
        candidate_workstreams = candidate_lists[row_idx]

        candidate_ws_desc = [workstreams_dict[ws] for ws in candidate_workstreams]

        prompt = methods_utils.build_llm_prompt(
            prompt_template=prompt_config["prompt_template"],
            ev_title=row["Title"],
            ev_desc=row["Description"],
            ev_dir=row["Directorates"],
            cand_ws_titles=candidate_workstreams,
            cand_ws_desc=candidate_ws_desc,
        )

        prompts.append((row_idx, prompt))

    prompt_lookup = dict(prompts)

    # Assign workstreams using LLM
    for row_idx, result in bedrock.invoke_bulk(prompts=prompts, model_id=llm_id):
        candidate_workstreams = candidate_lists[row_idx]

        processed_response = methods_utils.process_response_wrapper(
            llm_response=result["model_response_string"],
            original_prompt=prompt_lookup[row_idx],
            response_keys_required=response_keys_required,
            candidate_workstreams=candidate_workstreams,
            model_id=llm_id,
            max_repairs=3,
        )

        if processed_response.response:
            rankings = processed_response.response["rankings"]

            for ws_dict in rankings:
                rank = ws_dict["rank"]
                ws = ws_dict["workstream"]
                outputs[f"llm_workstream_{rank}"][row_idx] = ws

            ranked_ws = [
                ws["workstream"] for ws in sorted(rankings, key=lambda x: x["rank"])
            ]

            outputs["candidate_workstreams"][row_idx] = ranked_ws
            outputs["llm_error"][row_idx] = processed_response.error
            outputs["llm_justification"][row_idx] = processed_response.response[
                "justification"
            ]
            outputs["llm_num_repairs"][row_idx] = processed_response.repair_attempts
            outputs["llm_error_history"][row_idx] = processed_response.error_history

    return outputs
