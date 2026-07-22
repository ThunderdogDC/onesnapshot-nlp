import os
import shutil

import pandas as pd


def safe_copy(src: str, dst: str):
    """
    Creates a copy of file from src path and places it in dst path, if there isn't already a file with dst path.
    """
    if not os.path.exists(dst):
        shutil.copy(src, dst)


def prepare_model_inputs(
    evidence_inputs_df,
    ev_desc_col,
    ev_title_col,
    workstream_inputs_df=pd.DataFrame(),
    ws_title_col=None,
    ws_obj_col=None,
):
    """
    Prepare model inputs. Can further modularise this function and set column arguments of each df.

    Returns:
        tuple:
            evidence_inputs_dict (list[str]): A list of evidence library model inputs, ordered by how they appear in evidence_library_df
            workstream_inputs_dict (dict[str, str]): A dictionary mapping workstream names to its formatted model input
    """
    # Set checks to ensure that the necessary columns are within the evidence_inputs_df and workstream_inputs_df
    missing_evidence_cols = {ev_title_col, ev_desc_col} - set(
        evidence_inputs_df.columns
    )
    missing_workstream_cols = {
        col for col in [ws_title_col, ws_obj_col] if col is not None
    } - set(workstream_inputs_df.columns)

    if missing_evidence_cols:
        raise ValueError(f"{missing_evidence_cols} doesn't exist in evidence_inputs_df")
    if missing_workstream_cols:
        raise ValueError(
            f"{missing_workstream_cols} doesn't exist in workstream_inputs_df"
        )

    # Create evidence inputs as list
    evidence_descriptions = evidence_inputs_df[ev_desc_col].to_list()
    evidence_titles = evidence_inputs_df[ev_title_col].to_list()

    evidence_inputs = [
        f"Evidence Title: {title}. Evidence Description: {description}"
        for title, description in zip(evidence_titles, evidence_descriptions)
    ]

    workstream_inputs_dict = {}
    if not workstream_inputs_df.empty:
        # Create workstream inputs as dict
        workstream_titles = workstream_inputs_df[ws_title_col].to_list()
        workstream_descriptions = workstream_inputs_df[ws_obj_col].to_list()

        workstream_inputs_dict = {
            title: f"Workstream: {title}. Objective: {description}"
            for title, description in zip(workstream_titles, workstream_descriptions)
        }

    return (evidence_inputs, workstream_inputs_dict)


def check_ws_mappings(alb_map, directorate_map, workstreams, workstream_objectives_df):
    """
    Checks whether the workstreams defined in alb_map, directorate_map, and workstreams is contained within workstream_objectives_df.
    """
    alb_workstreams = {ws for lst in alb_map.values() for ws in lst}
    dir_workstreams = {ws for lst in directorate_map.values() for ws in lst}
    input_workstreams = set(workstreams)

    map_workstreams = alb_workstreams | dir_workstreams | input_workstreams
    valid_workstreams = set(workstream_objectives_df["Workstream"])

    valid_ws_missing = map_workstreams - valid_workstreams
    map_ws_missing = valid_workstreams - input_workstreams

    return valid_ws_missing, map_ws_missing


def create_initial_similarity_df(
    workstream_df, evidence_df, top_scores, top_indices, TOP_K
):
    rows = []
    for i, ev_row in evidence_df.iterrows():
        row = {
            "id": ev_row["ID"],
            "evidence_title": ev_row["Title"],
            "evidence_description": ev_row["Description"],
            "directorate": ev_row["Directorates"],
            "arms_length_body": ev_row["Arms length body"],
        }

        for rank in range(TOP_K):
            ws_idx = top_indices[i, rank].item()
            row[f"workstream_{rank + 1}"] = workstream_df.iloc[ws_idx]["Workstream"]
            row[f"similarity_{rank + 1}"] = top_scores[i, rank].item()

        rows.append(row)

    matches_df = pd.DataFrame(rows)

    return matches_df
