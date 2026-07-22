import pandas as pd


def get_ranked_workstream_columns(df: pd.DataFrame, starting_pattern: str) -> list[str]:
    """Return ranked prediction columns in numeric order, e.g. ss_workstream_1, ss_workstream_2, ..."""
    ranked_cols = [col for col in df.columns if col.startswith(starting_pattern)]
    ranked_cols = sorted(ranked_cols, key=lambda x: int(x.split("_")[-1]))
    return ranked_cols


def compute_top1_pred_metrics(
    eval_df: pd.DataFrame, top1_pred_col: str, true_label_col: str, classes: list[str]
) -> tuple[dict[str, float], pd.DataFrame]:
    """
    Computes precision, recall, and f1 score across each class using the top prediction from eval_df.

    Args:
    """
    records = []

    for cls in classes:
        tp = ((eval_df[top1_pred_col] == cls) & (eval_df[true_label_col] == cls)).sum()
        fp = ((eval_df[top1_pred_col] == cls) & (eval_df[true_label_col] != cls)).sum()
        fn = ((eval_df[top1_pred_col] != cls) & (eval_df[true_label_col] == cls)).sum()
        records.append({"workstream": cls, "TP": tp, "FP": fp, "FN": fn})

    confusion_by_class = pd.DataFrame(records)

    confusion_by_class["precision"] = round(
        confusion_by_class["TP"]
        / (confusion_by_class["TP"] + confusion_by_class["FP"]),
        2,
    )
    confusion_by_class["recall"] = round(
        confusion_by_class["TP"]
        / (confusion_by_class["TP"] + confusion_by_class["FN"]),
        2,
    )
    confusion_by_class["f1"] = round(
        (
            2
            * confusion_by_class["precision"]
            * confusion_by_class["recall"]
            / (confusion_by_class["precision"] + confusion_by_class["recall"])
        ),
        2,
    )

    accuracy = (eval_df[top1_pred_col] == eval_df[true_label_col]).mean()

    summary_scores = {
        "accuracy": round(accuracy, 2),
        "macro_precision": round(confusion_by_class["precision"].mean(), 2),
        "macro_recall": round(confusion_by_class["recall"].mean(), 2),
        "macro_f1_score": round(confusion_by_class["f1"].mean(), 2),
    }

    return summary_scores, confusion_by_class


def evaluate_ranked_predictions(
    prediction_df: pd.DataFrame,
    evaluation_df: pd.DataFrame,
    prediction_cols_pattern: str,
    evidence_id_col: str,
    true_label_col: str,
    evaluate_support: bool,
    core_workstreams: list[str],
    support_workstreams: list[str],
    recall_at_k_ints: list[int],
) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
    """
    Evaluates workstream predictions from some model against evaluation data.

    Args:
        prediction_df: pd.DataFrame
            A DataFrame containing an NLP model's workstream predictions of evidence library records.

        evaluation_df: pd.DataFrame
            A DataFrame containing the true workstream of some evidence library records.

        prediction_cols_pattern: str
            The starting pattern of the columns containing predictions, e.g. "ss_workstream_", "nli_workstream", "llm_workstream", etc.

        evidence_id_col: str
            The name of the column that holds the IDs of each evidence library record.

        true_label_col: str
            The name of the column in evaluation_df that holds the true labels.

        evaluate_support: bool
            Whether to include evidence records belonging to support workstreams in evaluation or not.

        core_workstreams: list[str]
            A list of core workstream names.

        support_workstreams: list[str]
            A list of support workstream names.

        recall_at_k_ints: list[int]
            A list of integers to compute recall for.
    """
    eval_df = pd.merge(prediction_df, evaluation_df, "inner", on=evidence_id_col)

    classes = (
        core_workstreams + support_workstreams if evaluate_support else core_workstreams
    )

    # Filter eval_df to contain only records where the true mapping and predicted mapping are within the classes defined
    top1_pred_col = prediction_cols_pattern + "1"
    eval_df = eval_df[
        eval_df[true_label_col].isin(classes) & eval_df[top1_pred_col].isin(classes)
    ]

    # Retrieve predictions and true labels
    pred_col_names = get_ranked_workstream_columns(eval_df, prediction_cols_pattern)
    predictions = eval_df[pred_col_names].to_numpy()
    true_labels = eval_df[true_label_col].to_numpy()[
        :, None
    ]  # [:, None] makes true_labels have shape (n, 1) instead of (n,) for broadcasting

    # Compute top 1 prediction metrics
    summary_scores, confusion_by_class = compute_top1_pred_metrics(
        eval_df, top1_pred_col, true_label_col, classes
    )

    # Compute recall@K
    recall_at_k_ints = sorted(set(recall_at_k_ints))
    available_k = len(pred_col_names)

    if max(recall_at_k_ints) > available_k:
        print(
            f"Warning: Requested recall@{max(recall_at_k_ints)} "
            f"but only {available_k} predictions available."
        )

    for k in recall_at_k_ints:
        k_effective = min(k, available_k)

        top_k_preds = predictions[:, :k_effective]  # shape (n, k)
        eval_df[f"recall_at_{k}"] = (top_k_preds == true_labels).any(axis=1).astype(int)
        summary_scores[f"recall_at_{k}"] = round(eval_df[f"recall_at_{k}"].mean(), 2)

    return eval_df, summary_scores, confusion_by_class
