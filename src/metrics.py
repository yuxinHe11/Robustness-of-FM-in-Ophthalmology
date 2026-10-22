import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
)


def compute_metrics(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    y_pred = np.argmax(y_prob, axis=1)
    num_classes = y_prob.shape[1]

    if num_classes == 2:
        auroc = roc_auc_score(y_true, y_prob[:, 1])
        auprc = average_precision_score(y_true, y_prob[:, 1])
    else:
        auroc = roc_auc_score(
            y_true,
            y_prob,
            multi_class="ovr",
            average="macro",
        )

        one_hot = np.eye(num_classes)[y_true]
        auprc = average_precision_score(
            one_hot,
            y_prob,
            average="macro",
        )

    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)

    return {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "acc": float(acc),
        "bacc": float(bacc),
        "selection_score": float(auroc + f1_macro),
    }