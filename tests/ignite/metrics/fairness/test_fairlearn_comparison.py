import pytest
import torch
from torch.testing import assert_close

# Skip if fairlearn is not installed
pytest.importorskip("fairlearn")

from fairlearn.metrics import MetricFrame, demographic_parity_difference
from sklearn.metrics import accuracy_score

from ignite.metrics.fairness.accuracy_difference import SubgroupAccuracyDifference
from ignite.metrics.fairness.demographic_parity import DemographicParityDifference


def test_compare_accuracy_difference_with_fairlearn():
    """Verifies SubgroupAccuracyDifference matches Fairlearn's MetricFrame(accuracy_score).difference()"""
    groups_list = [0, 1]
    ignite_metric = SubgroupAccuracyDifference(groups=groups_list)

    # Random binary data
    y_pred_probs = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.2, 0.8], [0.7, 0.3]])
    y_pred = torch.argmax(y_pred_probs, dim=1)
    y_true = torch.tensor([1, 0, 1, 1, 0, 0])
    group_labels = torch.tensor([0, 0, 0, 1, 1, 1])

    # Ignite update and compute
    ignite_metric.update((y_pred_probs, y_true, group_labels))
    ignite_result = ignite_metric.compute()

    # Fairlearn computation
    # Fairlearn's MetricFrame takes numpy arrays
    mf = MetricFrame(
        metrics=accuracy_score, y_true=y_true.numpy(), y_pred=y_pred.numpy(), sensitive_features=group_labels.numpy()
    )
    fairlearn_result = mf.difference()

    assert_close(ignite_result, float(fairlearn_result))


def test_compare_demographic_parity_with_fairlearn():
    """Verifies DemographicParityDifference matches Fairlearn's demographic_parity_difference"""
    groups_list = [0, 1]
    ignite_metric = DemographicParityDifference(groups=groups_list)

    # Multi-class case
    # G0: [0, 1, 0], G1: [1, 1, 1]
    y_pred_probs = torch.tensor(
        [
            [0.9, 0.1, 0.0],  # G0 -> class 0
            [0.1, 0.8, 0.1],  # G0 -> class 1
            [0.9, 0.1, 0.0],  # G0 -> class 0
            [0.1, 0.8, 0.1],  # G1 -> class 1
            [0.1, 0.8, 0.1],  # G1 -> class 1
            [0.1, 0.8, 0.1],  # G1 -> class 1
        ]
    )
    y_true = torch.zeros(6)  # ignored by SelectionRate
    group_labels = torch.tensor([0, 0, 0, 1, 1, 1])

    ignite_metric.update((y_pred_probs, y_true, group_labels))
    ignite_result = ignite_metric.compute()

    # Fairlearn verification for multiclass
    # We calculate demographic parity difference for each class independently (one-vs-rest)
    # and take the max, which is how Ignite's SubgroupDifference handles vector outputs.
    y_pred_classes = torch.argmax(y_pred_probs, dim=1).numpy()
    fairlearn_max_diff = 0.0
    for c in range(3):
        y_pred_bin = (y_pred_classes == c).astype(int)
        diff = demographic_parity_difference(
            y_true=y_true.numpy(), y_pred=y_pred_bin, sensitive_features=group_labels.numpy()
        )
        fairlearn_max_diff = max(fairlearn_max_diff, diff)

    assert_close(ignite_result, float(fairlearn_max_diff))

    # Simple binary case
    y_pred_binary = torch.tensor([1, 0, 1, 0, 0, 0])
    group_labels_binary = torch.tensor([0, 0, 0, 1, 1, 1])

    ignite_metric.reset()
    ignite_metric.update((y_pred_binary, y_true, group_labels_binary))
    ignite_res_bin = ignite_metric.compute()

    fairlearn_res_bin = demographic_parity_difference(
        y_true=y_true.numpy(), y_pred=y_pred_binary.numpy(), sensitive_features=group_labels_binary.numpy()
    )

    assert_close(ignite_res_bin, float(fairlearn_res_bin))
