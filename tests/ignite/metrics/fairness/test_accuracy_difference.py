import pytest
from typing import Callable
import torch
from torch.testing import assert_close

import ignite.distributed as idist
from ignite.exceptions import NotComputableError
from ignite.metrics.fairness.accuracy_difference import SubgroupAccuracyDifference


def test_subgroup_accuracy_difference_empty() -> None:
    """Tests if NotComputableError is raised when no data is provided."""
    metric = SubgroupAccuracyDifference(groups=[0, 1])
    with pytest.raises(NotComputableError, match="Fairness metrics must have at least one example"):
        metric.compute()


def test_subgroup_accuracy_difference_single_group() -> None:
    """Tests if NotComputableError is raised when only one subgroup is present."""
    metric = SubgroupAccuracyDifference(groups=[0])

    y_pred = torch.tensor([[0.9, 0.1], [0.8, 0.2]])
    y = torch.tensor([0, 0])
    group_labels = torch.tensor([0, 0])

    metric.update((y_pred, y, group_labels))

    with pytest.raises(NotComputableError, match="Fairness metrics require at least two unique subgroups"):
        metric.compute()


def test_subgroup_accuracy_difference_binary_labels() -> None:
    """Tests SubgroupAccuracyDifference with binary 0/1 labels."""
    metric = SubgroupAccuracyDifference(groups=[0, 1])

    # y_pred and y are (B,)
    # Group 0: 2/2 correct (1.0)
    # Group 1: 0/2 correct (0.0)
    y_pred = torch.tensor([1, 0, 1, 0])
    y = torch.tensor([1, 0, 0, 1])
    groups = torch.tensor([0, 0, 1, 1])

    metric.update((y_pred, y, groups))
    assert_close(metric.compute(), 1.0)


def test_subgroup_accuracy_difference_binary_probs() -> None:
    """Tests SubgroupAccuracyDifference with (B, 1) thresholded labels."""
    metric = SubgroupAccuracyDifference(groups=[0, 1])

    # y_pred is (B, 1), y is (B,)
    # Group 0: 2/2 correct (1.0)
    # Group 1: 1/2 correct (0.5)
    y_pred = torch.tensor([[1], [0], [1], [1]])
    y = torch.tensor([1, 0, 1, 0])
    groups = torch.tensor([0, 0, 1, 1])

    metric.update((y_pred, y, groups))
    assert_close(metric.compute(), 0.5)


def test_subgroup_accuracy_difference_multiclass() -> None:
    """Tests SubgroupAccuracyDifference with multiclass logits."""
    metric = SubgroupAccuracyDifference(groups=[0, 1])

    # y_pred is (B, C), y is (B,)
    # Group 0: 1/1 correct (1.0)
    # Group 1: 0/1 correct (0.0)
    y_pred = torch.tensor([[0.1, 0.8, 0.1], [0.8, 0.1, 0.1]])
    y = torch.tensor([1, 1])
    groups = torch.tensor([0, 1])

    metric.update((y_pred, y, groups))
    assert_close(metric.compute(), 1.0)


def test_subgroup_accuracy_difference_multilabel() -> None:
    """Tests SubgroupAccuracyDifference with multilabel data."""
    metric = SubgroupAccuracyDifference(groups=[0, 1], is_multilabel=True)

    # y_pred and y are (B, C)
    # Accuracy uses sample-wise correctness: all labels must match per sample.
    # Group 0 (sample 0): y_pred=[1,0], y=[1,0] -> all match -> correct. Accuracy = 1/1 = 1.0
    # Group 1 (sample 1): y_pred=[1,1], y=[1,0] -> not all match -> incorrect. Accuracy = 0/1 = 0.0
    y_pred = torch.tensor([[1, 0], [1, 1]])
    y = torch.tensor([[1, 0], [1, 0]])
    groups = torch.tensor([0, 1])

    metric.update((y_pred, y, groups))
    assert_close(metric.compute(), 1.0)


def test_subgroup_accuracy_difference_spatial() -> None:
    """Tests SubgroupAccuracyDifference with spatial (image) data."""
    metric = SubgroupAccuracyDifference(groups=[0, 1])

    # y_pred is (B, C, H, W), y is (B, H, W)
    # B=2, C=2, H=2, W=2
    # Group 0: spatial targets (4 pixels) all correct (1.0)
    # Group 1: spatial targets (4 pixels) all incorrect (0.0)

    y_pred = torch.zeros(2, 2, 2, 2)
    # Group 0 (index 0): predict class 0 for all
    y_pred[0, 0, :, :] = 1.0
    # Group 1 (index 1): predict class 0 for all
    y_pred[1, 0, :, :] = 1.0

    y = torch.zeros(2, 2, 2, dtype=torch.long)
    # Group 0: all class 0 (1.0 acc)
    y[0, :, :] = 0
    # Group 1: all class 1 (0.0 acc)
    y[1, :, :] = 1

    groups = torch.tensor([0, 1])

    metric.update((y_pred, y, groups))
    assert_close(metric.compute(), 1.0)


def _test_distrib_integration(device: torch.device) -> None:
    """Helper to test distributed integration."""
    rank = idist.get_rank()
    metric = SubgroupAccuracyDifference(groups=[0, 1], device=device)

    y = torch.tensor([0, 1], device=device)
    groups = torch.tensor([0, 1], device=device)

    if rank == 0:
        # Group 0: correct, Group 1: incorrect
        y_pred = torch.tensor([[0.8, 0.2], [0.8, 0.2]], device=device)
    else:
        # Group 0: correct, Group 1: incorrect
        y_pred = torch.tensor([[0.9, 0.1], [0.9, 0.1]], device=device)

    metric.update((y_pred, y, groups))
    res = metric.compute()

    # Across all ranks:
    # Group 0: all correct (1.0)
    # Group 1: all incorrect (0.0)
    assert_close(res, 1.0)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
def test_distrib_cpu(distributed_context_single_node_gloo: Callable) -> None:
    """Tests distributed integration on CPU."""
    device = torch.device("cpu")
    _test_distrib_integration(device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_distrib_gpu(distributed_context_single_node_nccl: Callable) -> None:
    """Tests distributed integration on GPU."""
    device = torch.device(f"cuda:{idist.get_local_rank()}")
    _test_distrib_integration(device)
