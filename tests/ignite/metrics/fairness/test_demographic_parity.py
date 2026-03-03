import pytest
from typing import Callable
import torch
from torch.testing import assert_close

import ignite.distributed as idist
from ignite.exceptions import NotComputableError
from ignite.metrics.fairness.demographic_parity import DemographicParityDifference


def test_demographic_parity_difference_empty() -> None:
    """Tests if NotComputableError is raised when no data is provided."""
    metric = DemographicParityDifference()
    with pytest.raises(NotComputableError, match="Fairness metrics must have at least one example"):
        metric.compute()


def test_demographic_parity_difference_single_group() -> None:
    """Tests if NotComputableError is raised when only one subgroup is present."""
    metric = DemographicParityDifference()
    y_pred = torch.tensor([[0.9, 0.1], [0.8, 0.2]])
    y = torch.tensor([0, 0])
    group_labels = torch.tensor([0, 0])
    metric.update((y_pred, y, group_labels))
    with pytest.raises(NotComputableError, match="Fairness metrics require at least two unique subgroups"):
        metric.compute()


def test_demographic_parity_difference_binary_probs_shape_B() -> None:
    """Tests DemographicParityDifference with shape (B,) thresholded inputs."""
    metric = DemographicParityDifference()
    # y_pred is (B,) already thresholded
    # Group 0: 1 pos / 2 total = 0.5
    # Group 1: 0 pos / 2 total = 0.0
    y_pred = torch.tensor([1, 0, 0, 0])
    y = torch.tensor([0, 0, 0, 0])  # ignored
    groups = torch.tensor([0, 0, 1, 1])

    metric.update((y_pred, y, groups))
    assert_close(metric.compute(), 0.5)


def test_demographic_parity_difference_binary_probs_shape_B_1() -> None:
    """Tests DemographicParityDifference with shape (B, 1) thresholded inputs."""
    metric = DemographicParityDifference()
    # y_pred is (B, 1) already thresholded
    y_pred = torch.tensor([[1], [0], [1], [0]])
    y = torch.tensor([0, 0, 0, 0])
    groups = torch.tensor([0, 0, 1, 1])
    metric.update((y_pred, y, groups))
    # G0 selection rate: 0.5, G1 selection rate: 0.5 -> Diff 0.0
    assert_close(metric.compute(), 0.0)


def test_demographic_parity_difference_multiclass() -> None:
    """Tests DemographicParityDifference with multiclass logits."""
    metric = DemographicParityDifference()
    # y_pred is (B, C)
    # G0 selection rates: [0.5, 0.5, 0.0]
    # G1 selection rates: [0.5, 0.0, 0.5]
    y_pred = torch.tensor(
        [
            [0.8, 0.1, 0.1],  # pred class 0
            [0.1, 0.8, 0.1],  # pred class 1
            [0.8, 0.1, 0.1],  # pred class 0
            [0.1, 0.1, 0.8],  # pred class 2
        ]
    )
    y = torch.tensor([0, 0, 0, 0])
    groups = torch.tensor([0, 0, 1, 1])

    metric.update((y_pred, y, groups))
    # Disparities: Class 0: 0.0, Class 1: 0.5, Class 2: 0.5
    assert_close(metric.compute(), 0.5)


def test_demographic_parity_difference_multilabel() -> None:
    """Tests DemographicParityDifference with multilabel data."""
    metric = DemographicParityDifference(is_multilabel=True)
    # y_pred is (B, C) indicators
    # G0: [1, 1, 0], [0, 0, 0] -> rates: [0.5, 0.5, 0.0]
    # G1: [1, 1, 1], [0, 1, 0] -> rates: [0.5, 1.0, 0.5]
    y_pred = torch.tensor([[1, 1, 0], [0, 0, 0], [1, 1, 1], [0, 1, 0]])
    y = torch.tensor([[0, 0, 0]] * 4)
    groups = torch.tensor([0, 0, 1, 1])

    metric.update((y_pred, y, groups))
    # Disparities: C0: 0.0, C1: 0.5, C2: 0.5
    assert_close(metric.compute(), 0.5)


def _test_distrib_integration(device: torch.device) -> None:
    """Helper to test distributed integration."""
    rank = idist.get_rank()
    metric = DemographicParityDifference(device=device)
    y = torch.tensor([0, 0], device=device)
    groups = torch.tensor([0, 1], device=device)

    if rank == 0:
        # G0: pos, G1: neg
        y_pred = torch.tensor([[0.2, 0.8], [0.8, 0.2]], device=device)
    else:
        # G0: pos, G1: neg
        y_pred = torch.tensor([[0.1, 0.9], [0.9, 0.1]], device=device)

    metric.update((y_pred, y, groups))
    res = metric.compute()
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
