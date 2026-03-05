import torch
import pytest
from ignite.metrics.fairness import SubgroupAccuracyDifference, DemographicParityDifference


def test_subgroup_accuracy_difference_validation() -> None:
    """Tests input validation for SubgroupAccuracyDifference."""
    metric = SubgroupAccuracyDifference(groups=[0, 1])

    # Valid multiclass logits
    y_pred = torch.randn(4, 3)
    y = torch.randint(0, 3, (4,))
    groups = torch.tensor([0, 0, 1, 1])
    metric.update((y_pred, y, groups))
    assert metric.compute() >= 0

    metric.reset()

    # Valid binary classes (0/1)
    y_pred = torch.tensor([0, 1, 0, 1])
    y = torch.tensor([0, 1, 1, 0])
    groups = torch.tensor([0, 0, 1, 1])
    metric.update((y_pred, y, groups))
    assert metric.compute() == 1.0  # Subgroup 0: 100%, Subgroup 1: 0%

    metric.reset()

    # Invalid binary probabilities (not 0 or 1)
    y_pred = torch.tensor([0.6, 0.4, 0.8, 0.1])
    y = torch.tensor([1, 0, 1, 0])
    groups = torch.tensor([0, 0, 1, 1])
    with pytest.raises(ValueError, match="y_pred must be comprised of 0's and 1's"):
        metric.update((y_pred, y, groups))


def test_demographic_parity_difference_validation() -> None:
    """Tests input validation for DemographicParityDifference."""
    metric = DemographicParityDifference(groups=[0, 1])

    # Valid multiclass logits
    y_pred = torch.tensor(
        [
            [0.1, 0.9, 0.0],
            [0.1, 0.9, 0.0],  # Group 0: all class 1
            [0.9, 0.1, 0.0],
            [0.9, 0.1, 0.0],  # Group 1: all class 0
        ]
    )
    y = torch.tensor([1, 1, 0, 0])
    groups = torch.tensor([0, 0, 1, 1])
    metric.update((y_pred, y, groups))
    # Class 0: Group 0 rate = 0, Group 1 rate = 1.0 -> Diff = 1.0
    # Class 1: Group 0 rate = 1.0, Group 1 rate = 0 -> Diff = 1.0
    assert metric.compute() == 1.0

    metric.reset()

    # Valid binary classes (0/1)
    y_pred = torch.tensor([1, 1, 0, 0])
    y = torch.tensor([1, 1, 0, 0])
    groups = torch.tensor([0, 0, 1, 1])
    metric.update((y_pred, y, groups))
    assert metric.compute() == 1.0

    metric.reset()

    # Invalid binary probabilities
    y_pred = torch.tensor([0.6, 0.4, 0.8, 0.1])
    y = torch.tensor([1, 0, 1, 0])
    groups = torch.tensor([0, 0, 1, 1])
    with pytest.raises(ValueError, match="y_pred must be comprised of 0's and 1's"):
        metric.update((y_pred, y, groups))


def test_multilabel_validation() -> None:
    """Tests input validation for multilabel data."""
    metric = SubgroupAccuracyDifference(groups=[0, 1], is_multilabel=True)

    # Valid multilabel (0/1)
    y_pred = torch.tensor([[1, 0], [1, 1], [0, 0], [0, 1]])
    y = torch.tensor([[1, 0], [1, 1], [1, 1], [0, 0]])
    groups = torch.tensor([0, 0, 1, 1])
    metric.update((y_pred, y, groups))
    # Accuracy uses sample-wise correctness: all labels must match per sample.
    # Group 0 (samples 0,1): [1,0]==[1,0] ✓, [1,1]==[1,1] ✓ -> Accuracy = 2/2 = 1.0
    # Group 1 (samples 2,3): [0,0]!=[1,1] ✗, [0,1]!=[0,0] ✗ -> Accuracy = 0/2 = 0.0
    # Disparity = 1.0 - 0.0 = 1.0
    assert metric.compute() == 1.0

    metric.reset()

    # Invalid multilabel (not 0/1)
    y_pred = torch.tensor([[0.6, 0.4], [0.8, 0.1]])
    y = torch.tensor([[1, 0], [1, 1]])
    groups = torch.tensor([0, 1])
    with pytest.raises(ValueError, match="y_pred must be comprised of 0's and 1's"):
        metric.update((y_pred, y, groups))


def test_shape_mismatch_validation() -> None:
    """Tests validation for shape mismatches between y and y_pred."""
    metric = SubgroupAccuracyDifference(groups=[0, 1])

    # y is (B,), y_pred is (B, C) - OK
    metric.update((torch.randn(4, 3), torch.randint(0, 3, (4,)), torch.zeros(4)))

    # y is (B,), y_pred is (B, C, H, W) - Error (dimension mismatch)
    with pytest.raises(ValueError, match="y must have shape of"):
        metric.update((torch.randn(4, 3, 2, 2), torch.randint(0, 3, (4,)), torch.zeros(4)))


def test_batch_size_mismatch_validation() -> None:
    """Tests validation for batch size mismatches across y_pred, y, and groups."""
    metric = SubgroupAccuracyDifference(groups=[0, 1])

    # y_pred (4), y (3) - Batch size mismatch
    with pytest.raises(ValueError, match="y_pred, y, and group_labels must have the same batch size"):
        metric.update((torch.randn(4, 3), torch.randint(0, 3, (3,)), torch.zeros(4)))

    # groups (3), y_pred (4) - Batch size mismatch
    with pytest.raises(ValueError, match="y_pred, y, and group_labels must have the same batch size"):
        metric.update((torch.randn(4, 3), torch.randint(0, 3, (4,)), torch.zeros(3)))


def test_type_switch_validation() -> None:
    """Tests if RuntimeError is raised when input type changes mid-epoch."""
    metric = SubgroupAccuracyDifference(groups=[0, 1])

    # First batch: binary
    metric.update((torch.tensor([1, 0]), torch.tensor([1, 0]), torch.tensor([0, 1])))

    # Second batch: multiclass - Should raise RuntimeError
    with pytest.raises(RuntimeError, match="Input data type has changed from binary to multiclass"):
        metric.update((torch.randn(2, 3), torch.tensor([0, 1]), torch.tensor([0, 1])))

    # Third batch: multiclass (3) then multiclass (5) - Should raise ValueError
    metric.reset()
    metric.update((torch.randn(4, 3), torch.randint(0, 3, (4,)), torch.zeros(4)))
    with pytest.raises(ValueError, match="Input data number of classes has changed"):
        metric.update((torch.randn(4, 5), torch.randint(0, 5, (4,)), torch.zeros(4)))
