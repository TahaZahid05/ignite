import torch
from typing import Callable
from ignite.metrics.fairness.base import _BaseFairness

__all__ = ["SubgroupAccuracyDifference"]


class SubgroupAccuracyDifference(_BaseFairness):
    r"""Calculates the Subgroup Accuracy Difference.

    This metric computes the accuracy for each unique subgroup in the dataset and returns
    the maximum difference in accuracy between any two subgroups. It is a strict measure
    of how disparate the performance of a model is across different categorical segments.

    - ``update`` must receive output of the form ``(y_pred, y, group_labels)`` or
      ``{'y_pred': y_pred, 'y': y, 'group_labels': group_labels}``.
    - `y_pred` must be in the following shape (batch_size, num_categories, ...) or (batch_size, ...).
    - `y` must be in the following shape (batch_size, ...).
    - `group_labels` must be a 1D tensor of shape (batch_size,) containing discrete labels.

    Args:
        is_multilabel: if True, multilabel accuracy is calculated. By default, False.
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric.
        device: specifies which device updates are accumulated on. Setting the
            metric's device to be the same as your ``update`` arguments ensures the ``update`` method is
            non-blocking. By default, CPU.

    Examples:
        To use with ``Engine`` and ``process_function``, simply attach the metric instance to the engine.
        The output of the engine's ``process_function`` needs to be in the format of
        ``(y_pred, y, group_labels)``.

        .. include:: defaults.rst
            :start-after: :orphan:

        .. testcode::

            metric = SubgroupAccuracyDifference()
            metric.attach(default_evaluator, 'subgroup_acc_diff')

            # Predictions for 4 items:
            # Items 1 and 3 are predicted as class 0 (index 0 has highest prob)
            # Items 2 and 4 are predicted as class 1 (index 1 has highest prob)
            y_pred = torch.tensor([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])

            # Targets
            y_true = torch.tensor([0, 1, 1, 0])

            # Subgroups (e.g., 0=Demographic A, 1=Demographic B)
            group_labels = torch.tensor([0, 0, 1, 1])

            # Subgroup 0: 2 correct predictions, accuracy = 100%
            # Subgroup 1: 0 correct predictions, accuracy = 0%

            state = default_evaluator.run([[y_pred, y_true, group_labels]])
            print(state.metrics['subgroup_acc_diff'])

        .. testoutput::

            1.0

    .. versionadded:: 0.6.0
    """

    def __init__(
        self,
        is_multilabel: bool = False,
        output_transform: Callable = lambda x: x,
        device: torch.device | str = torch.device("cpu"),
    ) -> None:
        self._is_multilabel = is_multilabel
        super().__init__(output_transform=output_transform, device=device)

    def compute_metric_for_group(self, y_pred: torch.Tensor, y: torch.Tensor) -> float:
        """Computes the accuracy for a specific subgroup.

        Args:
            y_pred: predictions for the specific subgroup.
            y: targets for the specific subgroup.

        Returns:
            The computed accuracy for the subgroup.
        """
        if self._type == "binary":
            correct = torch.eq(y_pred.view(-1).to(y), y.view(-1))
        elif self._type == "multiclass":
            indices = torch.argmax(y_pred, dim=1)
            correct = torch.eq(indices, y).view(-1)
        elif self._type == "multilabel":
            # Multilabel: Each label for each example is treated as a separate task to provide
            # a granular view of disparities across individual classification boundaries.
            correct = torch.eq(y_pred.type_as(y), y)
        else:
            raise ValueError(f"Unexpected type: {self._type}")

        correct_sum = torch.sum(correct).item()
        total = correct.numel()

        return float(correct_sum) / total if total > 0 else 0.0
