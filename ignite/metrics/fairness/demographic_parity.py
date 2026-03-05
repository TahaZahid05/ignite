import torch
from typing import Callable, Sequence

from ignite.metrics.fairness.base import _BaseFairness

__all__ = ["DemographicParityDifference"]


class DemographicParityDifference(_BaseFairness):
    r"""Calculates the Demographic Parity Difference.

    This metric computes the selection rate (the rate of positive predictions) for each unique
    subgroup in the dataset and returns the maximum difference in selection rates between any
    two subgroups.

    A lower value indicates that the model predicts the positive outcome at roughly the same rate
    across all subgroups, a standard definition of fairness. This metric is referred to as
    *Group Fairness / Statistical Parity* in the fairness literature
    (`Verma & Rubin, 2018 <https://fairware.cs.umass.edu/papers/Verma.pdf>`_).

    - ``update`` must receive output of the form ``(y_pred, y, group_labels)`` or
      ``{'y_pred': y_pred, 'y': y, 'group_labels': group_labels}``.
    - `y_pred` must be in the following shape (batch_size, num_categories, ...) or (batch_size, ...).
    - `y` must be in the following shape (batch_size, ...).
    - `group_labels` must be a 1D tensor of shape (batch_size,) containing discrete labels.

    Args:
        is_multilabel: if True, multilabel selection rate is calculated. By default, False.
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

            metric = DemographicParityDifference(groups=[0, 1])
            metric.attach(default_evaluator, 'demographic_parity_diff')

            # Predictions for 4 items:
            # Items 1 and 2 are predicted as class 1 (index 1 has highest prob)
            # Items 3 and 4 are predicted as class 0 (index 0 has highest prob)
            y_pred = torch.tensor([[0.1, 0.9], [0.2, 0.8], [0.9, 0.1], [0.9, 0.1]])

            # Targets (Not actually used for parity, but required by API)
            y_true = torch.tensor([1, 1, 0, 0])

            # Subgroups
            group_labels = torch.tensor([0, 0, 1, 1])

            # Subgroup 0: 2 positive predictions / 2 total = 1.0 selection rate
            # Subgroup 1: 0 positive predictions / 2 total = 0.0 selection rate

            state = default_evaluator.run([[y_pred, y_true, group_labels]])
            print(state.metrics['demographic_parity_diff'])

        .. testoutput::

            1.0

    .. versionadded:: 0.6.0
    """

    def __init__(
        self,
        groups: Sequence[int],
        is_multilabel: bool = False,
        output_transform: Callable = lambda x: x,
        device: torch.device | str = torch.device("cpu"),
    ) -> None:
        self._is_multilabel = is_multilabel
        super().__init__(groups=groups, output_transform=output_transform, device=device, requires_y=False)

    def _update_group(self, group: int, y_pred: torch.Tensor, y: torch.Tensor) -> None:
        """Updates per-group selection rate accumulators."""
        if self._type == "binary":
            positives = torch.bincount(y_pred.view(-1).to(torch.long), minlength=2).float()
            total = y_pred.numel()

        elif self._type == "multiclass":
            if self._num_classes is None:
                raise RuntimeError("num_classes must be set for multiclass data.")
            predicted_classes = torch.argmax(y_pred, dim=1)
            positives = torch.bincount(predicted_classes.view(-1), minlength=self._num_classes).float()
            total = predicted_classes.numel()

        elif self._type == "multilabel":
            if self._num_classes is None:
                raise RuntimeError("num_classes must be set for multilabel data.")
            num_classes: int = self._num_classes
            positives = y_pred.movedim(1, -1).reshape(-1, num_classes).sum(dim=0).float()
            total = int(y_pred.numel() / num_classes)

        else:
            raise ValueError(f"Unexpected type: {self._type}")

        if group not in self._group_numerator:
            self._group_numerator[group] = torch.zeros_like(positives, device=self._device)
        self._group_numerator[group] += positives.to(self._device)
        self._group_denominator[group] += total
