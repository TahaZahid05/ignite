import torch
from typing import Callable

from ignite.exceptions import NotComputableError
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

            metric = DemographicParityDifference()
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
        is_multilabel: bool = False,
        output_transform: Callable = lambda x: x,
        device: torch.device | str = torch.device("cpu"),
    ) -> None:
        self._is_multilabel = is_multilabel
        super().__init__(output_transform=output_transform, device=device, requires_y=False)

    def compute_metric_for_group(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes the selection rate (predicted positives) for a specific subgroup.

        Args:
            y_pred: predictions for the specific subgroup.
            y: targets for the specific subgroup. Note: The targets `y` are ignored for Demographic Parity.

        Returns:
            The computed selection rate(s) for the subgroup. Returns a tensor for all cases.
        """
        if self._type == "binary":
            total = y_pred.numel()
            if total > 0:
                # Expect thresholded input for binary
                positives = torch.bincount(y_pred.view(-1).to(torch.long), minlength=2)
                return positives.float() / total
            return torch.zeros(2, dtype=torch.float32, device=y_pred.device)

        elif self._type == "multiclass":
            if self._num_classes is None:
                raise RuntimeError("num_classes must be set for multiclass data.")
            num_classes: int = self._num_classes
            predicted_classes = torch.argmax(y_pred, dim=1)
            total = predicted_classes.numel()
            if total > 0:
                positives = torch.bincount(predicted_classes.view(-1), minlength=num_classes)
                return positives.float() / total
            return torch.zeros(num_classes, dtype=torch.float32, device=y_pred.device)

        elif self._type == "multilabel":
            # Multilabel: Compute selection rate for each class independently.
            if self._num_classes is None:
                raise RuntimeError("num_classes must be set for multilabel data.")
            num_classes: int = self._num_classes

            # Total examples = Total elements / Number of classes
            total_examples = int(y_pred.numel() / num_classes)

            if total_examples > 0:
                # Sum occurrences of 1s for each class across all other dimensions (Batch, H, W, etc.)
                # We do this by moving class dim to end, flattening others, then summing.
                positives_per_class = y_pred.movedim(1, -1).reshape(-1, num_classes).sum(dim=0)
                return positives_per_class.float() / total_examples
            return torch.zeros(num_classes, dtype=torch.float32, device=y_pred.device)

        else:
            raise ValueError(f"Unexpected type: {self._type}")

    def _compute_group_disparities(self, y_preds: torch.Tensor, ys: torch.Tensor, groups: torch.Tensor) -> float:
        """Computes the core disparity logic across groups.

        This method calculates the per-class selection rate for each subgroup,
        then finds the maximum disparity (max selection rate - min selection rate)
        across all classes and subgroups.

        Args:
            y_preds: predictions for all groups.
            ys: targets for all groups. Note: targets are ignored.
            groups: group labels for all predictions.

        Returns:
            The maximum difference in selection rate across all subgroups and classes.

        Raises:
            NotComputableError: if less than two unique subgroups have been processed.
        """
        unique_groups = torch.unique(groups)

        if unique_groups.numel() < 2:
            raise NotComputableError("Fairness metrics require at least two unique subgroups to compute a disparity.")

        group_rates: list[torch.Tensor] = []

        for g in unique_groups:
            mask = groups == g
            # ys is empty since requires_y=False, so we pass it as is to avoid mask shape mismatch
            rate = self.compute_metric_for_group(y_preds[mask], ys)
            group_rates.append(rate)

        group_rates_tensor = torch.stack(group_rates)

        max_rates = group_rates_tensor.max(dim=0).values
        min_rates = group_rates_tensor.min(dim=0).values
        disparities = max_rates - min_rates

        return float(disparities.max().item())
