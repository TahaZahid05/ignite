import torch
from abc import abstractmethod
from typing import Callable, Sequence, cast

import ignite.distributed as idist
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric, reinit__is_reduced


class _BaseFairness(Metric):
    """Base class for algorithmic fairness and subgroup bias metrics.

    This class handles grouping predictions and targets by their
    associated subgroup labels, allowing subclasses to easily compute disparity metrics
    across those groups.

    Args:
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s `process_function`'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
            By default, metrics require the output as `(y_pred, y, group_labels)` or
            `{'y_pred': y_pred, 'y': y, 'group_labels': group_labels}`.
        device: specifies which device updates are accumulated on. Setting the
            metric's device to be the same as your `update` arguments ensures the `update` method is
            non-blocking. By default, CPU.

    .. versionadded:: 0.6.0
    """

    required_output_keys = ("y_pred", "y", "group_labels")

    def __init__(
        self,
        groups: Sequence[int],
        output_transform: Callable = lambda x: x,
        device: str | torch.device = torch.device("cpu"),
        requires_y: bool = True,
    ) -> None:
        self._user_groups = list(groups)
        self._requires_y = requires_y
        self._type: str | None = None
        self._num_classes: int | None = None
        super().__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self) -> None:
        """Resets the metric state."""
        self._type = None
        self._num_classes = None
        self._updated = False
        self._group_numerator: dict[int, torch.Tensor] = {}
        self._group_denominator: dict[int, int] = {g: 0 for g in self._user_groups}

    def _check_shape(self, y_pred: torch.Tensor, y: torch.Tensor) -> None:
        """Internal method to check the compatibility of y_pred and y shapes.

        Args:
            y_pred: predictions tensor.
            y: targets tensor.
        """
        if not (y.ndimension() == y_pred.ndimension() or y.ndimension() + 1 == y_pred.ndimension()):
            raise ValueError(
                "y must have shape of (batch_size, ...) and y_pred must have "
                "shape of (batch_size, num_categories, ...) or (batch_size, ...), "
                f"but given {y.shape} vs {y_pred.shape}."
            )

        y_shape = y.shape
        y_pred_shape: tuple[int, ...] = y_pred.shape

        if y.ndimension() + 1 == y_pred.ndimension():
            y_pred_shape = (y_pred_shape[0],) + y_pred_shape[2:]

        if not (y_shape == y_pred_shape):
            raise ValueError("y and y_pred must have compatible shapes.")

        if (
            hasattr(self, "_is_multilabel")
            and getattr(self, "_is_multilabel")
            and not (y.shape == y_pred.shape and y.ndimension() > 1 and y.shape[1] > 1)
        ):
            raise ValueError(
                "y and y_pred must have same shape of (batch_size, num_categories, ...) and num_categories > 1."
            )

    def _check_binary_multilabel_cases(self, y_pred: torch.Tensor, y: torch.Tensor) -> None:
        """Internal method to check if y_pred and y items are binary (0 and 1).

        Args:
            y_pred: predictions tensor.
            y: targets tensor.
        """
        if not torch.equal(y, y**2):
            raise ValueError("For binary cases, y must be comprised of 0's and 1's.")

        if not torch.equal(y_pred, y_pred**2):
            raise ValueError("For binary/multilabel cases, y_pred must be comprised of 0's and 1's.")

    def _check_type(self, y_pred: torch.Tensor, y: torch.Tensor) -> str:
        """Internal method to check the input data type (binary, multiclass, multilabel).

        Args:
            y_pred: predictions tensor.
            y: targets tensor.

        Returns:
            The identified data type.
        """
        if y.ndimension() + 1 == y_pred.ndimension():
            num_classes = y_pred.shape[1]
            if num_classes == 1:
                update_type = "binary"
                self._check_binary_multilabel_cases(y_pred, y)
            else:
                update_type = "multiclass"
        elif y.ndimension() == y_pred.ndimension():
            self._check_binary_multilabel_cases(y_pred, y)

            if hasattr(self, "_is_multilabel") and getattr(self, "_is_multilabel"):
                update_type = "multilabel"
                num_classes = y_pred.shape[1]
            else:
                update_type = "binary"
                num_classes = 1
        else:
            raise RuntimeError(
                f"Invalid shapes of y (shape={y.shape}) and y_pred (shape={y_pred.shape}), check documentation."
                " for expected shapes of y and y_pred."
            )

        if self._type is None:
            self._type = update_type
            self._num_classes = num_classes
        else:
            if self._type != update_type:
                raise RuntimeError(f"Input data type has changed from {self._type} to {update_type}.")
            if self._num_classes != num_classes:
                raise ValueError(f"Input data number of classes has changed from {self._num_classes} to {num_classes}")

        return update_type

    def update(self, output: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> None:
        """Updates the metric state with new data.

        Args:
            output: a tuple containing `(y_pred, y, group_labels)`.
        """
        y_pred, y, group_labels = output[0].detach(), output[1].detach(), output[2].detach()

        if y_pred.shape[0] != y.shape[0] or y_pred.shape[0] != group_labels.shape[0]:
            raise ValueError("y_pred, y, and group_labels must have the same batch size.")

        self._check_shape(y_pred, y)
        self._check_type(y_pred, y)
        self._updated = True

        for g in self._user_groups:
            mask = group_labels == g
            if mask.any():
                group_y_pred = y_pred[mask]
                group_y = y[mask] if self._requires_y else y
                self._update_group(g, group_y_pred, group_y)

    @abstractmethod
    def _update_group(self, group: int, y_pred: torch.Tensor, y: torch.Tensor) -> None:
        """Update ``self._group_numerator[group]`` and ``self._group_denominator[group]``.

        Subclasses must implement this method to define how per-group running
        accumulators are updated from a batch of data.

        Args:
            group: the group label.
            y_pred: predictions for this group in the current batch.
            y: targets for this group in the current batch.
        """

    def _compute_group_metric(self, group: int) -> float | torch.Tensor:
        """Computes the metric for a group as ``numerator / denominator``.

        Args:
            group: the group label.

        Returns:
            The computed metric value for the group.
        """
        if self._group_denominator[group] == 0 or group not in self._group_numerator:
            for other_g in self._user_groups:
                if other_g in self._group_numerator:
                    return torch.zeros_like(self._group_numerator[other_g])
            return torch.tensor(0.0)
        return self._group_numerator[group] / self._group_denominator[group]

    def _sync_group_state(self) -> None:
        """Syncs per-group accumulated state across distributed ranks via all-reduce."""
        shape = None
        for g in self._user_groups:
            if g in self._group_numerator:
                shape = self._group_numerator[g].shape
                break
        if shape is None:
            return
        for g in self._user_groups:
            if g not in self._group_numerator:
                self._group_numerator[g] = torch.zeros(shape, device=self._device)
        nums = torch.stack([self._group_numerator[g] for g in self._user_groups])
        dens = torch.tensor(
            [self._group_denominator[g] for g in self._user_groups], device=self._device, dtype=torch.float
        )
        nums = cast(torch.Tensor, idist.all_reduce(nums))
        dens = cast(torch.Tensor, idist.all_reduce(dens))
        for i, g in enumerate(self._user_groups):
            self._group_numerator[g] = nums[i]
            self._group_denominator[g] = int(dens[i].item())

    def compute(self) -> float:
        """Computes the maximum disparity of the metric between any two subgroups.

        Returns:
            The maximum difference in metric value between any two subgroups.

        Raises:
            NotComputableError: if less than two unique subgroups have been processed.
        """
        if not self._updated:
            raise NotComputableError("Fairness metrics must have at least one example before it can be computed.")

        if len(self._user_groups) < 2:
            raise NotComputableError("Fairness metrics require at least two unique subgroups to compute a disparity.")

        ws = idist.get_world_size()
        if ws > 1:
            self._sync_group_state()

        group_metrics = []
        for g in self._user_groups:
            val = self._compute_group_metric(g)
            if isinstance(val, (int, float)):
                val = torch.tensor(val)
            group_metrics.append(val)

        stacked = torch.stack(group_metrics)
        disparities = stacked.max(dim=0).values - stacked.min(dim=0).values
        return float(disparities.max().item())
