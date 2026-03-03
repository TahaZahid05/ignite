import torch
from abc import abstractmethod
from typing import Callable, cast

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
        output_transform: Callable = lambda x: x,
        device: str | torch.device = torch.device("cpu"),
        requires_y: bool = True,
    ) -> None:
        self._requires_y = requires_y
        self._type: str | None = None
        self._num_classes: int | None = None
        super().__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self) -> None:
        """Resets the metric state."""
        self._y_preds: list[torch.Tensor] = []
        self._ys: list[torch.Tensor] = []
        self._groups: list[torch.Tensor] = []
        self._type = None
        self._num_classes = None

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

        self._y_preds.append(y_pred.cpu())
        if self._requires_y:
            self._ys.append(y.cpu())
        self._groups.append(group_labels.view(-1).cpu())

    @abstractmethod
    def compute_metric_for_group(self, y_pred: torch.Tensor, y: torch.Tensor) -> float | torch.Tensor:
        """Computes the core metric (e.g., accuracy, selection rate) for a single specific subgroup.

        Subclasses must implement this method.

        Args:
            y_pred: Predictions for the specific subgroup.
            y: Targets for the specific subgroup.

        Returns:
            The computed metric value for the subgroup.
        """

    def _compute_group_disparities(self, y_preds: torch.Tensor, ys: torch.Tensor, groups: torch.Tensor) -> float:
        """Helper to break the consolidated tensors into group-specific calculations.

        Args:
            y_preds: predictions for all groups.
            ys: targets for all groups.
            groups: group labels for all predictions and targets.

        Returns:
            The maximum difference in metric value between any two subgroups.

        Raises:
            NotComputableError: if less than two unique subgroups have been processed.
        """
        unique_groups = torch.unique(groups)

        if unique_groups.numel() < 2:
            raise NotComputableError("Fairness metrics require at least two unique subgroups to compute a disparity.")

        group_metrics = []
        for g in unique_groups:
            mask = groups == g
            group_y_pred = y_preds[mask]
            group_y = ys[mask] if self._requires_y else ys

            val = self.compute_metric_for_group(group_y_pred, group_y)
            group_metrics.append(val)

        diff = max(group_metrics) - min(group_metrics)
        if isinstance(diff, torch.Tensor):
            return float(diff.item())
        return float(diff)

    def compute(self) -> float:
        """Computes the maximum disparity of the metric between any two subgroups.

        Returns:
            The maximum difference.

        Raises:
            NotComputableError: if the metric has not seen any data.
        """
        if len(self._y_preds) == 0:
            raise NotComputableError("Fairness metrics must have at least one example before it can be computed.")

        # Concatenate local data
        all_y_preds = torch.cat(self._y_preds)
        all_ys = torch.cat(self._ys) if self._requires_y else torch.empty(0)
        all_groups = torch.cat(self._groups)

        ws = idist.get_world_size()
        if ws > 1:
            # Sync all data across ranks. We gather tensors from all ranks.
            all_y_preds = cast(torch.Tensor, idist.all_gather(all_y_preds))
            all_groups = cast(torch.Tensor, idist.all_gather(all_groups))
            if self._requires_y:
                all_ys = cast(torch.Tensor, idist.all_gather(all_ys))

        # Move back to target device for the final computation
        all_y_preds = all_y_preds.to(self._device)
        all_ys = all_ys.to(self._device) if self._requires_y else all_ys.to(self._device)
        all_groups = all_groups.to(self._device)

        return self._compute_group_disparities(all_y_preds, all_ys, all_groups)
