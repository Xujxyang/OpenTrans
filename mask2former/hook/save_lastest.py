from typing import Optional
from fvcore.common.checkpoint import Checkpointer

from detectron2.engine.train_loop import HookBase


class PeriodicCheckpointerLastest(HookBase):
    """
    Same as :class:`detectron2.checkpoint.PeriodicCheckpointer`, but as a hook.

    Note that when used as a hook,
    it is unable to save additional data other than what's defined
    by the given `checkpointer`.

    It is executed every ``period`` iterations and after the last iteration.
    """
    def __init__(
        self,
        checkpointer: Checkpointer,
        period: int,
        max_iter: Optional[int] = None,
        max_to_keep: Optional[int] = None,
        file_prefix: str = "model",
    ) -> None:
        """
        Args:
            checkpointer: the checkpointer object used to save checkpoints.
            period (int): the period to save checkpoint.
            max_iter (int): maximum number of iterations. When it is reached,
                a checkpoint named "{file_prefix}_final" will be saved.
            max_to_keep (int): maximum number of most current checkpoints to keep,
                previous checkpoints will be deleted
            file_prefix (str): the prefix of checkpoint's filename
        """
        self.checkpointer = checkpointer
        self.period = int(period)
        self.max_iter = max_iter
        # if max_to_keep is not None:
        #     assert max_to_keep > 0
        # self.max_to_keep = max_to_keep
        # self.recent_checkpoints = []
        self.path_manager = checkpointer.path_manager
        self.file_prefix = file_prefix
        self.file_name = "{}_lastest".format(self.file_prefix)

    def step(self, iteration: int, **kwargs) -> None:
        """
        Perform the appropriate action at the given iteration.
        Args:
            iteration (int): the current iteration, ranged in [0, max_iter-1].
            kwargs (Any): extra data to save, same as in
                :meth:`Checkpointer.save`.
        """
        iteration = int(iteration)
        additional_state = {"iteration": iteration}
        additional_state.update(kwargs)

        if ((iteration + 1) % self.period == 0) or (self.max_iter and (iteration >= self.max_iter - 1)):
            if self.path_manager.exists(self.file_name):
                self.path_manager.rm(self.file_name)

            self.checkpointer.save(
                self.file_name, **additional_state
            )

    def save(self, name: str, **kwargs) -> None:
        """
        Same argument as :meth:`Checkpointer.save`.
        Use this method to manually save checkpoints outside the schedule.
        Args:
            name (str): file name.
            kwargs (Any): extra data to save, same as in
                :meth:`Checkpointer.save`.
        """
        self.checkpointer.save(name, **kwargs)

    def before_train(self):
        self.max_iter = self.trainer.max_iter

    def after_step(self):
        # No way to use **kwargs
        self.step(self.trainer.iter)