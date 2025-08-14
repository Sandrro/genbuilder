from torch.utils.tensorboard import SummaryWriter

_writer = None


def configure(logdir: str, flush_secs: int = 5) -> None:
    """Initialise a global SummaryWriter.

    Parameters
    ----------
    logdir: str
        Directory where TensorBoard logs will be written.
    flush_secs: int
        How often, in seconds, to flush pending events to disk.
    """
    global _writer
    _writer = SummaryWriter(log_dir=logdir, flush_secs=flush_secs)


def log_value(tag: str, value, step: int) -> None:
    """Log a scalar variable to TensorBoard.

    Does nothing if :func:`configure` has not been called.
    """
    if _writer is not None:
        _writer.add_scalar(tag, value, step)

