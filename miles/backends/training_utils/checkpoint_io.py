"""Checkpoint directories: written collectively, complete at their final path."""

import json
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

import torch.distributed as dist

from miles.utils.distributed_utils import get_gloo_group


class DistributedCheckpointError(RuntimeError):
    """A checkpoint phase failed on one or more ranks."""

    def __init__(self, phase: str, failures: list[tuple[int, str]]) -> None:
        details = "; ".join(f"rank {rank}: {message}" for rank, message in failures)
        super().__init__(f"checkpoint phase {phase!r} failed ({details})")
        self.phase = phase
        self.failures = failures


def write_checkpoint_dir(
    path: str | Path,
    write_shards: Callable[[Path], None],
    metadata: dict | None = None,
    *,
    overwrite: bool = True,
) -> None:
    """Write collectively, then atomically point ``path`` at the completed version.

    All ranks must call. Readers may still hold an older version, so retain it.
    """
    final_dir = Path(path)
    tmp_dir = final_dir.parent / f"_tmp_{final_dir.name}"

    def make_tmp_dir():
        if _rank() == 0:
            if not overwrite and final_dir.exists():
                raise FileExistsError(f"checkpoint {final_dir} already exists")
            if final_dir.exists() and not final_dir.is_symlink():
                raise NotImplementedError(
                    f"cannot overwrite a legacy checkpoint directory {final_dir}; save under a new name"
                )
            # a crashed attempt may leave shards or an unpublished version link
            if tmp_dir.is_symlink():
                tmp_dir.unlink()
            elif tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            tmp_dir.mkdir(parents=True)

    def publish_dir():
        if _rank() != 0:
            return
        if metadata is not None:
            (tmp_dir / "META.json").write_text(json.dumps(metadata, indent=2))
        version_dir = final_dir.parent / f"_version_{final_dir.name}_{uuid4().hex}"
        os.replace(tmp_dir, version_dir)
        tmp_dir.symlink_to(version_dir.name, target_is_directory=True)
        os.replace(tmp_dir, final_dir)

    _run_phase("prepare", make_tmp_dir)
    _barrier()
    _run_phase("write_shards", lambda: write_shards(tmp_dir))
    _barrier()
    _run_phase("publish", publish_dir)
    _barrier()


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _barrier() -> None:
    if dist.is_initialized():
        dist.barrier(group=get_gloo_group())


def _run_phase(phase: str, operation: Callable[[], None]) -> None:
    """Run local checkpoint work and share failures before the next barrier."""
    local_error: Exception | None = None
    try:
        operation()
    except Exception as error:
        local_error = error

    if not dist.is_initialized():
        if local_error is not None:
            raise local_error
        return

    local_message = None
    if local_error is not None:
        local_message = f"{type(local_error).__name__}: {local_error}"
    group = get_gloo_group()
    messages: list[str | None] = [None] * dist.get_world_size(group=group)
    dist.all_gather_object(messages, local_message, group=group)
    failures = [(rank, message) for rank, message in enumerate(messages) if message is not None]
    if failures:
        error = DistributedCheckpointError(phase, failures)
        if local_error is not None:
            raise error from local_error
        raise error
