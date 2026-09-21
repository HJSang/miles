"""Checkpoint directories: written collectively, complete at their final path."""

import json
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

import torch.distributed as dist

from miles.utils.distributed_phase import DistributedPhaseGate


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
    phase_gate = DistributedPhaseGate()

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

    phase_gate.run_local_phase("checkpoint.prepare", make_tmp_dir)
    phase_gate.wait_for_all()
    phase_gate.run_local_phase("checkpoint.write_shards", lambda: write_shards(tmp_dir))
    phase_gate.wait_for_all()
    phase_gate.run_local_phase("checkpoint.publish", publish_dir)
    phase_gate.wait_for_all()


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0
