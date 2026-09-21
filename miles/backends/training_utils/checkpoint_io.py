"""Compatibility wrapper for the shared artifact directory writer."""

from collections.abc import Callable
from pathlib import Path

from miles.backends.training_utils.artifact_io import ArtifactStore


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
    store = ArtifactStore()
    with store.staging_dir(path, overwrite=overwrite) as staging:
        store.run_local_phase("checkpoint.write_shards", lambda: write_shards(staging))
        store.wait_for_all()
    store.publish(path, metadata=metadata)
