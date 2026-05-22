"""Backward-compatible imports for the old wrapper module."""
from ft.semantickitti_ft.lifting.render_poses import (
    generate_render_poses as _generate_render_poses,
)
from ft.semantickitti_ft.models.occany_ssc import OccAnyOccHead, OccAnySSCModel

__all__ = ["OccAnyOccHead", "OccAnySSCModel", "_generate_render_poses"]

