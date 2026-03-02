from importlib import import_module
from pathlib import Path
from typing import Any

# Registry maps stage name → module path
_STAGE_MODULES = {
    "drift":     "pipeline.visualization.drift_viz",
    "alignment": "pipeline.visualization.alignment_viz",
    "window":    "pipeline.visualization.window_viz",
    "quality":   "pipeline.visualization.quality_viz",
    "eqs":       "pipeline.visualization.eqs_viz",
    "dynamics":  "pipeline.visualization.dynamics_viz",
    "episode":   "pipeline.visualization.episode_viz",
    "diversity": "pipeline.visualization.diversity_viz",
    "rejection": "pipeline.visualization.rejection_viz",
}


def get_visualizer(stage: str, output_dir: str | Path, config: Any):
    """
    Factory: returns the StageVisualizer for the given pipeline stage.

    Args:
        stage:      One of the keys in _STAGE_MODULES.
        output_dir: Root plots directory (e.g. data_processed/plots/).
                    The visualizer will create its own subdirectory.
        config:     The parsed config object (supports config.visualization.*).

    Returns:
        StageVisualizer instance ready to call .plot_*() methods on.

    Raises:
        ValueError: If stage name is not registered.
    """
    if stage not in _STAGE_MODULES:
        raise ValueError(
            f"Unknown visualization stage '{stage}'. "
            f"Valid stages: {list(_STAGE_MODULES)}"
        )
    module = import_module(_STAGE_MODULES[stage])
    return module.StageVisualizer(
        output_dir=Path(output_dir) / stage,
        config=config,
    )


def is_stage_enabled(stage: str, config: Any) -> bool:
    """
    Check whether a stage's visualization is enabled in config.

    Config shape expected:
        visualization:
          enabled: true
          stages:
            quality: true
            drift: false
            ...

    Falls back to True if the stages key is absent (opt-in default).
    """
    try:
        if not config.visualization.enabled:
            return False
        return getattr(config.visualization.stages, stage, True)
    except AttributeError:
        return False


__all__ = ["get_visualizer", "is_stage_enabled"]