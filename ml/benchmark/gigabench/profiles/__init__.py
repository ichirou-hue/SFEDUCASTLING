"""Versioned inference presets, independent of the benchmark suite."""

from . import gigachess, qwen

QWEN_MODEL = qwen.MODEL
MODEL_PROFILES = {
    qwen.NAME: (qwen.BACKEND, qwen.INPUT_PROFILE),
    gigachess.NAME: (gigachess.BACKEND, gigachess.INPUT_PROFILE),
}


def resolve_profile(name, backend, model, input_profile):
    """Resolve defaults without silently pairing a preset with another backend."""
    if name is not None:
        preset_backend, preset_input = MODEL_PROFILES[name]
        if backend is not None and backend != preset_backend:
            raise ValueError(f"--model-profile {name} requires --backend {preset_backend}")
        backend = preset_backend
    else:
        backend = backend or gigachess.BACKEND
        preset_input = qwen.INPUT_PROFILE if backend == qwen.BACKEND else gigachess.INPUT_PROFILE
    if name == gigachess.NAME and model is not None:
        raise ValueError("Use --project-root to select the local GigaChess runtime, not --model")
    return backend, model or QWEN_MODEL, input_profile or preset_input
