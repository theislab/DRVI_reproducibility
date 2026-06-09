"""TensorFlow 2.x compatibility: v1 graph execution for legacy MichiGAN code."""

from __future__ import annotations

import gc
from typing import Optional

import tensorflow as tf

_v1_graph_ready = False


def ensure_v1_graph() -> None:
    """Disable eager execution once so placeholders, sessions, and v1 layers work."""
    global _v1_graph_ready
    if not _v1_graph_ready:
        tf.compat.v1.disable_eager_execution()
        _v1_graph_ready = True


def reset_tensorflow_state(
    *,
    reset_graph: bool = True,
    clear_keras: bool = True,
    close_default_session: bool = True,
    gc_collect: bool = True,
    device: Optional[str] = None,
) -> None:
    """
    Tear down TensorFlow graph/session state to free RAM and best-effort GPU memory.

    Call this between long training runs or after errors. It:

    - Closes the compat v1 default session when present (e.g. last ``InteractiveSession``).
    - Clears the Keras global state when ``clear_keras`` is True.
    - Resets the default graph when ``reset_graph`` is True (required for a fresh v1 build).
    - Runs ``gc.collect()`` when ``gc_collect`` is True.

    If ``device`` is set (for example ``GPU:0``), resets TF memory statistics for that
    device when supported (this does not guarantee the CUDA driver returns all memory to
    the OS; restarting the process is the only fully reliable reset).

    Does not re-enable eager execution; the next ``train()`` run still uses v1 graph mode.
    Close any extra ``InteractiveSession`` handles you hold before calling for maximum cleanup.
    """
    if close_default_session:
        try:
            sess = tf.compat.v1.get_default_session()
            if sess is not None:
                sess.close()
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass

    if clear_keras:
        try:
            tf.keras.backend.clear_session()
        except Exception:
            pass

    if reset_graph:
        try:
            tf.compat.v1.reset_default_graph()
        except Exception:
            pass

    if device is not None:
        try:
            tf.config.experimental.reset_memory_stats(device)
        except (AttributeError, RuntimeError, ValueError, TypeError):
            pass

    if gc_collect:
        gc.collect()
