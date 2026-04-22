"""Lazy exports for ``metadrive.engine`` to avoid circular imports.

``base_object`` imports ``metadrive.engine.asset_loader``; loading the
``metadrive.engine`` package must not eagerly import ``engine_utils`` (which
pulls ``EngineCore`` → ``Light`` → ``BaseObject`` while ``base_object`` is still
initializing). See https://github.com/metadriverse/metadrive issues around import
order; SIPACL uses a local clone — keep this file minimal at import time.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, Tuple

_LAZY: Dict[str, Tuple[str, str]] = {
    "get_engine": ("metadrive.engine.engine_utils", "get_engine"),
    "initialize_engine": ("metadrive.engine.engine_utils", "initialize_engine"),
    "engine_initialized": ("metadrive.engine.engine_utils", "engine_initialized"),
    "close_engine": ("metadrive.engine.engine_utils", "close_engine"),
    "get_global_config": ("metadrive.engine.engine_utils", "get_global_config"),
    "set_global_random_seed": ("metadrive.engine.engine_utils", "set_global_random_seed"),
    "get_logger": ("metadrive.engine.logger", "get_logger"),
    "AssetLoader": ("metadrive.engine.asset_loader", "AssetLoader"),
    "BaseRigidBodyNode": ("metadrive.engine.physics_node", "BaseRigidBodyNode"),
    "BaseGhostBodyNode": ("metadrive.engine.physics_node", "BaseGhostBodyNode"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod_name, attr_name = _LAZY[name]
    mod = importlib.import_module(mod_name)
    value = getattr(mod, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))
