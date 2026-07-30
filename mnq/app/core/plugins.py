"""Tiny plugin-discovery helper shared by the feed and indicator registries.

This is the mechanism that makes the project extensible by *adding* files
rather than editing them: drop a module into a package, register your class in
it, and importing the package picks it up.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from types import ModuleType

log = logging.getLogger(__name__)


def discover(package: ModuleType) -> list[str]:
    """Import every non-private submodule of ``package``.

    Returns the module names that were imported.  Import failures are logged
    and skipped so one broken plugin cannot take the whole app down.
    """
    imported: list[str] = []
    for info in pkgutil.iter_modules(package.__path__):
        if info.name.startswith("_") or info.name == "base":
            continue
        full_name = f"{package.__name__}.{info.name}"
        try:
            importlib.import_module(full_name)
            imported.append(full_name)
        except Exception:  # pragma: no cover - defensive
            log.exception("failed to load plugin module %s", full_name)
    return imported
