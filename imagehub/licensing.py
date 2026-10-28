"""Runtime notices for cataloged checkpoint licenses."""

from __future__ import annotations

import warnings

from imagehub.catalog import ModelSpec
from imagehub.exceptions import ImageHubLicenseWarning


def warn_if_noncommercial(spec: ModelSpec) -> None:
    """Emit the commercial-use notice for a restricted catalog model."""
    if spec.commercial_use and spec.code_commercial_use:
        return
    terms = f"{spec.license} (weights)"
    if not spec.code_commercial_use:
        terms = f"{terms} and {spec.code_license} (implementation)"
    warnings.warn(
        f"{spec.id}: {terms} is not licensed for commercial use; see the model catalog.",
        category=ImageHubLicenseWarning,
        stacklevel=3,
    )
