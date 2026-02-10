"""CLI subcommand for derivative computation.

Runs post-preprocessing analyses on cleaned BOLD data: ALFF/fALFF (amplitude
of low-frequency fluctuations), ReHo (regional homogeneity), network
centrality, and atlas-based timeseries extraction. These correspond to
pipeline steps 17-20 in the reimplementation guide.
"""

from __future__ import annotations
