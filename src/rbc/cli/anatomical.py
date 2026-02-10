"""CLI subcommand for anatomical processing.

Parses subject/session arguments and delegates to
``rbc.workflows.anatomical.single_session``, which runs the full anatomical
stream (reorientation -> brain extraction -> segmentation -> registration).
"""

from __future__ import annotations
