# RBC Pipeline

[![Build](https://github.com/childmindresearch/rbc/actions/workflows/test.yaml/badge.svg?branch=main)](https://github.com/childmindresearch/rbc/actions/workflows/test.yaml?query=branch%3Amain)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![LGPL-3.0-or-later](https://img.shields.io/badge/license-LGPL--3.0--or--later-blue.svg)](https://github.com/childmindresearch/rbc/blob/main/LICENSE)
[![pages](https://img.shields.io/badge/api-docs-blue)](https://childmindresearch.github.io/rbc)

> [!WARNING]
> This software is under active development and not yet intended for production use. APIs, outputs, and behavior may change without notice. Use at your own risk.

Reference implementation of the [Reproducible Brain Charts (RBC)](https://reprobrainchart.github.io/) preprocessing protocol for structural and functional MRI, built on [NiWrap](https://github.com/styx-api/niwrap). Handles brain extraction, tissue segmentation, registration, motion correction, nuisance regression, and resting-state metrics, with optional longitudinal (multi-session) support. See [Shafiei et al. (2025)](https://doi.org/10.1016/j.neuron.2024.08.026) for the full protocol description.

<p align="center">
  <img src="docs/pipeline_flow.svg" alt="RBC pipeline flow" width="720">
</p>

## Quick start

Input must be a [BIDS-organized](https://bids-specification.readthedocs.io/) dataset. Neuroimaging tools (AFNI, FSL, ANTs, FreeSurfer) run automatically via Docker by default.

```bash
pip install git+https://github.com/childmindresearch/rbc.git

# Run the full cross-sectional pipeline
rbc all /data -o /data/derivatives

# Run a single stage for specific subjects
rbc functional /data -o /data/derivatives --task rest --participant-label 01 02
```

Run any command with `--help` for full options (e.g., `rbc functional --help`). Requires Python 3.12+. Pass `--runner docker` explicitly if auto-detection doesn't find your container runtime, or see the [NiWrap docs](https://github.com/styx-api/niwrap) for other runners (Podman, Singularity, local installs).

## Workflows

| Command              | Description                                                                                      |
| -------------------- | ------------------------------------------------------------------------------------------------ |
| `rbc anatomical`     | Brain extraction (ANTs), tissue segmentation (FSL FAST), registration to MNI152                  |
| `rbc functional`     | Motion correction, slice timing, BBR coregistration, single-step resampling, nuisance regression |
| `rbc metrics`        | ALFF/fALFF, ReHo, smoothing, z-scoring, atlas-based timeseries and correlation matrices          |
| `rbc qc`             | XCP-D format quality metrics, framewise displacement, DVARS, RBC pass/fail thresholds            |
| `rbc all`            | Runs all four stages in sequence, passing results in memory between stages                       |

Stages must be run in order: `anatomical` -> `functional` -> `metrics` / `qc`. The `all` command handles this automatically.

Multiple input directories are supported when raw data and derivatives live in separate trees:

```bash
rbc functional /data /pipelines/rbc -o /pipelines/rbc
```

### Longitudinal

For multi-session datasets, longitudinal workflows build a within-subject template and reprocess derivatives in that common space. Run the cross-sectional pipeline first:

```bash
rbc all /data -o /data/derivatives
rbc longitudinal all /data/derivatives -o /data/derivatives
```

`rbc long` is an alias for `rbc longitudinal`.

<details>
<summary>Individual longitudinal stages</summary>

```bash
rbc longitudinal template   /data/derivatives -o /data/derivatives
rbc longitudinal anatomical /data/derivatives -o /data/derivatives
rbc longitudinal functional /data/derivatives -o /data/derivatives --regressor 36-parameter
rbc longitudinal metrics    /data/derivatives -o /data/derivatives --atlas schaefer_200
rbc longitudinal qc         /data/derivatives -o /data/derivatives
```

</details>

## Outputs

See the [data dictionary](docs/data_dictionary.md) for a complete description of every output file.

<p align="center">
  <img src="docs/output_tree.svg" alt="RBC output tree" width="720">
</p>

## Citation

If you use this pipeline, please cite:

```bibtex
@article{shafiei2025reproducible,
  title={Reproducible Brain Charts: An open data resource for mapping brain development and its associations with mental health},
  author={Shafiei, Golia and Esper, Nathalia B. and Hoffmann, Mauricio S. and Ai, Lei and Chen, Andrew A. and Cluce, Jon and Covitz, Sydney and Giavasis, Steven and Lane, Connor and Mehta, Kahini and Moore, Tyler M. and Salo, Taylor and Tapera, Tinashe M. and Calkins, Monica E. and Colcombe, Stanley and Davatzikos, Christos and Gur, Raquel E. and Gur, Ruben C. and Pan, Pedro M. and Jackowski, Andrea P. and Rokem, Ariel and Rohde, Luis A. and Shinohara, Russell T. and Tottenham, Nim and Zuo, Xi-Nian and Cieslak, Matthew and Franco, Alexandre R. and Kiar, Gregory and Salum, Giovanni A. and Milham, Michael P. and Satterthwaite, Theodore D.},
  journal={Neuron},
  volume={113},
  number={22},
  pages={3758--3779.e6},
  year={2025},
  publisher={Elsevier},
  doi={10.1016/j.neuron.2025.08.026}
}
```

## Development

```bash
uv sync && uv run pre-commit install
```

```bash
uv run pytest -m unit                             # fast, no runner needed
uv run pytest -m "integration and not slow"        # integration tests
uv run pytest -m full_pipeline                     # full pipeline (~30+ min)
```

See [tests/README.md](tests/README.md) for the full testing strategy and [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

[LGPL-3.0-or-later](LICENSE)

This implementation is based on the RBC protocol originally implemented in [C-PAC](https://fcp-indi.github.io/). Development is supported by the [Child Mind Institute](https://childmind.org). All data are openly shared via the [International Neuroimaging Data-sharing Initiative (INDI)](https://fcon_1000.projects.nitrc.org/).
