# RBC Pipeline

[![Build](https://github.com/childmindresearch/rbc/actions/workflows/test.yaml/badge.svg?branch=main)](https://github.com/childmindresearch/rbc/actions/workflows/test.yaml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/childmindresearch/rbc/branch/main/graph/badge.svg?token=22HWWFWPW5)](https://codecov.io/gh/childmindresearch/rbc)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![stability-stable](https://img.shields.io/badge/stability-stable-green.svg)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/childmindresearch/rbc/blob/main/LICENSE)
[![pages](https://img.shields.io/badge/api-docs-blue)](https://childmindresearch.github.io/rbc)

Reference implementation of the Reproducible Brain Charts (RBC) preprocessing protocol for anatomical, functional, derivatives, and longitudinal neuroimaging data.

## Overview

This package provides a standalone implementation of the RBC preprocessing pipeline, originally developed in [C-PAC](https://fcp-indi.github.io/). This repository serves as the maintained reference implementation using [NiWrap](https://github.com/styx-api/niwrap) for neuroimaging tool integration.

**Reference:** Shafiei et al. (2024). "Reproducible Brain Charts: An open data resource for mapping brain development and its associations with mental health." *Neuron*, 113(22):3758-3779.

## Features

- **Anatomical preprocessing**: Brain extraction (ANTs), tissue segmentation (FSL FAST), registration to MNI152
- **Functional preprocessing**: Motion correction, slice timing, coregistration, single-step resampling to template space
- **Nuisance regression**: 36-parameter and aCompCor methods with bandpass filtering
- **Quantitative metrics**: ALFF/fALFF, ReHo, network centrality, atlas-based timeseries extraction
- **Quality control**: XCP-style QC metrics with RBC-recommended thresholds
- **BIDS-compatible**: Follows BIDS conventions for inputs and outputs

## Installation

```bash
pip install git+https://github.com/childmindresearch/rbc.git
```

### Dependencies

The pipeline requires the following neuroimaging tools:
- **AFNI**: Motion correction, despiking, nuisance regression
- **FSL**: Tissue segmentation, registration, brain masking
- **ANTs**: Brain extraction, registration

## Usage

```python
from rbc.workflows import anatomical, functional

# Anatomical preprocessing
anatomical.single_session_preprocess(
    in_t1w="sub-01_T1w.nii.gz",
    output_dir="derivatives/rbc"
)

# Functional preprocessing (in development)
functional.single_session_preprocess(
    in_bold="sub-01_task-rest_bold.nii.gz",
    in_t1w="sub-01_T1w.nii.gz",
    output_dir="derivatives/rbc"
)
```

## Documentation

- **[API Documentation](https://childmindresearch.github.io/rbc)**: Full API reference

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Testing

```bash
# Fast tests only (for development)
pytest -m "not slow"

# All tests including integration tests
pytest

# Full pipeline tests (slow, ~30+ min)
pytest -m "full_pipeline"
```

See [tests/README.md](tests/README.md) for testing strategy and instructions.

## License

[MIT License](LICENSE)

## Citation

If you use this pipeline, please cite:

```bibtex
@article{shafiei2024reproducible,
  title={Reproducible Brain Charts: An open data resource for mapping brain development and its associations with mental health},
  author={Shafiei, Golia and Esper, Natasha B and Hoffmann, Madeleine S and Ai, Lei and Chen, Andrew A and Cluce, Julia and Covitz, Sydney and Giavasis, Steven and Lane, Connor and Mehta, Kahini and Moore, Tyler M and Salo, Taylor and Tapera, Tinashe M and Calkins, Monica E and Colcombe, Stanley and Davatzikos, Christos and Gur, Raquel E and Gur, Ruben C and Pan, Pedro M and Jackowski, Andrea P and Rokem, Ariel and Rohde, Luis A and Shinohara, Russell T and Tottenham, Nim and Zuo, Xi-Nian and Cieslak, Matthew and Franco, Alexandre R and Kiar, Gregory and Salum, Giovanni A and Milham, Michael P and Satterthwaite, Theodore D},
  journal={Neuron},
  volume={113},
  number={22},
  pages={3758--3779},
  year={2024},
  publisher={Elsevier},
  doi={10.1016/j.neuron.2024.08.026}
}
```

## About RBC

Reproducible Brain Charts (RBC) is an open resource integrating data from 5 large studies of brain development in youth from three continents (N = 6,346). The resource provides:

- Harmonized psychiatric phenotypes using bifactor models
- Quality-assured neuroimaging data processed with consistent pipelines
- All data openly shared via the International Neuroimaging Data-sharing Initiative (INDI)

RBC facilitates large-scale, reproducible, and generalizable research in developmental and psychiatric neuroscience.

## Acknowledgments

This implementation is based on the RBC protocol described in Shafiei et al. (2024) and originally implemented in C-PAC. Development is supported by the Child Mind Institute.
