# DLC-Hackathon-2026

Basic benchmarking, visualization, and utility scripts for the DeepLabCut Lemanic Life Sciences Hackathon 2026 (group 9).

## Getting Started

Requires Python 3.11 or 3.12.

### 1. Install the DeepLabCut fork (detector infrastructure)

Clone and install the modified DeepLabCut branch in editable mode so you can
easily tinker with the code (<https://github.com/C-Achard/DeepLabCut/pull/4>):

```bash
git clone -b cy/h-detectors https://github.com/C-Achard/DeepLabCut.git
pip install -e DeepLabCut
```

### 2. Install this repo

```bash
git clone https://github.com/deruyter92/DLC-Hackathon-2026.git
cd DLC-Hackathon-2026
pip install -e .
```

> **Note:** You can skip step 1 and run `pip install -e ".[dlc-mod]"` instead, which
> installs the DeepLabCut fork directly from Git — but an editable local clone is
> recommended if you want to modify the DeepLabCut code.


## Useful Links

- [This repo](https://github.com/deruyter92/DLC-Hackathon-2026)
- [DeepLabCut fork with detector infrastructure](https://github.com/C-Achard/DeepLabCut/pull/4)
- [Hackathon slides](https://docs.google.com/presentation/d/1pm3lMqjKjiMlNrT2Fu-fedDT8H3BM-kYVpLEZY_7dZE/edit?usp=sharing)
- [DeepLabCut Model Zoo – SuperAnimals (HuggingFace demo)](https://huggingface.co/spaces/DeepLabCut/DeepLabCutModelZoo-SuperAnimals)
