# MAPS Baseline

Implementation of the MAPS (Machine learning for Analysis of Proteomics in Spatial biology) MLP classifier for cell type classification in multiplexed imaging data.

**Paper:** Nature Communications 2023, DOI: [10.1038/s41467-023-44188-w](https://doi.org/10.1038/s41467-023-44188-w)

**Original code:** https://github.com/mahmoodlab/MAPS

## Architecture

4-layer MLP with 512 hidden dimensions, ReLU activations, and dropout (0.25). Input features are mean marker intensities per cell (269 channels, globally aligned). Z-score normalization is applied to inputs.

## Installation

```bash
pip install -e .
```

## Usage

```bash
python -m maps --model_name maps_0 --device_num cuda:0 --split_file splits/fov_split_v7.json
```

### CLI Options

- `--model_name`: Name for the model checkpoint (default: `maps_0`)
- `--device_num`: CUDA device (default: `cuda:0`)
- `--enable_wandb`: Enable Weights & Biases logging (default: `False`)
- `--zarr_dir`: Path to TissueNet zarr archive
- `--hidden_dim`: Hidden layer dimension (default: `512`)
- `--dropout`: Dropout rate (default: `0.25`)
- `--learning_rate`: Learning rate (default: `0.001`)
- `--batch_size`: Training batch size (default: `128`)
- `--min_epochs`: Minimum epochs before early stopping (default: `250`)
- `--max_epochs`: Maximum training epochs (default: `500`)
- `--patience`: Early stopping patience (default: `100`)
- `--split_file`: Path to pre-computed FOV split JSON (required)
- `--features_cache`: Path to cache extracted features as `.npz`
- `--min_channels`: Minimum non-DAPI channels per dataset (default: `3`)

## Known Adaptations from Original

- **CosineAnnealingLR scheduler** -- original uses constant LR
- **Macro accuracy for model selection** -- original uses validation loss
- **Sqrt-inverse-frequency sampling weights** -- original uses full inverse-frequency
- **Global 269-marker alignment** -- original uses per-dataset CSV columns
- **Missing `/255.0` post-normalization step** -- original z-scores then divides by 255 again (likely a bug in original, but part of their effective recipe)
- **Missing cell size feature** -- original includes cellSize as input
- **Dropout default 0.25** -- matching original experiment scripts, not original class default of 0.10
