# Prerequisites

This repository requires the following packges: pytorch, numpy, matplotlib, scikit-learn, argparse, and torchdiffeq. These can all be installed using a Python package manager such as pip.

The file `LDEM_60S_240MPP_ADJ.TIF` must be downloaded from NASA Goddard Space Flight Center's Planetary Geology, Geophysics, and Geochemstry [Laboratory website](https://pgda.gsfc.nasa.gov/products/90) [1]. This is a 240 meter-per-pixel resolution DEM of the lunar south polar region from 60-70 degrees South. Place this file in a new folder called `dem`.


# Usage

Step 1: Pre-process the LDEM data file by running `preprocess_data.py`. This will generate a new file `dem/lola_240MPP_dem_patches_64.npy` containined 7,320 patches of shape 64 x 64. It will also plot their singular spectrum and place this figure in a new folder `results`.

Step 2: Train the JKO flow model using `train_jko_flow.py` with optional arguments listed therein; see the JKO flow paper [2] for details. Every time this script is run, a new folder will be generated in `results` corresponding to the model.

Step 3: Evaluate the learned JKO flow model using `evaluate_jko_flow.py`. Be sure to specify `--model_id`, the integer corresponding to the model in `results`, and `--L`, the integer number of blocks to be used in the trained flow. The argument `--eval_type` specifies whether to evaluate (in forward and reverse time) a single flow, a group of 5 flows, or the entire dataset (reporting error metrics).

Step 4: Sample from the learned JKO flow model using `sample_jko_flow.py`. As in step 3, specify the `--model_id` and `--L`, and whether to generate a single sample or a group of 5 samples using `--sample_type`.

The following is a suggested sequence of the above steps:

`python preprocess_data.py`

`python train_jko_flow.py --L 5 --plot_freq 1 --h 1e0 --lr_decay 0.5`

`python evaluate_jko_flow.py --eval_type "group" --model_id 0 --L 5 --dt_factor 10`

`python sample_jko_flow.py --sample_type "group" --model_id 0 --L 5 --dt_factor 10`




[1] Barker, Michael K., et al. "A New View of the Lunar South Pole from the Lunar Orbiter Laser Altimeter (LOLA)." The Planetary Science Journal 4.9 (2023): 183. https://doi.org/10.3847/PSJ/acf3e1

[2] Xu, Chen et al., "Invertible normalizing flow neural networks by JKO scheme." arXiv preprint arXiv:2212.14424 (2022).