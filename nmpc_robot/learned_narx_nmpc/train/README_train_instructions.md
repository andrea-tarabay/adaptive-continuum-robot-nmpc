# Training Folder: Learned NARX Model

This folder is used to collect open-loop hardware data, check the quality of the collected data, train a delayed NARX prediction model, and validate the model using rollout plots.

## Main files

```text
collect_data.py       Collects open-loop system-identification data from the robot
check_data.py         Checks collected CSV files and plots dataset coverage
train_narx.py         Trains the stable delayed NARX ridge model
plot_rollouts.py      Plots model rollouts against measured data
train_models.py       Optional helper script for running the full training pipeline
hardware_robot.py     Hardware interface for data collection
learned_nmpc.py       NARX feature construction and prediction utilities
params.py             Hardware and data-collection parameters
save_results.py       Helper functions for saving logs and plots
```

The `hardware/` folder must also be available, because the training scripts use the IMU and Dynamixel hardware drivers.

## Requirements

Install the required Python packages:

```bash
pip install numpy scipy pandas matplotlib tqdm
```

## Folder structure

Recommended structure:

```text
train/
├── collect_data.py
├── check_data.py
├── train_narx.py
├── plot_rollouts.py
├── train_models.py
├── hardware_robot.py
├── learned_nmpc.py
├── params.py
├── data/
│   ├── raw/
│   ├── selected/
│   └── analysis/
└── models/
    └── narx/
```

Raw collected data should first be saved in:

```text
data/raw/
```

Good runs selected for training should be copied to:

```text
data/selected/
```

## Hardware settings

Before collecting data, check the hardware settings in `params.py`:

```python
hardware = {
    "imu_port": "COM5",
    "dxl_port": "COM4",
    "motor_ids": [1, 2, 3],
    "gain_ma_per_n": 64.7,
    "i_max_ma": 270.0,
}
```

Also check the basic tension limits:

```python
mpc = {
    "pretension": 0.45,
    "u_min": 0.45,
    "u_max": 4.0,
}
```

## 1. Collect training data

Use `collect_data.py` to apply open-loop tendon tension commands and log the robot response.

### PRBS excitation

```bash
python collect_data.py --pattern prbs --duration 240 --dt 0.10 --seed 1
```

### Sine excitation

```bash
python collect_data.py --pattern sine --duration 240 --dt 0.10
```

### Sweep excitation

```bash
python collect_data.py --pattern sweep --dt 0.10
```

The main options are:

```text
--pattern       prbs, sine, or sweep
--duration      collection time in seconds
--dt            sampling time
--seed          random seed for PRBS
--pretension    baseline tendon tension
--amp           excitation amplitude
--hold-s        hold time for each PRBS/sweep command
--max-tension   maximum allowed tendon tension
--out-dir       output folder
```

The default output folder is:

```text
data/raw/
```

Each collected file is saved as:

```text
sysid_<pattern>_<timestamp>.csv
```

## 2. Check the collected data

After collecting data, check the CSV files:

```bash
python check_data.py --runs "data/raw/*.csv" --out-dir data/analysis
```


Use this step to verify that the data has reasonable bend-space coverage, tension variation, timing, and no NaN/Inf values.

## 3. Select good runs

Copy only the good data-collection runs into:

```text
data/selected/
```

Training uses the files in `data/selected/`. At least two selected CSV files are required for leave-one-run-out validation.

## 4. Train the NARX model

Train the stable delayed NARX ridge model using:

```bash
python train_narx.py --runs "data/selected/*.csv" --state-mode bend --ny 10 --nu 10 --ndu 4 --out-dir models/narx
```

The main options are:

```text
--state-mode    bend or phi_theta
--ny            number of output-history delays
--nu            number of input-history delays
--ndu           number of input-difference delays
--ridge-grid    ridge regularization values to test
--clip-factor   residual clipping factor for rollout stability
--out-dir       output folder for the trained model
```

The recommended state mode for this project is:

```text
bend
```

because the controller tracks:

```text
y = [bx, by] = [theta cos(phi), theta sin(phi)]
```

## 5. Training outputs

The trained model is saved as:

```text
models/narx/stable_narx_ridge_bend.npz
```

The validation summary is saved as:

```text
models/narx/stable_narx_ridge_validation_summary.csv
```

The `.npz` model file contains the learned weights, feature normalization values, delay settings, residual clipping values, and training metadata.

## 6. Plot rollout validation

After training, validate the model using rollout plots:

```bash
python plot_rollouts.py --model models/narx/stable_narx_ridge_bend.npz --runs "data/selected/*.csv" --out-dir models/narx/rollout_plots
```

This saves:

```text
models/narx/rollout_plots/*_time_rollout.png
models/narx/rollout_plots/*_bendspace_rollout.png
models/narx/rollout_plots/rollout_metrics.csv
```

The rollout validation is important because the learned model is used inside MPC, where multi-step prediction stability matters more than only one-step accuracy.

## 7. Optional full pipeline

`train_models.py` can be used as a helper script:

```bash
python train_models.py
```

It expects at least two CSV files in:

```text
data/selected/
```

It trains the NARX model and generates rollout plots. 
## 8. Using the trained model

After training, copy or reference the trained model in the controller parameter file:

```python
model = {
    "path": "models/narx/stable_narx_ridge_bend.npz",
}
```

or copy it to the controller folder, for example:

```text
identified_models/stable_narx_ridge_bend.npz
```

Then update the controller parameter file so that it points to the correct model path.
