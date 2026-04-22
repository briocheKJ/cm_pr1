# Minimal 2DGS Starter Code

This project is a teaching-oriented starter code for a simple 2D Gaussian Splatting image fitting task.

The default baseline does one thing:

- use a fixed `256x256` cropped version of Van Gogh's *Starry Night*
- fit it with a fixed number of isotropic 2D Gaussians
- render with normalized weighted average blending
- optimize with PyTorch Adam
- save the target image, intermediate reconstructions, the final reconstruction, the loss curve, and evaluation summaries

The code is intentionally modular, but still small enough for undergraduate students to read.

## Directory Structure

```text
minimal_2dgs/
├── README.md
├── requirements.txt
├── main.py
├── generate_target.py
├── config.py
├── train.py
├── losses.py
├── utils.py
├── renderer/
│   ├── __init__.py
│   └── gaussian_renderer.py
├── models/
│   ├── __init__.py
│   └── gaussian_model.py
├── target_generators/
│   ├── __init__.py
│   ├── factory.py
│   ├── image_target_generator.py
│   ├── synthetic_shapes_generator.py
│   ├── txt_gaussian_generator.py
│   └── custom_target_generator_template.py
├── optimizers/
│   ├── __init__.py
│   ├── factory.py
│   ├── torch_baselines.py
│   ├── student_sgd.py
│   ├── student_adam.py
│   ├── student_adamw.py
│   ├── student_muon.py
│   ├── student_newton.py
│   └── custom_optimizer_template.py
├── initializers/
│   ├── __init__.py
│   ├── factory.py
│   ├── random_init.py
│   ├── grid_init.py
│   ├── image_sample_init.py
│   ├── bright_spot_init.py
│   └── custom_initializer_template.py
├── data/
│   ├── .gitkeep
│   ├── Starry_Night_256.png
│   └── sample_two_gaussians.txt
├── outputs/
│   └── .gitkeep
├── tests/
│   └── .gitkeep
└── smoke_test.py
```

## What The Baseline Does

The default setting uses:

- `config.target.name = "image"`
- `config.loss.name = "mse"`
- `config.optimizer.name = "torch_adam"`
- `config.initializer.name = "random"`
- `config.train.num_steps = 200`
- a center-cropped `256x256` RGB target image
- `100` isotropic Gaussians

Each Gaussian has:

- center: `(x, y)`
- scale: `sigma`
- color: `(r, g, b)`

The model stores raw trainable parameters and maps them to valid values:

- `center = sigmoid(center_raw)`
- `sigma = softplus(scale_raw) + 1e-4`
- `rgb = sigmoid(color_raw)`

For pixel position `p = (u, v)`, Gaussian `i` has weight:

```text
w_i(p) = exp( - ||p - mu_i||^2 / (2 * sigma_i^2) )
```

The renderer uses normalized weighted average blending:

```text
I(p) = (sum_i w_i(p) * c_i + eps * bg_color) / (sum_i w_i(p) + eps)
```

The training loss is plain RGB MSE.

## How To Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the default baseline:

```bash
python main.py
```

Generate only the target image:

```bash
python generate_target.py
```

Run a quick smoke test:

```bash
python smoke_test.py
```

Outputs are saved under `outputs/`.

Students should pay attention to:

- `metrics.txt`: compact summary of final numbers
- `comparison.png`: direct target / prediction / error comparison
- `loss_curve.png`: whether training is stable

## Module Responsibilities

- `config.py`
  Stores all configurable settings in grouped dataclasses.
- `main.py`
  Small entry point that launches training.
- `train.py`
  Contains the training loop and wires the system together.
- `losses.py`
  Contains several reconstruction losses and a loss factory.
- `utils.py`
  Contains helper functions for seeds, device selection, image loading/cropping, image saving, plotting, and target generation.
- `evaluation.py`
  Computes simple metrics and saves evaluation artifacts.
- `models/gaussian_model.py`
  Stores Gaussian parameters and converts raw parameters to valid render-time values.
- `renderer/gaussian_renderer.py`
  Turns current Gaussian parameters into an RGB image.
- `target_generators/`
  Contains pluggable target image generators, including a txt-driven Gaussian target generator.
- `optimizers/`
  Contains the optimizer factory, the default PyTorch baseline, and a template for student implementations.
- `initializers/`
  Contains initialization strategies and a template for student implementations.

## Default Training Flow

The main pipeline is:

1. Build a `Config`
2. Build a target generator and create the target image
3. Build a `Gaussian2DModel`
4. Build an initializer and initialize the model
5. Build a renderer
6. Build an optimizer
7. Run the training loop
8. Save images and the loss curve
9. Save evaluation metrics and a comparison figure

## What Students Will Implement Later

This starter code is structured so future homework can focus on a few clear extension points.

Students can later:

- write their own target image generator in `target_generators/`
- write their own optimizer in `optimizers/`
- write their own initialization strategy in `initializers/`
- compare different settings by editing `Config`
- add evaluation and ablation code on top of the current baseline

The current evaluation already gives a simple scoreboard:

- lower `MSE` is better
- lower `MAE` is better
- higher `PSNR` is better

This is meant to make method comparison obvious during homework runs.

## Config Layout

All configurable parameters now live in `config.py`, grouped by purpose:

- `config.system`
  Seed, device, and output directory
- `config.target`
  Target generator choice, image size, and target file paths
- `config.model`
  Number of Gaussians and renderer feature switches
- `config.render`
  Background color and numerical epsilon
- `config.train`
  Training step count and logging / saving frequency
- `config.loss`
  Loss choice and loss-specific weights
- `config.optimizer`
  Optimizer choice and per-optimizer hyperparameters
- `config.initializer`
  Initializer choice and per-initializer hyperparameters
- `config.evaluation`
  Output filenames for evaluation artifacts
- `config.visualization`
  Optional animation export settings

Example:

```python
from config import Config

config = Config()
config.optimizer.name = "student_adam"
config.optimizer.student_adam.lr = 1e-2
config.initializer.name = "bright_spot"
config.model.use_anisotropic = True
```

This layout is meant to make it obvious to students which settings they should edit for a given experiment.

## Optional Video Export

The default run does not export an optimization video.

If you want an animation of the fitting process, enable it in `config.py`:

```python
config.visualization.save_video = True
config.visualization.video_every = 10
config.visualization.video_filename = "optimization.gif"
```

This will save:

- frame images under `outputs/video_frames/`
- an animation such as `outputs/optimization.gif`

If you set `video_filename = "optimization.mp4"` and `imageio` is available, the code will try to write an mp4. Otherwise it falls back to a gif.

## Training Loss Options

The project now supports multiple training losses through `config.loss.name`:

- `mse`
  Strong default baseline for image fitting.
- `l1`
  More robust to outliers, but sometimes less smooth.
- `charbonnier`
  Smooth approximation to L1, often a good compromise.
- `mse_l1`
  Combines stable MSE fitting with sharper L1 behavior.
- `mse_edge`
  Adds an edge-matching term using Sobel gradients, which can help preserve boundaries.

Examples:

```python
config.loss.name = "mse"
```

```python
config.loss.name = "mse_l1"
```

```python
config.loss.name = "mse_edge"
config.loss.edge_weight = 0.1
```

These losses are designed to be simple enough for students to read and modify.

## Initializer Interface

All initializers now follow the same interface:

```python
initializer.initialize(model, target_image=target)
```

The important convention is:

- `model` is always required
- `target_image` is optional

This means a student initializer may:

- ignore `target_image` completely
- use `target_image` to build a smarter image-aware initialization

That makes it easy to compare target-independent and target-dependent methods under one interface.

Current initializer choices:

- `random`
  Does not use the target image.
- `grid`
  Does not use the target image.
- `image_sample`
  Samples colors from the target image at random center locations.
- `bright_spot`
  Places centers on bright regions of the target image and samples their colors.

This is useful because some image fitting tasks benefit a lot from better starting centers and colors.

## Gaussian Txt Target Generator

The project now includes a target image generator that reads Gaussian parameters
from a txt file and renders them into an RGB target image.

Set this in `Config`:

```python
config.target.name = "txt_gaussians"
config.target.gaussian_txt_path = "data/sample_two_gaussians.txt"
config.target.image_size = 256
```

Supported txt formats:

```text
# x  y  sigma  r  g  b
0.30 0.32 0.11 0.95 0.25 0.20
0.72 0.68 0.14 0.18 0.35 0.95
```

You can also use richer formats:

```text
# isotropic + alpha
x  y  sigma  alpha  r  g  b

# anisotropic
x  y  sigma_x  sigma_y  theta  r  g  b

# anisotropic + alpha
x  y  sigma_x  sigma_y  theta  alpha  r  g  b
```

Each valid line defines one Gaussian. If the txt file has 2 lines, the target
generator renders exactly those 2 Gaussians into the target image.

This is useful for:

- debugging student optimizers
- debugging initialization logic
- building controlled target images for experiments

Students can add their own target generators by creating a new file under
`target_generators/` and registering it in `target_generators/factory.py`.

Five built-in txt examples are provided under `data/examples/`:

- `01_single_gray_isotropic_star.txt`
- `02_single_gray_anisotropic_star.txt`
- `03_ten_gray_stars.txt`
- `04_ten_translucent_stars.txt`
- `05_ten_colorful_stars.txt`

## How To Add A New Optimizer

1. Create a new file under `optimizers/` or extend an existing one.
2. Implement the optimizer logic.
3. Register it in `optimizers/factory.py`.
4. Set `config.optimizer.name` in `Config`.

For example, future student names are already reserved:

- `student_sgd`
- `student_momentum`
- `student_adam`

This starter code already includes two student-editable optimizer files:

- `optimizers/student_sgd.py`
- `optimizers/student_adam.py`
- `optimizers/student_adamw.py`
- `optimizers/student_muon.py`
- `optimizers/student_newton.py`

Students can switch optimizers just by editing `Config`:

```python
config.optimizer.name = "student_sgd"
```

or

```python
config.optimizer.name = "student_adam"
```

or

```python
config.optimizer.name = "student_adamw"
```

or

```python
config.optimizer.name = "student_muon"
```

or

```python
config.optimizer.name = "student_newton"
```

The goal is that a student can modify only that one optimizer file and then run
the training loop again to compare metrics.

See `optimizers/custom_optimizer_template.py` for the minimal interface.

## How To Add A New Initializer

1. Create a new file under `initializers/`.
2. Implement an object with an `initialize(model, target_image=None)` method.
3. Register it in `initializers/factory.py`.
4. Set `config.initializer.name` in `Config`.

Current built-in strategies are:

- `random`
- `grid`
- `image_sample`
- `bright_spot`

See `initializers/custom_initializer_template.py` for the starter template.

## Migration Notes From The Earlier Minimal Version

To keep the starter code aligned with the course specification, a few pieces were simplified during refactoring:

- the old `model.py` logic moved to `models/gaussian_model.py`
- the old `renderer.py` logic moved to `renderer/gaussian_renderer.py`
- the old training code in `main.py` moved to `train.py`
- target image generation is now a pluggable module under `target_generators/`
- initialization is no longer embedded in the model
- the default baseline remains isotropic Gaussians with normalized weighted blending
- the default target is now the `256x256` cropped Van Gogh image again

This makes the code easier to extend for assignments without turning it into a large framework.
