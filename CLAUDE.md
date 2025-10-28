# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`cluster_utils` is a Python package for simplifying interaction with compute clusters (Slurm, HTCondor) for machine learning research. It manages grid searches, hyperparameter optimization, job submission, monitoring, and result aggregation. Developed by the Autonomous Learning group at the University of Tübingen.

## Installation

```bash
# For client-side only (jobs that run on cluster)
pip install .

# For server-side (running grid_search/hp_optimization)
pip install ".[runner]"

# For all features including report generation
pip install ".[all]"

# For development
pip install ".[all-dev]"
```

## Testing Commands

```bash
# Run linting
nox -t lint

# Run type checking
nox -s mypy

# Run unit tests
nox -s pytest

# Run all integration tests
nox -s integration_tests

# Run integration tests with report generation
nox -s integration_tests_with_report_generation

# Run integration tests with nevergrad optimizer
nox -s integration_tests_with_nevergrad

# Run integration tests with virtualenv
nox -s integration_tests_with_venv

# Run specific test file
pytest tests/test_utils.py

# Run all tests for all Python versions
nox
```

## Code Formatting and Linting

```bash
# Format code
black .

# Check formatting
black --check .

# Lint code
ruff check .

# Type checking
mypy .
```

## Usage Commands

### Grid Search

```bash
python -m cluster_utils.grid_search examples/basic/grid_search.json
```

### Hyperparameter Optimization

```bash
python -m cluster_utils.hp_optimization examples/basic/optimization_metaopt.json
```

### Generate Report from Results

```bash
python -m cluster_utils.scripts.generate_report <results_directory>
```

### Plot Job Timeline

```bash
cluster_utils_plot_timeline <path_to_cluster_run_log>
```

## Architecture Overview

### Client-Server Model

The package uses a **client-server architecture** where:

- **Server**: Runs on the cluster login node, manages job submission, monitoring, and optimization
- **Client**: Code that runs within each cluster job, communicates results back to server via UDP sockets

### Package Structure

- `src/cluster_utils/base/`: Shared code between client and server (constants, communication, settings)
- `src/cluster_utils/client/`: Client-side code for jobs running on cluster
  - `server_communication.py`: UDP socket communication with server
  - `submission_state.py`: Maintains job state (id, server IP/port)
- `src/cluster_utils/server/`: Server-side code for managing cluster runs
  - `job_manager.py`: Core logic for grid search and hyperparameter optimization
  - `cluster_system.py`: Abstract interface for cluster backends
  - `slurm_cluster_system.py`: Slurm implementation
  - `condor_cluster_system.py`: HTCondor implementation
  - `dummy_cluster_system.py`: Local execution for testing
  - `communication_server.py`: UDP server receiving messages from jobs
  - `optimizers.py`: Hyperparameter optimization algorithms
  - `data_analysis.py`: Result aggregation and analysis
  - `report.py`, `latex_utils.py`: PDF report generation
- `src/cluster_utils/grid_search.py`: Entry point for grid search
- `src/cluster_utils/hp_optimization.py`: Entry point for hyperparameter optimization
- `src/cluster/`: Legacy package name (deprecated, redirects to cluster_utils)

### Client API

Jobs communicate with cluster_utils using two patterns:

**Decorator pattern:**
```python
from cluster_utils import cluster_main

@cluster_main
def main(working_dir, id, **kwargs):
    # working_dir: directory for results/checkpoints
    # id: unique job identifier
    # kwargs: user-defined parameters from config
    results = {"metric": value}
    return results  # Sent to server automatically
```

**Manual pattern:**
```python
import cluster_utils

params = cluster_utils.initialize_job()  # Get parameters and establish connection
results = {"metric": value}
cluster_utils.finalize_job(results)  # Send results to server
```

**Additional client functions:**
- `exit_for_resume()`: Exit with special code to trigger job resubmission
- `announce_fraction_finished(fraction)`: Report progress
- `announce_early_results(results)`: Send intermediate results

### Configuration Files

JSON configuration files specify:
- `optimization_procedure_name`: Name for this run
- `results_dir`: Where to store results
- `git_params`: Branch and commit to clone (for reproducibility)
- `script_relative_path`: Path to Python script to execute
- `cluster_requirements`: CPU/GPU/memory requirements
- `environment_setup`: Environment variables, pre-job scripts
- `fixed_params`: Parameters passed to all jobs
- `hyperparam_list` (grid search): Parameter values to iterate over
- `optimized_params` (hp optimization): Parameter distributions for optimization
- `optimization_setting` (hp optimization): Metric to optimize, number of samples, etc.
- `optimizer_str` (hp optimization): Optimizer algorithm (e.g., "cem_metaoptimizer", "nevergrad")

See `examples/basic/grid_search.json` and `examples/basic/optimization_metaopt.json` for complete examples.

### Job Lifecycle

1. Server parses config and creates job specifications
2. Server clones git repository to isolated directory
3. Server submits jobs to cluster with generated run scripts
4. Jobs execute and communicate results via UDP to server
5. Server aggregates results, updates optimization state
6. For hp_optimization: server proposes new hyperparameters for next iteration
7. Server generates reports (CSV, optional PDF)

### Hyperparameter Distributions

Supported distributions for hp_optimization:
- `TruncatedNormal`: Normal distribution with bounds
- `TruncatedLogNormal`: Log-normal distribution with bounds
- `IntNormal`: Integer-valued normal distribution
- `IntLogNormal`: Integer-valued log-normal distribution
- `Discrete`: Discrete set of options

### Optimizers

Available optimization algorithms:
- `cem_metaoptimizer`: Cross-Entropy Method
- `nevergrad`: Integration with Nevergrad optimization library (requires `pip install ".[nevergrad]"`)

### Environment Setup

The Python environment can be specified via:
1. Activate environment before running `python -m cluster_utils.grid_search` (simplest)
2. `environment_setup.pre_job_script`: Shell script to source before job
3. `environment_setup.variables`: Environment variables to set
4. Virtualenv/conda environment creation (see config documentation)

**Important**: If your local package is installed in the environment, it may override the git clone. Consider using `environment_setup` options for clean reproducibility.

## Development Notes

- The package uses `setuptools_scm` for versioning based on git tags
- Dependencies are split: `base` and `client` have minimal deps, `server` requires `runner` optional dependencies
- `src/cluster/` exists for backward compatibility (redirects to `cluster_utils`)
- Communication uses UDP sockets for fault tolerance (fire-and-forget messaging)
- Jobs have special exit code (defined in `RETURN_CODE_FOR_RESUME`) to trigger automatic resubmission
- Slurm job states are monitored via `sacct` command
- The server runs with progress bars (using `tqdm`) showing submitted/running/completed jobs
- Reports can be generated at different times: "never", "when_finished", or "every_iteration"
- Git integration ensures all jobs run from a clean clone at specified commit for reproducibility
