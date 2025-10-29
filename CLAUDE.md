# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**cluster_utils** is a Python package for managing compute cluster jobs, designed for machine learning research workflows. It provides a client-server architecture where:

- **Client**: Job scripts that execute on cluster nodes, instrumented to communicate with the server
- **Server**: Main process that orchestrates job submission, monitoring, and result aggregation

Supports multiple cluster backends: Slurm, HTCondor, and local execution.

## Installation

```bash
# Basic installation (client only)
pip install .

# Full installation with server/runner capabilities
pip install ".[runner]"

# Development installation with all tools
pip install ".[all-dev]"
```

## Common Commands

### Running Tests

```bash
# Run all tests with nox (tests across Python 3.8-3.13)
nox

# Run specific test suite
nox -s pytest
nox -s lint
nox -s mypy
nox -s integration_tests

# Run pytest directly (single Python version)
pytest
```

### Linting and Formatting

```bash
# Check code formatting
black --check .

# Format code
black .

# Run linter
ruff check .

# Type checking
mypy .
```

### Running Grid Search or Hyperparameter Optimization

```bash
# Grid search
python3 -m cluster_utils.grid_search <config.json>

# Hyperparameter optimization
python3 -m cluster_utils.hp_optimization <config.json>

# Examples
python3 -m cluster_utils.grid_search examples/basic/grid_search.json
python3 -m cluster_utils.hp_optimization examples/basic/optimization_metaopt.json
```

## Code Architecture

### Client-Server Communication Model

The package uses a socket-based communication protocol between job scripts (clients) and the orchestration process (server):

1. **Job Initialization**: `initialize_job()` reads parameters and registers with server
2. **Job Execution**: User code runs with provided parameters
3. **Job Finalization**: `finalize_job()` saves metrics and reports back to server

### Key Modules

**Client Side** (`src/cluster_utils/client/`):
- `__init__.py`: Main client API (`initialize_job`, `finalize_job`, `cluster_main` decorator)
- `server_communication.py`: Socket communication with server
- `submission_state.py`: Tracks job connection state

**Server Side** (`src/cluster_utils/server/`):
- `job_manager.py`: Core orchestration logic for grid search and hyperparameter optimization
- `cluster_system.py`: Abstract interface for cluster backends
- `slurm_cluster_system.py`: Slurm-specific implementation
- `condor_cluster_system.py`: HTCondor-specific implementation
- `dummy_cluster_system.py`: Local execution (no cluster)
- `optimizers.py`: Hyperparameter optimization algorithms
- `communication_server.py`: Server-side socket handling
- `git_utils.py`: Clones and checks out specified git commit for reproducibility
- `report.py`: PDF report generation with results and plots

**Configuration** (`src/cluster_utils/base/`):
- Uses `smart_settings` library for flexible JSON/TOML/YAML configuration
- `settings.py`: Configuration schemas and validation

### Two Main Entry Points

1. **`grid_search.py`**: Exhaustive grid search over hyperparameter combinations
   - Takes `hyperparam_list` with discrete `values` for each parameter
   - Submits all combinations as separate jobs

2. **`hp_optimization.py`**: Iterative hyperparameter optimization
   - Takes `optimized_params` with continuous distributions
   - Uses optimization algorithms (CEM, Nevergrad) to sample promising regions
   - Supports early stopping of unpromising jobs

### Configuration Files

Configuration JSON/TOML files specify:
- `optimization_procedure_name`: Experiment identifier
- `results_dir`: Where to store results
- `git_params`: Branch/commit to checkout for reproducibility
- `script_relative_path`: Job script to execute (relative to git repo root)
- `environment_setup`: Pre-job scripts and environment variables
- `cluster_requirements`: CPUs, GPUs, memory, node constraints
- `fixed_params`: Parameters passed to all jobs
- `hyperparam_list` (grid search) or `optimized_params` (hp optimization): Parameters to vary

### Job Script Instrumentation

Two patterns for instrumenting job scripts:

**Pattern 1: Decorator** (simplest):
```python
from cluster_utils import cluster_main

@cluster_main
def main(working_dir, id, **kwargs):
    # Your code here
    return {"metric": value}
```

**Pattern 2: Explicit calls** (more control):
```python
from cluster_utils import initialize_job, finalize_job

params = initialize_job()
# Your code here
metrics = {"metric": value}
finalize_job(metrics, params)
```

### Result Collection

Results are aggregated in:
- `full_df.csv`: All job results with parameters and metrics
- Individual job directories with `cluster_params.csv` and `cluster_metrics.csv`
- Optional PDF reports with plots and analysis

### Reproducibility Features

- **Git integration**: Jobs run from a fresh `git clone` at specified commit
- **Parameter archiving**: All parameters saved as JSON in job working directory
- **Environment capture**: Can specify exact Python environment or virtual environment setup

### Checkpointing and Resuming

Jobs can use `exit_for_resume()` to split long-running tasks:
1. Save checkpoint to `working_dir`
2. Call `exit_for_resume()`
3. Job exits with special return code
4. cluster_utils resubmits job, which loads checkpoint and continues

## Development Notes

### Package Structure

- Minimal dependencies for client (jobs don't need scipy, pandas, etc.)
- Optional dependencies for server via `[runner]` extra
- Three sub-packages: `base` (shared), `client` (jobs), `server` (orchestration)

### Testing

- Unit tests in `tests/test_*.py`
- Integration tests in `tests/run_integration_tests.sh` and similar
- Tests use local cluster backend (no actual Slurm/HTCondor needed)
- `noxfile.py` defines test matrix across Python versions

### Adding New Cluster Backend

Subclass `ClusterSystem` in `server/cluster_system.py` and implement:
- `submit_job()`: Submit job and return job ID
- `terminate_job()`: Cancel running job
- `get_job_state()`: Query job status
- `get_job_infos()`: Get detailed job information

### Optional Dependencies

- `[runner]`: Server dependencies (pandas, numpy, scipy, tqdm, gitpython)
- `[report]`: PDF report generation (matplotlib, seaborn, scikit-learn)
- `[nevergrad]`: Nevergrad-based hyperparameter optimization
- `[dev]`: Development tools (nox, pre-commit, linters)
- `[test]`: Testing tools (pytest)
- `[mypy]`: Type checking

## Examples

See `examples/` directory:
- `examples/basic/`: Simple toy optimization problem with various configuration examples
- `examples/rosenbrock/`: Classic optimization benchmark
- `examples/checkpointing/`: Demonstrates checkpointing and resuming
- `examples/slurm_timeout_signal/`: Handling Slurm timeout signals gracefully
