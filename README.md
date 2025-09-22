# Spectrum-based Failure Attribution for Multi-Agent Systems

FAMAS is the first **spectrum-based failure attribution** framework for Multi-Agent Systems (MASs).  
It automatically identifies failure-responsible actions through:

- **Re-executing failed tasks**: collects multiple counterpart trajectories for comparison  
- **LLM-based clustering**: segments verbose system logs and reconstructs coherent agent–action–state trajectories  
- **Spectrum-based analysis**: computes suspiciousness scores for each action based on occurrence frequency  


## Quick Start
### 1.Setup
```bash
# Create and activate environment
conda env create -f environment.yml
conda activate FAMASenv
```

### 2.Data Preparation
To construct the spectrum matrix required for the experiment, run:

```bash
python sbfl_matrix.py
```
The results are stored in data/sbfl_matrixs/.
- data/trajectories/: contains the execution logs of 20 re-runs plus 1 original run (from the Who&When benchmark) for each failed task.
- data/cluster3/: contains the abstracted and clustered trajectories, i.e., the processed agent–action–state sequences used for spectrum analysis.

### 3.Failure Attribution
This step performs fault localization analysis using the spectrum matrix generated in the previous step.

#### Execution Command

```bash
python failure_attribution.py -i [input_dir] -o [output_dir] -f [formula] (--no_lambda --no_gamma --no_beta --lambda_value [lambda_value])
```

#### Parameter Description
| Parameter | Short | Required | Default | Description |
|-----------|-------|----------|---------|-------------|
| `--input` | `-i` | ✅ | None | Path to input directory containing spectrum matrix files |
| `--output` | `-o` | ✅ | None | Path to output directory for results |
| `--formula` | `-f` | ❌ | `kulczynski2` | Suspiciousness formula (`ochiai`, `tarantula`, `jaccard`, `dstar`, `kulczynski2`) |
| `--no_lambda` | None | ❌ | `False` | where use λ configuration |
| `--no_gamma ` | None | ❌ | `False` | where use γ configuration |
| `--no_beta ` | None | ❌ | `False` | where use β configuration |
| `--lambda_value` | None | ❌ | `0.9` | λ parameter value (0.0-1.0) |

#### Examples

```bash
# basic use
python failure_attribution.py -i data/sbfl_matrixs/algorithm_generated -o output/results/algorithm_generated -f kulczynski2
# set λ value
python failure_attribution.py -i data/sbfl_matrixs/hand_crafted -o output/results/hand_crafted -f kulczynski2 --lambda_value 0.85
# not use β
python failure_attribution.py -i data/sbfl_matrixs/algorithm_generated -o output/results/algorithm_generated -f ochiai --no_beta
```

### 4.Metric
This step extracts action-level and agent-level rankings from the fault localization results, and generates summary statistics for each benchmark.  
You can provide either a single benchmark directory or the root `output/results/` containing multiple benchmark subdirectories.

#### Execution Command

```bash
python fl_analysis.py --fl_dir [fl_results_dir] --output_dir [output_dir]
```

####Example

```bash
# Generate metrics for a single benchmark
python fl_analysis.py --fl_dir output/results/hand_crafted --output_dir output/metric

# Generate metrics for all benchmarks under the root directory
python fl_analysis.py --fl_dir output/results/ --output_dir output/metric

```
