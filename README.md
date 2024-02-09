# llm-meta-analysis
Automating meta-analysis of clinical trials (randomized controlled trials)

## SETUP

Create conda environment from the environment.yml: `conda env create -f environment.yml`

Activate the conda environment: `conda activate llm-meta-analysis`

## EXPERIMENTS

### Running Tasks

Evaluating models on different tasks for automating meta-analysis can be by running `evaluation/run_task.py`

There are 3 tasks:
- `outcome_type`: given an outcome of an randomized controlled trial (RCT), output the correct type. `binary` and `continuous` are the options. Although there are additional categories, these two options are most often found in RCTs.
- `binary_outcomes`: produce a 2x2 contingency table given the outcome, intervention, comparator, and also the abstract + results section of RCT.
- `continuous_outcomes`: produce a table of mean, standard deviation, and group sizes given the outcome, intervention, comparator, and also the abstract + results section of RCT.

Example script for running a specific task on GPT-3.5:
```bash
python3 evaluation/run_task.py --model gpt35 --task binary_outcomes --split test --output_path evaluation/outputs/binary_outcomes
```

You can change the arguments to run different tasks and models.

Arguments of `run_task.py`:
- `--model`: model to evaluate ("gpt35", "gpt4")
- `--task`: task name ("outcome_type", "binary_outcomes", "continuous_outcomes")
- `--split`: whether to run "dev" or "test" split of the dataset
- `--output_path`: path where json and csv files of the outputs from model should be saved
- `--test`: adding this flag will only run 10 randomly sampled examples from dataset. this is for debugging purposes.

### Running Evaluation/Calculating Metrics

After getting all the outputs from the models, you can get the metrics by running `evaluation/evaluate_output.py`. This script will output several different types of metrics:
- number of times the model produced "unknown" answer when it was actually "known" by human annotators
- accuracy
    - exact match: an instance is counted as correct if all the parts are correct
    - partial match: an instance is counted as correct even if only part of the answers are correct
- mean absolute error, mean squared error, and root mean squared error (only for `binary_outcomes` and `continuous_outcome` tasks)

Example script for running `evaluate_output.py`:
```bash
python3 evaluation/evaluate_output.py --task binary_outcomes --output_path evaluation/outputs/binary_outcomes/gpt35_binary_outcomes_output_20240208.json --metrics_path evaluation/metrics/binary_outcomes/
```
Arguments of `evaluate_output.py`:
- `--task`: name of the task to evaluate and calculate metrics for
- `--output_path`: path where the model outputs are located. this should include both model outputs and reference answers.
- `--metrics_path`: path where the calculated metrics should be saved

## CITATION

## TODO LIST

- [ ] add end-to-end task (chaining of all tasks together in a row)
- [ ] add finalized annotated data (dev and test) with articles
- [ ] add results from running models
- [ ] if open source results look promising adding code and results
