import os
from pathlib import Path
from typing import List
import pandas as pd
from agent_step import AgentStep


def convert_path_to_tuple(input_dir: str, output_dir: str, is_hierarchical=False):
    input_path = Path(input_dir)
    path_dir = input_path / "llm_cluster3"
    output_path = Path(output_dir)
    tuple_path = output_path / "tuple"
    tuple_path.mkdir(parents=True, exist_ok=True)

    all_agentsteps = set()
    file_agentsteps = []
    file_agentsteps_all = []
    # file_bigrams = []  # 新增：存储每个文件的二元组序列

    indexed_path_files = []
    for path_file in path_dir.glob("*.jsonl"):
        try:
            index = int(path_file.stem)
            indexed_path_files.append((index, path_file))
        except ValueError:
            continue

    indexed_path_files.sort(key=lambda x: x[0])

    for _, path_file in indexed_path_files:
        with open(path_file, "r", encoding="utf-8") as f:
            steps = set()
            steps_all = []
            if input_path.parent.name != "algorithmgenerated_plan":
                next(f, None)
            if f is not None:
                for line in f:
                    step = AgentStep.from_jsonl_line(line, is_hierarchical)
                    if step and step.agent == "Computer_terminal":
                        continue
                    if step:
                        steps.add(step)
                        steps_all.append(step)

            file_agentsteps.append(steps)
            all_agentsteps.update(steps)
            file_agentsteps_all.append(steps_all)


            tuple_file = tuple_path / f"{path_file.stem}.tuple"
            with open(tuple_file, "w", encoding="utf-8") as out:
                for step in sorted(steps, key=lambda x: (x.agent, x.action, x.state)):
                    out.write(f"{step}\n")


    all_agentsteps_sorted = sorted(all_agentsteps, key=lambda x: (x.agent, x.action, x.state))
    with open(tuple_path / "all.tuple", "w", encoding="utf-8") as out:
        for step in all_agentsteps_sorted:
            out.write(f"{step}\n")

    return file_agentsteps, all_agentsteps_sorted, file_agentsteps_all


def generate_step_matrix(file_agentsteps: List[set], all_agentsteps_sorted: List[AgentStep],
                         input_dir: str, output_dir: str, file_agentsteps_all: List[List]):
    matrix = []
    for step in all_agentsteps_sorted:
        row = []
        for i in range(len(file_agentsteps)):
            if step in file_agentsteps[i]:
                row.append(file_agentsteps_all[i].count(step))
            else:
                row.append(0)
        # row = [1 if step in step_set else 0 for step_set in file_agentsteps]
        matrix.append(row)
    df = pd.DataFrame(matrix)

    matrix_file = Path(output_dir) / "step_matrix.csv"
    df.to_csv(matrix_file, index=False, header=False)


    answer_file = Path(input_dir) / "answers.txt"
    test_file = Path(output_dir) / "test"
    if answer_file.exists():
        with open(answer_file, "r", encoding="utf-8") as fin, open(test_file, "w", encoding="utf-8") as fout:
            for line in fin:
                if line.startswith("[Expected Answer]:"):
                    break
                parts = line.strip().split("###")
                if len(parts) >= 3:
                    fout.write(parts[1].strip() + "\n")
    else:
        print(f"Warning: {answer_file} not found.")

    return str(matrix_file), str(test_file)


def get_all_steps_and_all_bigrams(input_dir:str):
    all_agentsteps = []
    with open(os.path.join(input_dir, "tuple", "all.tuple") , "r", encoding="utf-8") as out:
        for line in out.readlines():
            all_agentsteps.append(line.strip())
            # step = AgentStep.from_jsonl_line(line)
            # all_agentsteps.append(step)
    return all_agentsteps


def get_all_matrix_files(input_dir:str):
    return os.path.join(input_dir, "step_matrix.csv"), os.path.join(input_dir, "test")


def generate_all_matrix(analysis_dir: str, output_dir):

    for dirname in os.listdir(analysis_dir):
        input_example = os.path.join(analysis_dir, dirname)
        if not os.path.isdir(input_example): continue
        output_example = os.path.join(output_dir, dirname)
        os.makedirs(output_example, exist_ok=True)
        file_agentsteps, all_agentsteps_sorted, file_agentsteps_all = convert_path_to_tuple(input_example, output_example)
        generate_step_matrix(file_agentsteps, all_agentsteps_sorted, input_example, output_example, file_agentsteps_all)


if __name__ == '__main__':
    generate_all_matrix("data/trajectories/gaia_level1", "data/sbfl_matrixs/hand_crafted")
    generate_all_matrix("data/trajectories/gaia_level2", "data/sbfl_matrixs/hand_crafted")
    generate_all_matrix("data/trajectories/gaia_level3", "data/sbfl_matrixs/hand_crafted")
    generate_all_matrix("data/trajectories/assistantbench_medium", "data/sbfl_matrixs/hand_crafted")
    generate_all_matrix("data/trajectories/assistantbench_hard", "data/sbfl_matrixs/hand_crafted")


    generate_all_matrix("data/trajectories/algorithmgenerated_plan", "data/sbfl_matrixs/algorithm_generated")
    generate_all_matrix("data/trajectories/algorithmgenerated_normal", "data/sbfl_matrixs/algorithm_generated")





