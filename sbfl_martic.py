import os
from pathlib import Path
from typing import List

import pandas as pd

from agent_step import AgentStep


def convert_path_to_tuple(input_dir: str, output_dir: str, mode: int, is_hierarchical=False):
    input_path = Path(input_dir)
    if mode == 0:
        path_dir = input_path / "normal_path"
    else:
        path_dir = input_path / "llm_cluster3"
    output_path = Path(output_dir)
    tuple_path = output_path / "tuple"
    norm_tuple_path = output_path / "bigram_tuple"
    tuple_path.mkdir(parents=True, exist_ok=True)
    norm_tuple_path.mkdir(parents=True, exist_ok=True)

    all_agentsteps = set()
    all_bigrams = set()  # 新增：存储所有二元组
    file_agentsteps = []
    file_agentsteps_all = []
    # file_bigrams = []  # 新增：存储每个文件的二元组序列
    file_bigrams_all = []

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
            bigrams_set = set()
            bigrams = []  # 新增：存储当前文件的二元组

            if next(f, None) is not None:
            # if f is not None:
                prev_step = None
                for line in f:
                    step = AgentStep.from_jsonl_line(line, is_hierarchical)
                    if step and step.agent == "Computer_terminal":
                        prev_step = step
                        continue
                    if step:
                        steps.add(step)
                        steps_all.append(step)
                    if prev_step is not None:
                        bigram = (prev_step, step)
                        bigrams_set.add(bigram)
                        bigrams.append(bigram)
                    prev_step = step

            file_agentsteps.append(steps)
            all_agentsteps.update(steps)
            file_agentsteps_all.append(steps_all)

            all_bigrams.update(bigrams)
            file_bigrams_all.append(bigrams)

            tuple_file = tuple_path / f"{path_file.stem}.tuple"
            with open(tuple_file, "w", encoding="utf-8") as out:
                for step in sorted(steps, key=lambda x: (x.agent, x.action, x.label)):
                    out.write(f"{step}\n")

            norm_tuple_file = norm_tuple_path / f"{path_file.stem}.tuple"
            with open(norm_tuple_file, "w", encoding="utf-8") as out:
                for bigram_step in sorted(bigrams_set, key=lambda x: (x[0].agent, x[0].action, x[0].label)):
                    out.write(f"{bigram_step}\n")

    all_agentsteps_sorted = sorted(all_agentsteps, key=lambda x: (x.agent, x.action, x.label))
    all_bigrams_sorted = sorted(all_bigrams, key=lambda x: (x[0].agent, x[0].action, x[0].label))
    with open(tuple_path / "all.tuple", "w", encoding="utf-8") as out:
        for step in all_agentsteps_sorted:
            out.write(f"{step}\n")
    with open(norm_tuple_path / "all.tuple", "w", encoding="utf-8") as out:
        for bigram_step in all_bigrams_sorted:
            out.write(f"{bigram_step}\n")

    return file_agentsteps, all_agentsteps_sorted, file_agentsteps_all, all_bigrams_sorted, file_bigrams_all


def generate_step_matrix(file_agentsteps: List[set], all_agentsteps_sorted: List[AgentStep],
                         input_dir: str, output_dir: str, file_agentsteps_all: List[List],  all_bigrams_sorted, file_bigrams_all):
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
    bigram_matrix = []
    for step in all_bigrams_sorted:
        row = []
        for i in range(len(file_bigrams_all)):
            if step in file_bigrams_all[i]:
                row.append(file_bigrams_all[i].count(step))
            else:
                row.append(0)
        # row = [1 if step in step_set else 0 for step_set in file_agentsteps]
        bigram_matrix.append(row)
    df = pd.DataFrame(matrix)

    matrix_file = Path(output_dir) / "step_matrix.csv"
    df.to_csv(matrix_file, index=False, header=False)

    bigram_df = pd.DataFrame(bigram_matrix)
    bigram_matrix_file = Path(output_dir) / "bigram_matrix.csv"
    bigram_df.to_csv(bigram_matrix_file, index=False, header=False)


    answer_file = Path(input_dir) / "normal_path" / "answers.txt"
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

    return str(matrix_file), str(test_file), str(bigram_matrix_file)


def get_all_steps_and_all_bigrams(input_dir:str):
    all_bigrams = []
    all_agentsteps = []
    # if mode == 0:
    #     path_dir = os.path.join(input_dir, "normal_path")
    # else:
    #     path_dir = os.path.join(input_dir, "llm_cluster")
    with open(os.path.join(input_dir, "tuple", "all.tuple") , "r", encoding="utf-8") as out:
        for line in out.readlines():
            all_agentsteps.append(line.strip())
            # step = AgentStep.from_jsonl_line(line)
            # all_agentsteps.append(step)
    with open(os.path.join(input_dir, "bigram_tuple","all.tuple"), "r", encoding="utf-8") as out:
        for line in out.readlines():
            all_bigrams.append(line.strip())
    return all_agentsteps, all_bigrams


def get_all_matrix_files(input_dir:str):
    # if mode == 0:
    #     path_dir = os.path.join(input_dir, "normal_path")
    # else:
    #     path_dir = os.path.join(input_dir, "llm_cluster")
    return os.path.join(input_dir, "step_matrix.csv"), os.path.join(input_dir, "test"), os.path.join(input_dir, "bigram_matrix.csv")


def generate_all_matrix(analysis_dir: str, output_dir, mode:int):

    for dirname in os.listdir(analysis_dir):
        input_example = os.path.join(analysis_dir, dirname)
        if not os.path.isdir(input_example): continue
        output_example = os.path.join(output_dir, dirname, "normal" if mode == 0 else "llm_cluster")
        os.makedirs(output_example, exist_ok=True)
        file_agentsteps, all_agentsteps_sorted, file_agentsteps_all, all_bigrams_sorted, file_bigrams_all = convert_path_to_tuple(input_example, output_example, mode)
        generate_step_matrix(file_agentsteps, all_agentsteps_sorted, input_example, output_example, file_agentsteps_all, all_bigrams_sorted, file_bigrams_all)






