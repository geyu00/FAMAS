from collections import defaultdict
import math
import os
import argparse
import pandas as pd
from pathlib import Path
from typing import List, Tuple
from scipy.stats import rankdata

import sbfl_martic
from agent_step import AgentStep
from formulas import FORMULA_MAP, get_formula


def get_agent_count_and_presence_map(matrix, tuple_file):
    """
    :param matrix: 2D list (rows × cols)
    :param tuple_file: path to tuple file, 每一行对应 matrix 的一行
    :return:
        count_map: dict -> {agent: 在所有列中出现的总次数}
        presence_map: dict -> {agent: 出现过的列数}
    """

    # 读取 tuple 文件里的 agent
    with open(tuple_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    agents = []
    for line in lines:
        step = AgentStep.from_jsonl_line(line)
        if step:
            agents.append(step.agent)
        else:
            agents.append(None)

    n_rows = len(matrix)
    n_cols = len(matrix[0]) if n_rows > 0 else 0

    count_map = defaultdict(int)
    presence_map = defaultdict(int)

    for j in range(n_cols):  # 遍历列
        appeared_agents = set()
        for i in range(n_rows):  # 遍历行
            if matrix[i][j] != 0:
                agent = agents[i]
                if agent is None:
                    continue
                # 累加出现次数
                count_map[agent] += matrix[i][j]
                appeared_agents.add(agent)
        # 更新 presence_map：这一列出现过的 agent +1
        for agent in appeared_agents:
            presence_map[agent] += 1

    return agents, dict(count_map), dict(presence_map)


def compute_spectrum_score(matrix_file: str, test_file: str, formula_name: str, input_dir: str,
                         output_dir: str, all_agentsteps_sorted: List[AgentStep],
                         no_lambda, no_gamma, no_beta, lambda_value):
    matrix = pd.read_csv(matrix_file, header=None).values

    with open(test_file, "r", encoding="utf-8") as f:
        test_results = [line.strip() == "True" for line in f]

    tuple_path = Path(input_dir) / "tuple"
    agents, count_map, presence_map = get_agent_count_and_presence_map(matrix, os.path.join(tuple_path, "all.tuple"))


    y = len(test_results)
    x = len(all_agentsteps_sorted)
    Nf = sum(not r for r in test_results)
    Ns = sum(r for r in test_results)
    formula = get_formula(formula_name)
    
    suspiciousness = []
    for i in range(x):
        ncf = sum(matrix[i][j] != 0 and not test_results[j] for j in range(y))
        ncs = sum(matrix[i][j] != 0 and test_results[j] for j in range(y))
        nc = ncf + ncs
        w_ncf = sum(lambda_value**(matrix[i][j] - 1) if matrix[i][j] != 0 and not test_results[j] else 0 for j in range(y))
        w_ncs = sum(lambda_value**(matrix[i][j] - 1) if matrix[i][j] != 0 and test_results[j] else 0 for j in range(y))
        w_Nf = Nf - ncf + w_ncf
        w_Ns = Ns - ncs + w_ncs

        cc = sum(matrix[i][j] for j in range(y))
        scores = []
        for j in range(y):
            # score = formula.compute(ncf, ncs, Nf, Ns, weights[i][j])
            # score = formula.compute(ncf, ncs, Nf, Ns, math.log(1.0+(ncf*1.0)/(1.0+ncs)))
            # score = formula.compute(ncf, ncs, Nf, Ns, weights[i][j]*(1+TF[i]), mpIdfs[i])
            # score = formula.compute(ncf, ncs, Nf, Ns)
            if no_lambda:
                score = formula.compute(ncf, ncs, Nf, Ns)
            else:
                score = formula.compute(w_ncf, w_ncs, w_Nf, w_Ns)
                if lambda_value == 1:
                    score *= matrix[i][j]
                elif matrix[i][j] != 0:
                    score *= (1 + math.log(matrix[i][j], 1/lambda_value))
                    
            a1 = nc / presence_map.get(agents[i], 1)
            a2 = cc / count_map.get(agents[i], 1)
            if matrix[i][j] != 0:
                if not no_gamma:
                    score *= (1 + a1)
                if not no_beta:
                    score *= (1 + a2)
            scores.append(score)
        suspiciousness.append(scores)

    tuple_path = Path(input_dir) / "tuple"
    spectrum_results_path = Path(output_dir) / f"{formula_name}_{0 if no_lambda else lambda_value}_{0 if no_gamma else 1}_{0 if no_beta else 1}"
    spectrum_results_path.mkdir(parents=True, exist_ok=True)
    spectrum_map = {}
    for i, tup in enumerate(all_agentsteps_sorted):
        spectrum_map[f"{tup}"] = i

    for j, passed in enumerate(test_results):
        if passed:
            continue
        tuple_j_file = tuple_path / f"{j}.tuple"
        if not tuple_j_file.exists():
            continue

        selected = []
        with open(tuple_j_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            max_count = 0
            for line in lines:
                name = line.strip()
                if matrix[spectrum_map[name]][j] > max_count:
                    max_count = matrix[spectrum_map[name]][j]
            for line in lines:
                name = line.strip()
                if name in spectrum_map:
                    score = (suspiciousness[spectrum_map[name]][j])
                    selected.append((name, score))

        if not selected:
            continue

        scores = [-item[1] for item in selected]
        ranks = rankdata(scores, method="average")
        selected_with_rank = [(selected[i][0], selected[i][1], int(ranks[i])) for i in range(len(selected))]
        selected_with_rank.sort(key=lambda x: x[2])

        with open(spectrum_results_path / f"{j}.result", "w", encoding="utf-8") as out:
            for name, score, rank in selected_with_rank:
                out.write(f"{name} $$$### {score:.6f} $$$### {rank}\n")


def process_all_examples(input_root: str, output_root: str, formula: str,
                         no_lambda = False, no_gamma = False, no_beta = False, lambda_value = 0.9):
    for dirname in os.listdir(input_root):
        if dirname == ".DS_Store":continue
        input_example = os.path.join(input_root, dirname)
        output_example = os.path.join(output_root, dirname)

        if os.path.isdir(input_example):
            print(f"Processing: {dirname}")
            os.makedirs(output_example, exist_ok=True)
            all_agentsteps_sorted = sbfl_martic.get_all_steps_and_all_bigrams(input_example)
            matrix_file, test_file = sbfl_martic.get_all_matrix_files(input_example)
            compute_spectrum_score(matrix_file, test_file, formula, input_example, output_example, all_agentsteps_sorted, no_lambda, no_gamma, no_beta, lambda_value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fault localization pipeline.")
    parser.add_argument("-i", "--input", required=True, help="Path to input root directory")
    parser.add_argument("-o", "--output", required=True, help="Path to output root directory")
    parser.add_argument("-f", "--formula", type=str, default="kulczynski2", help="Suspiciousness formula (ochiai, tarantula, jaccard)")
    parser.add_argument("--no-lambda", action="store_true", help="Disable lambda")
    parser.add_argument("--no-beta", action="store_true", help="Disable beta") 
    parser.add_argument("--no-gamma", action="store_true", help="Disable gamma")
    parser.add_argument("--lambda_value", type=float, default=0.9, help="Lambda value (0.5-1.0")


    args = parser.parse_args()

    process_all_examples(args.input, args.output, args.formula,
                         args.no_lambda, args.no_gamma, args.no_beta, args.lambda_value)
    