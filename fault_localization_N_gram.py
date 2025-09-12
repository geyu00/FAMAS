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
from fault_localization import get_agent_frequency
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



def compute_tf(test_results,all_agentsteps_sorted, matrix, is_norm=False):
    y = len(test_results)
    x = len(all_agentsteps_sorted)
    Nf = sum(not r for r in test_results)
    Ns = sum(r for r in test_results)

    TF = []
    for i in range(x):
        count = 0
        for j in range(y):
            if not test_results[j]:
                count += matrix[i][j]
        # count = sum(matrix[i][j] if not test_results[j] for j in range(y))
        ncf = sum(matrix[i][j] != 0 and not test_results[j] for j in range(y))
        ncs = sum(matrix[i][j] != 0 and test_results[j] for j in range(y))
        if ncf == 0:
            score = 0
        else:
            score = math.log(1 + (count*1.0)/(1.0*ncf))
        # score = math.log(1 + (count * 1.0))
        TF.append(score)
        # 添加归一化到 [0, 1] 的逻辑
    if is_norm:
        min_score = min(TF)
        max_score = max(TF)
        if max_score > min_score:  # 避免除零错误
            TF = [(s - min_score) / (max_score - min_score) for s in TF]
        else:
            TF = [0.5 for _ in TF]  # 所有值相同时，设为中性值 0.5

    return TF


def compute_correct_tf(test_results,all_agentsteps_sorted, matrix, is_norm=False):
    y = len(test_results)
    x = len(all_agentsteps_sorted)
    Nf = sum(not r for r in test_results)
    Ns = sum(r for r in test_results)

    TF = []
    for i in range(x):
        count = 0
        for j in range(y):
            if test_results[j]:
                count += matrix[i][j]
        # count = sum(matrix[i][j] if not test_results[j] for j in range(y))
        ncf = sum(matrix[i][j] != 0 and not test_results[j] for j in range(y))
        ncs = sum(matrix[i][j] != 0 and test_results[j] for j in range(y))
        score = math.log(1 + count)
        TF.append(score)
        # 添加归一化到 [0, 1] 的逻辑
    if is_norm:
        min_score = min(TF)
        max_score = max(TF)
        if max_score > min_score:  # 避免除零错误
            TF = [(s - min_score) / (max_score - min_score) for s in TF]
        else:
            TF = [0.5 for _ in TF]  # 所有值相同时，设为中性值 0.5

    return TF


def compute_ncf_tf(test_results, all_agentsteps_sorted, matrix, is_norm=False):
    y = len(test_results)
    x = len(all_agentsteps_sorted)
    Nf = sum(not r for r in test_results)
    Ns = sum(r for r in test_results)

    TF = []
    for i in range(x):

        ncf = sum(matrix[i][j] != 0 and not test_results[j] for j in range(y))
        ncs = sum(matrix[i][j] != 0 and test_results[j] for j in range(y))
        score = math.log(1 + ncf)
        TF.append(score)
        # 添加归一化到 [0, 1] 的逻辑
    if is_norm:
        min_score = min(TF)
        max_score = max(TF)
        if max_score > min_score:  # 避免除零错误
            TF = [0.2 + 0.7 *((s - min_score) / (max_score - min_score)) for s in TF]
        else:
            TF = [0.5 for _ in TF]  # 所有值相同时，设为中性值 0.5

    return TF


def compute_idf(test_results,all_agentsteps_sorted, matrix):
    y = len(test_results)
    x = len(all_agentsteps_sorted)
    Nf = sum(not r for r in test_results)
    Ns = sum(r for r in test_results)

    IDF = []
    for i in range(x):
        count = sum(matrix[i][j] for j in range(y))
        # ncf = sum(matrix[i][j] != 0 and not test_results[j] for j in range(y))
        # ncs = sum(matrix[i][j] != 0 and test_results[j] for j in range(y))
        score = math.log(1.0 + 1.0 * (Nf + Ns) / (1 + count))
        IDF.append(score)
    return IDF


def compute_tf_idf(test_results,all_agentsteps_sorted, matrix):
    x = len(all_agentsteps_sorted)
    TF = compute_idf(test_results,all_agentsteps_sorted, matrix)
    IDF = compute_idf(test_results, all_agentsteps_sorted, matrix)
    TF_IDF = []
    for i in range(x):
        TF_IDF.append(TF[i]*IDF[i])
    # max_TF_IDF = max(TF_IDF)
    # min_TF_IDF = min(TF_IDF)
    # TF_IDF_Norm = [(s-min_TF_IDF)/(max_TF_IDF-min_TF_IDF) for s in TF_IDF]
    return TF_IDF


def norm_list(target_list):
    max_l = max(target_list)
    min_l = min(target_list)
    return [(s-min_l)/(max_l-min_l) for s in target_list]


def compute_novelty_weight(test_results, all_agentsteps_sorted, matrix):
    y = len(test_results)
    x = len(all_agentsteps_sorted)
    Nf = sum(not r for r in test_results)
    Ns = sum(r for r in test_results)

    weights = []
    for i in range(x):
        count = sum(matrix[i][j] for j in range(y))

        scores = []
        ncf = sum(matrix[i][j] != 0 and not test_results[j] for j in range(y))
        ncs = sum(matrix[i][j] != 0 and test_results[j] for j in range(y))
        for j in range(y):
            if ncf+ncs-1 == 0:
                average = 0
            else:
                average = (count * 1.0 - matrix[i][j]) / ((ncf+ncs-1) * 1.0)

            # if countaverage < 1:
            #     return count
            # else:
            #     return
            # if average == 0:
            #     score = count
            # else:
            #     score = 1.0*(math.fabs(1.0*matrix[i][j])/(1.0+average))
            UF = 1 + 1.0* max(0, 2*matrix[i][j] - count)/((matrix[i][j] + 1)*1.0)

            # if matrix[i][j] == 0:
            #     UF = 1.0
            # else:
            #     UF = 1.0 + 1.0 * max(0, matrix[i][j] - average) / ((matrix[i][j]) * 1.0)
                # score = 0.8
            scores.append(UF)
        weights.append(scores)
    return weights


def compute_mpIdf(test_results, all_agentsteps_sorted, matrix):
    y = len(test_results)
    x = len(all_agentsteps_sorted)
    Nf = sum(not r for r in test_results)
    Ns = sum(r for r in test_results)

    mpIDF = []
    for i in range(x):
        ncf = sum(matrix[i][j] != 0 and not test_results[j] for j in range(y))
        ncs = sum(matrix[i][j] != 0 and test_results[j] for j in range(y))
        score = math.log(1 + (1.0*Ns)/((1 + ncs)*1.0))
        mpIDF.append(score)
        # 添加归一化到 [0, 1] 的逻辑
    min_score = min(mpIDF)
    max_score = max(mpIDF)
    if max_score > min_score:  # 避免除零错误
        mpIDF = [(s - min_score) / (max_score - min_score) for s in mpIDF]
    else:
        mpIDF = [0.5 for _ in mpIDF]  # 所有值相同时，设为中性值 0.5
    for i in range(len(mpIDF)):
        mpIDF[i] = math.exp(math.log(2)*(1-mpIDF[i]))
    return mpIDF

def compute_weight_2(test_results,all_agentsteps_sorted, matrix):
    y = len(test_results)
    x = len(all_agentsteps_sorted)
    Nf = sum(not r for r in test_results)
    Ns = sum(r for r in test_results)

    weights = []
    for i in range(x):
        count = sum(matrix[i][j] for j in range(y))

        scores = []
        for j in range(y):
            average = (count * 1.0 - matrix[i][j]) / Nf * 1.0
            if matrix[i][j] > average:
                score = 2.0
            else:
                score = 1.0*(math.fabs(1.0*matrix[i][j])/average)
                # score = 0.8
            scores.append(score)
        weights.append(scores)
    return weights


def compute_bigram_spectrum_score(bigram_matrix_file: str, test_file: str, formula_name: str, all_bigram_sorted):
    bigram_matrix = pd.read_csv(bigram_matrix_file, header=None).values
    with open(test_file, "r", encoding="utf-8") as f:
        test_results = [line.strip() == "True" for line in f]

    y = len(test_results)
    x = len(all_bigram_sorted)
    Nf = sum(not r for r in test_results)
    Ns = sum(r for r in test_results)
    formula = get_formula(formula_name)
    suspiciousness = []
    for i in range(x):
        ncf = sum(bigram_matrix[i][j] != 0 and not test_results[j] for j in range(y))
        ncs = sum(bigram_matrix[i][j] != 0 and test_results[j] for j in range(y))
        # scores = []
        # for j in range(y):
            # score = formula.compute(ncf, ncs, Nf, Ns, weights[i][j])
            # score = formula.compute(ncf, ncs, Nf, Ns, math.log(1.0+(ncf*1.0)/(1.0+ncs)))
            # score = formula.compute(ncf, ncs, Nf, Ns, weights[i][j]*(1+TF[i]), mpIdfs[i])
        score = formula.compute(ncf, ncs, Nf, Ns)
            # scores.append(score)
        suspiciousness.append(score)
    return suspiciousness


def compute_global_bigram_score(all_agentsteps_sorted, bigram_matrix_file, all_bigram_sorted, test_file, formula_name):
    all_bigram_scores = compute_bigram_spectrum_score(bigram_matrix_file, test_file, formula_name, all_bigram_sorted)
    bigram_scores = []
    for agent_step in all_agentsteps_sorted:
        sum_bigram_score = 0.0
        count_bigram = 0
        for j in range(len(all_bigram_sorted)):
            if agent_step in all_bigram_sorted[j]:
                sum_bigram_score += all_bigram_scores[j]
                count_bigram += 1
        if count_bigram == 0:
            bigram_scores.append(0.0)
        else:
            bigram_scores.append(sum_bigram_score/(1.0*count_bigram))
    return bigram_scores





def compute_spectrum_score(matrix_file: str, test_file: str, formula_name: str, input_dir: str,
                         output_dir: str, all_agentsteps_sorted: List[AgentStep], bigram_matrix_file: str=None, all_bigram_sorted=None):
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

    bigram_scores = [0.0 * x]
    if bigram_matrix_file is not None:
        bigram_scores = compute_global_bigram_score(all_agentsteps_sorted, bigram_matrix_file, all_bigram_sorted,
                                                    test_file, formula_name)

    TF = compute_tf(test_results,all_agentsteps_sorted, matrix,True)
    IDF = compute_idf(test_results, all_agentsteps_sorted, matrix)
    TF_IDF = compute_tf_idf(test_results,all_agentsteps_sorted, matrix)
    correct_TF = compute_correct_tf(test_results, all_agentsteps_sorted, matrix, True)
    ncf_TF = compute_ncf_tf(test_results, all_agentsteps_sorted, matrix, True)
    agent_freq = get_agent_frequency(input_dir, test_results)
    # weights = [[TF_IDF[i]] * y for i in range(x)]
    weights = compute_novelty_weight(test_results, all_agentsteps_sorted, matrix)
    mpIdfs = compute_mpIdf(test_results, all_agentsteps_sorted, matrix)
    # weights = [[1.0] * y for _ in range(x)]
    suspiciousness = []
    for i in range(x):
        ncf = sum(matrix[i][j] != 0 and not test_results[j] for j in range(y))
        ncs = sum(matrix[i][j] != 0 and test_results[j] for j in range(y))
        nc = ncf + ncs
        base = 0.9
        w_ncf = sum(base**(matrix[i][j] - 1) if matrix[i][j] != 0 and not test_results[j] else 0 for j in range(y))
        w_ncs = sum(base**(matrix[i][j] - 1) if matrix[i][j] != 0 and test_results[j] else 0 for j in range(y))
        w_Nf = Nf - ncf + w_ncf
        w_Ns = Ns - ncs + w_ncs

        cc = sum(matrix[i][j] for j in range(y))
        scores = []
        for j in range(y):
            # score = formula.compute(ncf, ncs, Nf, Ns, weights[i][j])
            # score = formula.compute(ncf, ncs, Nf, Ns, math.log(1.0+(ncf*1.0)/(1.0+ncs)))
            # score = formula.compute(ncf, ncs, Nf, Ns, weights[i][j]*(1+TF[i]), mpIdfs[i])
            # score = formula.compute(ncf, ncs, Nf, Ns)
            score = formula.compute(w_ncf, w_ncs, w_Nf, w_Ns)
            # score = 1
            a1 = nc / presence_map.get(agents[i], 1)
            a2 = cc / count_map.get(agents[i], 1)
            if matrix[i][j] != 0:
                # score = score * (1 + a1)
                # score = score * (1 + a2)
                # score = score * (1 + math.log(matrix[i][j], 1/base))
                # score = score * (1 + a1) * (1 + a2)
                # score = score * (1 + a1) * (1 + math.log(matrix[i][j], 1/base))
                # score = score * (1 + a2) * (1 + math.log(matrix[i][j], 1/base))
                score = score * (1 + a1) * (1 + a2) * (1 + math.log(matrix[i][j], 1/base))
                # score = score * (1 + a1) * (1 + a2) * matrix[i][j]
                1
            # score = score * (nc / presence_map.get(agents[i], 1)) * (cc / count_map.get(agents[i], 1))
            scores.append(score)
        suspiciousness.append(scores)

    # ranks = rankdata([-s for s in suspiciousness], method='average')
    # results = [
    #     (f"{tup}", suspiciousness[i], int(ranks[i]))
    #     for i, tup in enumerate(all_agentsteps_sorted)
    # ]
    # results.sort(key=lambda x: x[2])
    #
    # result_file = Path(output_dir) / f"all.{formula_name}"
    # with open(result_file, "w", encoding="utf-8") as f:
    #     for name, score, rank in results:
    #         f.write(f"{name} $$$### {score:.6f} $$$### {rank}\n")

    tuple_path = Path(input_dir) / "tuple"
    spectrum_results_path = Path(output_dir) / f"{formula_name}_results"
    spectrum_results_path.mkdir(parents=True, exist_ok=True)
    # agent_freq = get_agent_frequency(output_dir, test_results)
    spectrum_map = {}
    # with open(result_file, "r", encoding="utf-8") as f:
    #     for line in f:
    #         name, score_str, _ = line.strip().split("$$$###")
    #         spectrum_map[name.strip()] = float(score_str)
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
                    step = AgentStep.from_jsonl_line(name)
                    # score = suspiciousness[spectrum_map[name]][j]
                    # score = suspiciousness[spectrum_map[name]][j]*(1.0*(agent_freq.get(step.agent)))
                    # score =  (TF_IDF[spectrum_map[name]])
                    # if max(agent_freq.values()) == agent_freq.get(step.agent):
                    #
                    #     score = suspiciousness[spectrum_map[name]][j] * (1.0 * weights[spectrum_map[name]][j])*(1.0-(agent_freq.get(step.agent)))
                    # else:
                    #     score = suspiciousness[spectrum_map[name]][j] * (1.0 * weights[spectrum_map[name]][j])
                    # score = (1.0 * weights[spectrum_map[name]][j])
                    # if max(agent_freq.values()) == agent_freq.get(step.agent):
                    #     score = suspiciousness[spectrum_map[name]][j] * (1.0+1.0 * TF[spectrum_map[name]])*(weights[spectrum_map[name]][j])*((agent_freq.get(step.agent)))
                    # else:
                    # if max(agent_freq.values()) == agent_freq.get(step.agent):
                    #     score = 1.0 * ncf_TF[spectrum_map[name]] * (weights[spectrum_map[name]][j]) * (
                    #                 1.0 - 0.2 * correct_TF[spectrum_map[name]]) * (1.0 + TF[spectrum_map[name]])*(0.85)
                    #
                    # else:
                    # score = 1.0 * ncf_TF[spectrum_map[name]]* (weights[spectrum_map[name]][j])*(1.0-0.2*correct_TF[spectrum_map[name]])*(1.0+TF[spectrum_map[name]])
                    # score = suspiciousness[spectrum_map[name]][j]
                    # if max(agent_freq.values()) == agent_freq.get(step.agent):
                    #     score = (suspiciousness[spectrum_map[name]][j]+(TF[spectrum_map[name]]) * weights[spectrum_map[name]][j])*((agent_freq.get(step.agent)))
                    # else:
                    #     score = (suspiciousness[spectrum_map[name]][j] + (TF[spectrum_map[name]]) *
                    #              weights[spectrum_map[name]][j])
                    # score = (suspiciousness[spectrum_map[name]][j] + (TF[spectrum_map[name]]) *
                    #               weights[spectrum_map[name]][j])
                    # score = 3.0*(suspiciousness[spectrum_map[name]][j] + 2.0*(TF[spectrum_map[name]]) *
                    #          weights[spectrum_map[name]][j])
                    # score = 1.0 * TF[spectrum_map[name]]
                    score = (suspiciousness[spectrum_map[name]][j])
                    #*weights[spectrum_map[name]][j]
                    # if step.agent == "MagenticOneOrchestrator" and step.action == "plan":  score *= 0.3
                    selected.append((name, score))

        if not selected:
            continue

        scores = [-item[1] for item in selected]
        ranks = rankdata(scores, method="average")
        selected_with_rank = [(selected[i][0], selected[i][1], int(ranks[i])) for i in range(len(selected))]
        selected_with_rank.sort(key=lambda x: x[2])

        with open(spectrum_results_path / f"{j}.{formula_name}", "w", encoding="utf-8") as out:
            for name, score, rank in selected_with_rank:
                out.write(f"{name} $$$### {score:.6f} $$$### {rank}\n")


def process_all_examples(input_root: str, output_root: str, formula: str, mode: int, is_hierarchical=False):
    for dirname in os.listdir(input_root):
        if dirname == ".DS_Store":continue
        input_example = os.path.join(input_root, dirname, "normal" if mode == 0 else "llm_cluster")
        output_example = os.path.join(output_root, dirname, "normal" if mode == 0 else "llm_cluster")

        if os.path.isdir(input_example):
            print(f"Processing: {dirname}")
            os.makedirs(output_example, exist_ok=True)
            all_agentsteps_sorted, all_bigrams_sorted = sbfl_martic.get_all_steps_and_all_bigrams(input_example)
            matrix_file, test_file, bigram_matrix_file = sbfl_martic.get_all_matrix_files(input_example)
            # file_agentsteps, all_agentsteps_sorted, file_agentsteps_all,  all_bigrams_sorted, file_bigrams_all = convert_path_to_tuple(input_example, output_example, mode, is_hierarchical)
            # matrix_file, test_file, bigram_matrix_file = generate_step_matrix(file_agentsteps, all_agentsteps_sorted, input_example, output_example, file_agentsteps_all, all_bigrams_sorted, file_bigrams_all)
            if formula == "all":
                for formula_name in FORMULA_MAP.keys():
                    compute_spectrum_score(matrix_file, test_file, formula_name, input_example, output_example, all_agentsteps_sorted,bigram_matrix_file, all_bigrams_sorted)
            else:
                compute_spectrum_score(matrix_file, test_file, formula, input_example, output_example, all_agentsteps_sorted,bigram_matrix_file, all_bigrams_sorted)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fault localization pipeline.")
    parser.add_argument("-i", "--input", required=True, help="Path to input root directory")
    parser.add_argument("-o", "--output", required=True, help="Path to output root directory")
    parser.add_argument("-f", "--formula", type=str, default="ochiai", help="Suspiciousness formula (ochiai, tarantula, jaccard)")

    args = parser.parse_args()

    process_all_examples(args.input, args.output, args.formula, 1)
    