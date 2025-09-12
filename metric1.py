from collections import defaultdict
import json
import os
from statistics import median

from formulas import FORMULA_MAP

# === 全局路径变量定义 ===
ANALYSIS_DIR = "data/analysis_results_v3"
HFL_DIR = "data/fault_localization_H"
# FL_DIR = "data/fault_localization_Weight"
BUGGY_DIR = "data/benchmark"
OUTPUT_DIR = "data/metric"


# === 功能一：提取正确用例数量 ===
def extract_pass_counts(benchmark: str, difficulty: str):
    result_lines = ["bug_id\tpass_num"]
    dir_path = os.path.join(ANALYSIS_DIR, f"{benchmark}_{difficulty}")
    if not os.path.isdir(dir_path):
        print(f"Directory not found: {dir_path}")
        return

    for bug_id in sorted(os.listdir(dir_path)):
        sub_path = os.path.join(dir_path, bug_id, "answers.txt")
        if not os.path.isfile(sub_path):
            continue

        count = 0
        with open(sub_path, "r", encoding="utf-8") as f:
            for line in f:
                if "### True" in line:
                    count += 1
        result_lines.append(f"{bug_id}\t{count}")

    os.makedirs(os.path.join(OUTPUT_DIR, "tests"), exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "tests", f"{benchmark}_{difficulty}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(result_lines))
    print(f"[Saved] Pass count to {out_path}")

# === 功能二：提取频谱定位公式下的bug定位排名 ===
def extract_fl_rankings(FL_DIR, benchmark: str, difficulty: str, formulas: list):
    # 获取所有 bug_id -> buggy_action 映射
    buggy_path = os.path.join(BUGGY_DIR, benchmark, f"{difficulty}.buggy_embedding_cluster")
    bug_map = {}
    with open(buggy_path, "r", encoding="utf-8") as f:
        for line in f:
            if "$$$" in line:
                bug_id, action = line.strip().split("$$$")
                bug_map[bug_id] = action

    # 遍历每个 bug 的定位结果
    hfl_root = os.path.join(HFL_DIR, f"{benchmark}_{difficulty}")
    fl_root = os.path.join(FL_DIR, f"{benchmark}_{difficulty}")
    result_lines = []
    header = ["bug_id"] + formulas + ["best"]
    result_lines.append("\t".join(header))

    for bug_id in sorted(os.listdir(fl_root)):
        if bug_id not in bug_map:
            continue
        action = bug_map[bug_id]
        row = [bug_id]
        ranks = []

        for formula in formulas:
            hfl_file = os.path.join(hfl_root, bug_id, "llm_cluster", f"{formula}_results", f"5.{formula}")
            fl_file = os.path.join(fl_root, bug_id, "llm_cluster", f"{formula}_results", f"5.{formula}")
            if not os.path.isfile(fl_file):
                row.append("0")  # 原来是 NA，现在设为 0
                ranks.append(0)
                continue

            rank = 0
            with open(fl_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split("$$$###")
                    if len(parts) != 3:
                        continue
                    action_string = parts[0].strip()
                    if action_string == action:
                        rank = int(parts[2].strip())
                        break

            row.append(str(rank))
            ranks.append(rank)

        # 计算 best 值（排除 0）
        nonzero_ranks = [r for r in ranks if r > 0]
        best_rank = min(nonzero_ranks) if nonzero_ranks else 0
        row.append(str(best_rank))

        result_lines.append("\t".join(row))

    os.makedirs(os.path.join(OUTPUT_DIR, "fault_localizaiton"), exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "fault_localizaiton", f"{benchmark}_{difficulty}_ranking_by_llm_embedding_cluster.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(result_lines))
    print(f"[Saved] FL rankings to {out_path}")

def extract_fl_agent_first_place(FL_DIR, benchmark: str, difficulty: str, formulas: list):
    buggy_path = os.path.join(BUGGY_DIR, benchmark, f"{difficulty}.buggy_embedding_cluster")
    bug_map = {}
    with open(buggy_path, "r", encoding="utf-8") as f:
        for line in f:
            if "$$$" in line:
                bug_id, json_str = line.strip().split("$$$")
                try:
                    obj = json.loads(bug_id) if bug_id.startswith("{") else json.loads(json_str)
                except Exception:
                    obj = None
                if obj is None:
                    try:
                        obj = json.loads(json_str)
                    except Exception:
                        continue
                if obj and isinstance(obj, dict):
                    # 这里bug_id是前半部分（uuid），json_str是json字符串，实际bug_id是前半部分
                    bug_map[bug_id] = obj.get("agent", "")
    fl_root = os.path.join(FL_DIR, f"{benchmark}_{difficulty}")
    result_lines = []
    header = ["bug_id"] + formulas + ["best"]
    result_lines.append("\t".join(header))

    for bug_id in sorted(os.listdir(fl_root)):
        if bug_id not in bug_map:
            continue
        target_agent = bug_map[bug_id]
        row = [bug_id]
        wins = []

        for formula in formulas:
            fl_file = os.path.join(fl_root,  bug_id, "llm_cluster", f"{formula}_results", f"5.{formula}")
            if not os.path.isfile(fl_file):
                print("!!!")
                row.append("0")
                wins.append(0)
                continue

            first_rank_lines = []
            first_rank_value = 1

            with open(fl_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split("$$$###")
                    if len(parts) != 3:
                        continue
                    try:
                        rank = int(parts[2].strip())
                    except Exception:
                        continue
                    if first_rank_value is None:
                        first_rank_value = rank
                    if rank <= first_rank_value:
                        first_rank_lines.append(line)
                    else:
                        break  # 排名已变，不再是第一组
            # 判断 target_agent 是否出现在第一名中
            found = False
            for line in first_rank_lines:
                parts = line.split("$$$###")
                if len(parts) != 3:
                    continue
                try:
                    info = json.loads(parts[0])
                    agent = info.get("agent", "")
                    if agent == target_agent:
                        found = True
                        break
                except Exception:
                    continue

            if found:
                row.append("1")
                wins.append(1)
            else:
                row.append("0")
                wins.append(0)

        row.append("1" if any(wins) else "0")
        result_lines.append("\t".join(row))

    os.makedirs(os.path.join(OUTPUT_DIR, "fault_localizaiton", "agent_level"), exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "fault_localizaiton", "agent_level", f"{benchmark}_{difficulty}_agent_first_place.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(result_lines))
    print(f"[Saved] Agent-level FL ranking to {out_path}")


# extract_fl_agent_first_place(benchmark="gaia", difficulty="level3", formulas=list(FORMULA_MAP.keys()))
# extract_fl_rankings(FL_DIR="data/fault_localization_v4", benchmark="gaia", difficulty="level1", formulas=list(FORMULA_MAP.keys()))
# extract_fl_rankings(FL_DIR="data/fault_localization_v4", benchmark="gaia", difficulty="level2", formulas=list(FORMULA_MAP.keys()))
# extract_fl_rankings(FL_DIR="data/fault_localization_v4", benchmark="gaia", difficulty="level3", formulas=list(FORMULA_MAP.keys()))
# extract_fl_rankings(FL_DIR="data/fault_localization_v4", benchmark="assistantbench", difficulty="medium", formulas=list(FORMULA_MAP.keys()))
# extract_fl_rankings(FL_DIR="data/fault_localization_v4", benchmark="assistantbench", difficulty="hard", formulas=list(FORMULA_MAP.keys()))
if __name__ == '__main__':
    extract_fl_agent_first_place(FL_DIR="data/fault_localization_4_6", benchmark="gaia", difficulty="level1", formulas=list(FORMULA_MAP.keys()))
    extract_fl_agent_first_place(FL_DIR="data/fault_localization_4_6", benchmark="gaia", difficulty="level2", formulas=list(FORMULA_MAP.keys()))
    extract_fl_agent_first_place(FL_DIR="data/fault_localization_4_6", benchmark="gaia", difficulty="level3", formulas=list(FORMULA_MAP.keys()))
    extract_fl_agent_first_place(FL_DIR="data/fault_localization_4_6", benchmark="assistantbench", difficulty="medium", formulas=list(FORMULA_MAP.keys()))
    extract_fl_agent_first_place(FL_DIR="data/fault_localization_4_6", benchmark="assistantbench", difficulty="hard", formulas=list(FORMULA_MAP.keys()))
    extract_fl_agent_first_place(FL_DIR="data/fault_localization_4_6",benchmark="algorithmgenerated", difficulty="normal", formulas=list(FORMULA_MAP.keys()))
    extract_fl_agent_first_place(FL_DIR="data/fault_localization_4_6",benchmark="algorithmgenerated", difficulty="plan", formulas=list(FORMULA_MAP.keys()))

