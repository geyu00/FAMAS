import os
import json
import glob
import pandas as pd
import argparse

def get_max_result_file(bug_method_dir):
    """Return the .result file with the largest index in the directory."""
    result_files = glob.glob(os.path.join(bug_method_dir, "*.result"))
    max_index = 0
    for rf in result_files:
        basename = os.path.basename(rf)
        if basename.endswith(".result"):
            try:
                idx = int(basename.split(".")[0])
                if idx > max_index:
                    max_index = idx
            except ValueError:
                continue
    if max_index == 0:
        return None
    return os.path.join(bug_method_dir, f"{max_index}.result")


def extract_fl_rankings(fl_dir, benchmark, output_dir):
    """Extract action-level ranking results for each bug."""
    buggy_path = os.path.join("data/expected_results", f"{benchmark}.txt")
    bug_map = {}
    with open(buggy_path, "r", encoding="utf-8") as f:
        for line in f:
            if "$$$" in line:
                bug_id, action = line.strip().split("$$$")
                bug_map[bug_id] = action

    fl_root = os.path.join(fl_dir, benchmark)
    result_lines = []
    header = None
    methods = []

    for bug_id in sorted(os.listdir(fl_root)):
        if bug_id not in bug_map:
            continue
        action = bug_map[bug_id]
        row = [bug_id]
        bug_root = os.path.join(fl_root, bug_id)
        if header is None:
            for method in os.listdir(bug_root):
                methods.append(method)
            header = ["bug_id"] + methods
            result_lines.append("\t".join(header))
        for method in methods:
            method_dir = os.path.join(bug_root, method)
            fl_file = get_max_result_file(method_dir)
            if fl_file is None or not os.path.isfile(fl_file):
                row.append("0")
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
                        try:
                            rank = int(parts[2].strip())
                        except:
                            rank = 0
                        break
            row.append(str(rank))
        result_lines.append("\t".join(row))

    out_dir = os.path.join(output_dir, "action_level")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{benchmark}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(result_lines))
    print(f"[Saved] Action-level rankings -> {out_path}")


def extract_fl_agent_first_place(fl_dir, benchmark, output_dir):
    """Extract agent-level first-place hit results for each bug."""
    buggy_path = os.path.join("data/expected_results", f"{benchmark}.txt")
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
                    bug_map[bug_id] = obj.get("agent", "")
    fl_root = os.path.join(fl_dir, benchmark)
    result_lines = []
    header = None
    methods = []

    for bug_id in sorted(os.listdir(fl_root)):
        if bug_id not in bug_map:
            continue
        target_agent = bug_map[bug_id]
        row = [bug_id]
        bug_root = os.path.join(fl_root, bug_id)
        if header is None:
            for method in os.listdir(bug_root):
                methods.append(method)
            header = ["bug_id"] + methods
            result_lines.append("\t".join(header))

        for method in methods:
            method_dir = os.path.join(bug_root, method)
            fl_file = get_max_result_file(method_dir)
            if fl_file is None or not os.path.isfile(fl_file):
                row.append("0")
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
                    if rank <= first_rank_value:
                        first_rank_lines.append(line)
                    else:
                        break
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
            row.append("1" if found else "0")

        result_lines.append("\t".join(row))

    out_dir = os.path.join(output_dir, "agent_level")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{benchmark}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(result_lines))
    print(f"[Saved] Agent-level rankings -> {out_path}")


def summarize_counts(folder_path: str, output_csv: str):
    """Count the number of 1s in each column and output a summary CSV."""
    txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
    results = []
    for file_path in txt_files:
        df = pd.read_csv(file_path, sep="\t")
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        count_dict = {"benchmark": file_name}
        for col in df.columns[1:]:
            count_dict[col] = (df[col].astype(str) == "1").sum()
        results.append(count_dict)

    if results:
        result_df = pd.DataFrame(results)
        total_row = {"benchmark": "total"}
        for col in result_df.columns[1:]:
            total_row[col] = result_df[col].sum()
        result_df = pd.concat([result_df, pd.DataFrame([total_row])], ignore_index=True)
        result_df.to_csv(output_csv, index=False)
        print(f"[Saved] Summary counts -> {output_csv}")
        return result_df
    else:
        print(f"[Warn] No txt files found in {folder_path}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FAMAS FL rankings and summary generation.")
    parser.add_argument("--fl_dir", type=str, required=True, help="Path to the FL results directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the output rankings and summaries.")
    args = parser.parse_args()

    FL_DIR = args.fl_dir
    OUTPUT_DIR = args.output_dir
    benchmarks = ["hand_crafted", "algorithm_generated"]

    # 1. Generate action-level and agent-level results per bug
    for bm in benchmarks:
        extract_fl_rankings(FL_DIR, bm, OUTPUT_DIR)
        extract_fl_agent_first_place(FL_DIR, bm, OUTPUT_DIR)

    # 2. Summarize counts per benchmark
    agent_csv = os.path.join(OUTPUT_DIR, "agent_level_summary.csv")
    action_csv = os.path.join(OUTPUT_DIR, "action_level_summary.csv")
    summarize_counts(os.path.join(OUTPUT_DIR, "agent_level"), agent_csv)
    summarize_counts(os.path.join(OUTPUT_DIR, "action_level"), action_csv)
