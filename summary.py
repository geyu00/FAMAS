import os
import csv
import pandas as pd

# 参数
FAULT_DIR = "data/metric/fault_localizaiton/agent_level"
FORMULAS = ["ochiai", "tarantula", "jaccard", "dstar2", "kulczynski2"]
TOP_Ns = [1, 3, 5, 10]

def parse_rank(val):
    try:
        val = int(val)
        return val if val > 0 else None
    except:
        return None

def count_top_hits(file_path):
    stats = {n: [0] * len(FORMULAS) for n in TOP_Ns}

    with open(file_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            for i, formula in enumerate(FORMULAS):
                val = parse_rank(row[formula])
                if val is not None:
                    for n in TOP_Ns:
                        if val <= n:
                            stats[n][i] += 1
    return stats

def generate_summary():
    summary = []
    header = ["dataset"] + [f"top-{n}" for n in TOP_Ns]
    total_counts = {n: [0] * len(FORMULAS) for n in TOP_Ns}

    for filename in sorted(os.listdir(FAULT_DIR)):
        if filename.endswith(".txt"):
            path = os.path.join(FAULT_DIR, filename)
            stats = count_top_hits(path)
            dataset_name = filename.replace(".txt", "")
            row = [dataset_name]
            for n in TOP_Ns:
                counts = stats[n]
                total_counts[n] = [total_counts[n][i] + counts[i] for i in range(len(FORMULAS))]
                row.append("/".join(str(c) for c in counts))
            summary.append(row)

    # 添加 total 行
    total_row = ["total"]
    for n in TOP_Ns:
        total_row.append("/".join(str(c) for c in total_counts[n]))
    summary.append(total_row)

    # 打印输出
    print("\t".join(header))
    for row in summary:
        print("\t".join(row))

    # 写入文件
    with open("data/metric/fault_summary.txt", "w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for row in summary:
            f.write("\t".join(row) + "\n")


def generate_summary_csv():
    summary_data = []
    total_counts = {n: [0] * len(FORMULAS) for n in TOP_Ns}

    for filename in sorted(os.listdir(FAULT_DIR)):
        if filename.endswith(".txt"):
            path = os.path.join(FAULT_DIR, filename)
            stats = count_top_hits(path)
            dataset_name = filename.replace(".txt", "")
            row = {"dataset": dataset_name}
            for n in TOP_Ns:
                counts = stats[n]
                total_counts[n] = [total_counts[n][i] + counts[i] for i in range(len(FORMULAS))]
                row[f"top-{n}"] = "/".join(str(c) for c in counts)
            summary_data.append(row)

    # 添加 total 行
    total_row = {"dataset": "total"}
    for n in TOP_Ns:
        total_row[f"top-{n}"] = "/".join(str(c) for c in total_counts[n])
    summary_data.append(total_row)

    # 转为 DataFrame 并保存为 CSV
    df = pd.DataFrame(summary_data)
    output_path = "data/metric/fault_summary.csv"
    output_path = "data/metric/fault_summary.csv"
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Summary written to {output_path}")
    
    
if __name__ == "__main__":
    generate_summary_csv()
