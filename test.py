from fault_localization_N_gram import process_all_examples
from formulas import FORMULA_MAP
from metric1 import extract_fl_rankings, extract_fl_agent_first_place
from summary import generate_summary_csv
from sbfl_martic import generate_all_matrix

if __name__ == '__main__':
    # generate_all_matrix("data/analysis_results_v4/gaia_level1", "data/sbfl_matrixs/gaia_level1", 1)
    # generate_all_matrix("data/analysis_results_v4/gaia_level2", "data/sbfl_matrixs/gaia_level2", 1)
    # generate_all_matrix("data/analysis_results_v4/gaia_level3", "data/sbfl_matrixs/gaia_level3", 1)
    # generate_all_matrix("data/analysis_results_v4/assistantbench_medium", "data/sbfl_matrixs/assistantbench_medium", 1)
    # generate_all_matrix("data/analysis_results_v4/assistantbench_hard", "data/sbfl_matrixs/assistantbench_hard", 1)

    process_all_examples("data/sbfl_matrixs_5_0/gaia_level1", "data/fault_localization_4_6/gaia_level1",
                         "all", 1, False)
    process_all_examples("data/sbfl_matrixs_5_0/gaia_level2", "data/fault_localization_4_6/gaia_level2",
                         "all", 1, False)
    process_all_examples("data/sbfl_matrixs_5_0/gaia_level3", "data/fault_localization_4_6/gaia_level3",
                         "all", 1, False)

    process_all_examples("data/sbfl_matrixs_5_0/assistantbench_medium", "data/fault_localization_4_6/assistantbench_medium",
                         "all", 1, False)
    process_all_examples("data/sbfl_matrixs_5_0/assistantbench_hard", "data/fault_localization_4_6/assistantbench_hard",
                         "all", 1, False)
    extract_fl_agent_first_place(FL_DIR="data/fault_localization_4_6", benchmark="gaia", difficulty="level1", formulas=list(FORMULA_MAP.keys()))
    extract_fl_agent_first_place(FL_DIR="data/fault_localization_4_6",benchmark="gaia", difficulty="level2", formulas=list(FORMULA_MAP.keys()))
    extract_fl_agent_first_place(FL_DIR="data/fault_localization_4_6",benchmark="gaia", difficulty="level3", formulas=list(FORMULA_MAP.keys()))
    extract_fl_agent_first_place(FL_DIR="data/fault_localization_4_6",benchmark="assistantbench", difficulty="medium", formulas=list(FORMULA_MAP.keys()))
    extract_fl_agent_first_place(FL_DIR="data/fault_localization_4_6",benchmark="assistantbench", difficulty="hard", formulas=list(FORMULA_MAP.keys()))
    
    # generate_all_matrix("data/analysis_results_v4/algorithmgenerated_normal", "data/sbfl_matrixs/algorithmgenerated_normal", 1)
    process_all_examples("data/sbfl_matrixs_5_0/algorithmgenerated_normal", "data/fault_localization_4_6/algorithmgenerated_normal",
                         "all", 1, False)
    extract_fl_agent_first_place(FL_DIR="data/fault_localization_4_6",benchmark="algorithmgenerated", difficulty="normal", formulas=list(FORMULA_MAP.keys()))

    # # generate_all_matrix("data/analysis_results_v4/algorithmgenerated_plan", "data/sbfl_matrixs/algorithmgenerated_plan", 1)
    process_all_examples("data/sbfl_matrixs_5_0//algorithmgenerated_plan", "data/fault_localization_4_6/algorithmgenerated_plan",
                         "all", 1, False)
    extract_fl_agent_first_place(FL_DIR="data/fault_localization_4_6",benchmark="algorithmgenerated", difficulty="plan", formulas=list(FORMULA_MAP.keys()))


    generate_summary_csv()