#!/usr/bin/env bash
set -eu

# ================= 0. Environment and Variable Configuration =================
export CUDA_VISIBLE_DEVICES=""
export HF_ENDPOINT="https://hf-mirror.com"

category="Toys_and_Games"
exp_name=""

# If you want to use Python from a specific environment, uncomment the line below and update the path
# PYTHON_BIN="/home/b/anaconda3/envs/ljc/bin/python"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Check the model path
if [[ ! -d "$exp_name" ]]; then
    echo "[Error] Model folder not found: $exp_name" >&2
    exit 1
fi

# ================= 1. Specify Data Files =================
test_file_name="Toys_and_Games_5_2016-10-2018-11.csv"
info_file_name="Toys_and_Games_5_2016-10-2018-11.txt"
head_titles_file_name="Toys_and_Games_head_titles.json"

test_file="./data/Amazon/test/$test_file_name"
info_file="./data/Amazon/info/$info_file_name"
head_titles_path="./data/Amazon/train/$head_titles_file_name"

if [[ ! -f "$test_file" ]]; then
    echo "[Error] Test file not found: $test_file" >&2
    exit 1
fi

if [[ ! -f "$info_file" ]]; then
    echo "[Error] Info file not found: $info_file" >&2
    exit 1
fi

if [[ ! -f "$head_titles_path" ]]; then
    echo "[Error] Head titles file not found: $head_titles_path" >&2
    exit 1
fi

# temp_dir="./temp/${category}-eval"

echo "================ Configuration Check ================"
echo "Task: $category"
echo "Data: $test_file"
echo "Model: $exp_name"
echo "=========================================="

# ================= 2. Split =================
# echo "Step 1: Splitting data (Split)..."
# "$PYTHON_BIN" -u split.py --input_path "$test_file" --output_path "$temp_dir" --cuda_list "0"

# ================= 3. Evaluate =================
echo "Step 2: Starting model inference (Evaluate)..."
inputFile="$test_file"
outputJson="$exp_name/final_result.json"

if [[ ! -f "$inputFile" ]]; then
    echo "[Error] Temp file $inputFile not found." >&2
    exit 1
fi

"$PYTHON_BIN" -u evaluate.py \
    --base_model "$exp_name" \
    --info_file "$info_file" \
    --category "$category" \
    --test_data_path "$inputFile" \
    --result_json_data "$outputJson"

# ================= 4. Merge =================
# echo "Step 3: Merging results (Merge)..."
# "$PYTHON_BIN" merge.py --input_path "$temp_dir" --output_path "$exp_name/final_result.json" --cuda_list "0"

# ================= 5. Calc =================
echo "Step 4: Calculating final scores (Calc)..."
"$PYTHON_BIN" calc.py \
    --path "$outputJson" \
    --item_path "$info_file" \
    --head_titles_path "$head_titles_path"

echo "Finished!"
