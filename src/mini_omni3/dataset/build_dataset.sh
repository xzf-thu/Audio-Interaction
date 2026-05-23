set -e
export PYTHONPATH=$(pwd)/src   # run from repo root

INPUT_JSONL=/path/to/input/shard_00.jsonl
OUTPUT_JSONL=/path/to/output/shard_00.jsonl
ERROR_JSONL=/path/to/output/error_shard_00.jsonl
FEATURE_DIR=/path/to/audio_features/shard_00

mkdir -p "$(dirname "$OUTPUT_JSONL")" "$FEATURE_DIR"

CUDA_VISIBLE_DEVICES=0 python src/mini_omni3/dataset/build_online.py \
    "$INPUT_JSONL" "$OUTPUT_JSONL" "$ERROR_JSONL" "$FEATURE_DIR"

