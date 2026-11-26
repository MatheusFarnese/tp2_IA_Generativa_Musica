set -euo pipefail

CHECKPOINT="trained_models/context_128/best_model.pt"
TOKENIZER_CONFIG="processed/tokenizer.json"
OUTPUT_DIR="generated/output"

mkdir -p "$OUTPUT_DIR"

for seed in $(seq 1 200); do
    echo "=========================================="
    echo "Gerando amostra $seed/200 (seed=$seed)"
    echo "=========================================="

    python3 generate_music.py \
        --checkpoint "$CHECKPOINT" \
        --tokenizer-config "$TOKENIZER_CONFIG" \
        --decoding random \
        --seed "$seed" \
        --output "$OUTPUT_DIR/music_seed_${seed}.mid"
done

echo "Concluído! Os 200 arquivos MIDI e JSON foram salvos em '$OUTPUT_DIR'."
