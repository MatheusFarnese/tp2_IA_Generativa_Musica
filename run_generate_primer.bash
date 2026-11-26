set -euo pipefail

CHECKPOINT="trained_models/context_128/best_model.pt"
TOKENIZER_CONFIG="processed/tokenizer.json"
PRIMER_DIR="raw"
OUTPUT_DIR="generated/primers"

DECODING="random"   # "random" ou "beam"
SEED=42

mkdir -p "$OUTPUT_DIR"

shopt -s nullglob nocaseglob
midi_files=("$PRIMER_DIR"/*.mid "$PRIMER_DIR"/*.midi)
shopt -u nullglob nocaseglob

total=${#midi_files[@]}

if [ "$total" -eq 0 ]; then
    echo "Nenhum arquivo .mid ou .midi encontrado em '$PRIMER_DIR'."
    exit 1
fi

echo "Encontrados $total arquivos MIDI em '$PRIMER_DIR'."

count=0
for primer_path in "${midi_files[@]}"; do
    count=$((count + 1))
    filename=$(basename "$primer_path")
    stem="${filename%.*}"
    out_file="$OUTPUT_DIR/${stem}_generated.mid"

    echo "=========================================="
    echo "[$count/$total] Gerando a partir de: $filename"
    echo "=========================================="

    python generate_music.py \
        --checkpoint "$CHECKPOINT" \
        --tokenizer-config "$TOKENIZER_CONFIG" \
        --primer "$primer_path" \
        --decoding "$DECODING" \
        --seed "$SEED" \
        --output "$out_file"
done

echo "Concluído! Todas as $total músicas foram salvas em '$OUTPUT_DIR'."
