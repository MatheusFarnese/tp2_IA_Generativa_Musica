# GRU para a geração de músicas baseadas no soundtrack original de The Legend of Zelda: Twilight Princess

Para o treinamento do modelo, utilize: python train_gru.py

Para a geração de músicas, utilize: python3 generate_music.py --checkpoint trained_models/context_128/best_model.pt (pode substituir por qualquer modelo .pt treinado)

Para a análise objetiva das músicas, utilize: python3 compare_musical_metric_distributions.py raw/ generated_musics/model_128 (pode substituir os folders por qualquer folder contendo arquivos MIDI)

Para o teste de permutação entre duas distribuições, utilize os outputs do comando acima e faça: python3 permutation_test.py (estão hardcoded nas linhas 259 e 260 do arquivo os outputs do compare_musical_metric_distributions a serem utilizados).
