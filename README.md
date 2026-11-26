GRU para Geração de Músicas — The Legend of Zelda: Twilight Princess

Este projeto utiliza uma Gated Recurrent Unit (GRU) para gerar músicas inspiradas no soundtrack original de The Legend of Zelda: Twilight Princess, utilizando arquivos MIDI para o treinamento e a geração musical.

Como utilizar
1. Treinamento do modelo

Para iniciar o treinamento da rede neural GRU, execute:

bash
python train_gru.py

2. Geração de músicas

Após o treinamento, utilize um checkpoint do modelo para gerar novas músicas:

bash
python3 generate_music.py --checkpoint trained_models/context_128/best_model.pt

Nota: O caminho do checkpoint pode ser substituído por qualquer outro arquivo .pt gerado durante o treinamento.
3. Análise objetiva das músicas

Para comparar as distribuições de métricas musicais entre os arquivos MIDI originais e os arquivos gerados, execute:

bash
python3 compare_musical_metric_distributions.py raw/ generated_musics/model_128

raw/: diretório contendo os arquivos MIDI originais.
generated_musics/model_128: diretório contendo os arquivos MIDI gerados pelo modelo.
Nota: Ambos os diretórios podem ser substituídos por quaisquer outros diretórios que contenham arquivos MIDI.
4. Teste de permutação entre distribuições

Para realizar o teste de permutação estatística entre duas distribuições de métricas musicais, utilize os resultados obtidos na etapa anterior:

bash
python3 permutation_test.py

Importante: Os resultados utilizados pelo teste estão definidos diretamente no código (hardcoded), nas linhas 259 e 260 do arquivo permutation_test.py. É necessário substituir esses valores pelos outputs desejados do script compare_musical_metric_distributions.py.
Fluxo de execução

O fluxo geral de utilização do projeto é:

Treinamento: treinar a GRU com train_gru.py.
Geração: gerar novas composições com generate_music.py.
Análise: comparar as métricas musicais dos arquivos originais e gerados.
Avaliação estatística: aplicar o teste de permutação para comparar as distribuições.

Projeto desenvolvido para explorar a geração de música procedural com redes neurais recorrentes, utilizando como referência o soundtrack de The Legend of Zelda: Twilight Princess.
