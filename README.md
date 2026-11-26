# GRU para Geração de Músicas — *The Legend of Zelda: Twilight Princess*

Este repositório contém a implementação de uma rede neural recorrente (**GRU**) treinada para gerar composições musicais em formato MIDI inspiradas na trilha sonora original de *The Legend of Zelda: Twilight Princess*, além de scripts para avaliação estatística e análise objetiva das músicas geradas.

---

## 🛠️ Como Utilizar

Siga as etapas abaixo para treinar o modelo, gerar novas faixas e avaliar os resultados obtidos.

### 1. Treinamento do Modelo
Para iniciar o treinamento da rede GRU a partir do conjunto de dados, execute:

```bash
python train_gru.py
```

---

### 2. Geração de Músicas
Para gerar novas composições utilizando um modelo já treinado, execute o script passando o caminho do arquivo de *checkpoint*:

```bash
python3 generate_music.py --checkpoint trained_models/context_128/best_model.pt
```

> **Nota:** Você pode substituir `trained_models/context_128/best_model.pt` pelo caminho de qualquer outro modelo `.pt` previamente treinado.

---

### 3. Análise Objetiva das Músicas
Para comparar as distribuições de métricas musicais entre as músicas originais e as geradas pelo modelo, execute:

```bash
python3 compare_musical_metric_distributions.py raw/ generated_musics/model_128
```

> **Nota:** Os diretórios `raw/` e `generated_musics/model_128` podem ser substituídos por quaisquer pastas que contenham arquivos `.mid` / `.midi`.

---

### 4. Teste de Permutação
Para realizar o teste de permutação estatística entre duas distribuições utilizando os *outputs* gerados na etapa anterior, execute:

```bash
python3 permutation_test.py
```

> **⚠️ Atenção:** Os caminhos dos arquivos de saída gerados pelo `compare_musical_metric_distributions.py` a serem comparados estão definidos diretamente no código (*hardcoded*) nas **linhas 259 e 260** do arquivo `permutation_test.py`. Lembre-se de atualizá-los caso utilize diretórios ou arquivos diferentes.
