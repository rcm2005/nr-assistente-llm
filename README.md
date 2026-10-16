# NR-Assistente

Assistente experimental em português para perguntas sobre Normas Regulamentadoras brasileiras.
O projeto compara o `Qwen2.5-0.5B-Instruct` original com adapters LoRA treinados em perguntas e
respostas sintéticas, fundamentadas em PDFs oficiais do Ministério do Trabalho e Emprego (MTE).

> Uso acadêmico. As respostas podem conter erros e não substituem a leitura da NR vigente nem a
> orientação de profissional habilitado.

## Problema e recorte

O fast-track usa NR-06, NR-10, NR-12, NR-17, NR-18, NR-33 e NR-35. Os textos são baixados da página
de NRs vigentes do MTE, têm URL e SHA-256 registrados em `data/raw/manifest.json` e são divididos
por item normativo. Perguntas fora do trecho-fonte são usadas como armadilhas para treinar recusa.

O Qwen2.5 foi escolhido por ser um modelo instruct aberto (Apache-2.0), pequeno o bastante para a
RTX 4060 Ti de 8 GB disponível e compatível com LoRA. O modelo de 0,5B permite comparar quatro
configurações curtas; o de 1,5B é uma extensão opcional com QLoRA de 4 bits ou LoRA em bf16.

## Resultado obtido em 15/10/2026

Quatro configurações LoRA foram treinadas de fato na RTX 4060 Ti, e o modelo original e os quatro
adapters foram avaliados nos mesmos 120 exemplos retidos. O melhor compromisso no recorte foi
`q05_r8_lr3e4_e2`: ROUGE-L **0,353**, recusa correta em **22/26** perguntas sem resposta e
**1/94** citações exatamente corretas nas perguntas respondíveis. O último número é uma limitação
séria: o modelo ainda não serve para consulta normativa autônoma. Veja as métricas e exemplos no
[relatório](docs/relatorio.md) e os dados brutos em [`results/eval`](results/eval).

![Comparação de qualidade](docs/img/metricas-qualidade.png)

## Instalação na Lightning (Windows)

Requer Python 3.11, o driver NVIDIA medido em [docs/gpu.md](docs/gpu.md) e PowerShell:

```powershell
py -3.11 -m venv C:\nr-assistente\.venv
C:\nr-assistente\.venv\Scripts\python.exe -m pip install --upgrade pip
C:\nr-assistente\.venv\Scripts\python.exe -m pip install `
  -r C:\nr-assistente\requirements.txt `
  --extra-index-url https://download.pytorch.org/whl/cu130
C:\nr-assistente\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

O wheel CUDA e a operação real de verificação usados neste trabalho estão documentados, sem
substituir resultados por exemplos, em [docs/gpu.md](docs/gpu.md).

## Reprodução

Na raiz do repositório:

```powershell
# Fontes e chunks do recorte
python scripts/download_nrs.py --only 6 10 12 17 18 33 35
python scripts/extract_items.py --only 6 10 12 17 18 33 35 `
  --output data/interim/items-subset.jsonl

# Geração autenticada pelo agy, validação e split sem item compartilhado
python scripts/generate_dataset.py --target 720 --batch-size 18 --parallel 2
python scripts/validate_dataset.py
python scripts/split_dataset.py --test-size 120
python scripts/build_dataset_card.py

# Um run; os demais IDs estão em configs/experiments.yaml
python -m scripts.train --run q05_r8_lr1e4_e1

# Avaliação do conjunto retido na GPU (troque os caminhos pelos adapters disponíveis)
python -m scripts.evaluate --model base=Qwen/Qwen2.5-0.5B-Instruct `
  --model r8_lr3e4_e2=outputs/q05_r8_lr3e4_e2

# Aplicação
python app.py
```

O gerador não cria exemplos locais de fallback: lotes que falham ficam registrados e a execução
pode ser retomada. A validação exige identidade da fonte, formato de citação, recusa sem citação e
perguntas não duplicadas. O split agrupa por `source_id` para impedir vazamento do mesmo item.

## Interface

A aplicação Gradio oferece pergunta, seletor entre modelo base e adapters existentes,
`temperature`, `top_p`, `max_tokens`, resposta, citações extraídas, tokens/s e estado da GPU.
Os pesos do modelo base são baixados do Hugging Face. O adapter `q05_r8_lr3e4_e2` está incluído
no repositório; os outros são produzidos por `scripts/train.py` em `outputs/` e seus pesos foram
excluídos do Git por tamanho. A interface lista somente os adapters presentes no computador.
A interface é uma demonstração acadêmica;
as citações produzidas pelo modelo devem ser conferidas na NR oficial.

## Documentação

- [Relatório técnico](docs/relatorio.md)
- [Ambiente e evidências da GPU](docs/gpu.md)
- [Cartão do dataset](docs/dataset-card.md)
- [Roteiro de apresentação](docs/apresentacao.md)

## Integrantes

- Rafael Carvalho Mattos — RM 99874
- Rafael Autieri dos Anjos — RM 550885
- Luiza Cristina Silva — RM 99367
- Levy Nascimento Junior — RM 98655

## Trabalhos futuros

Ampliar o corpus para todas as 36 NRs vigentes, chegar a cerca de 3 mil pares revisados, executar
mais combinações de hiperparâmetros e sementes, aprofundar os testes do modelo de 1,5B e incluir
revisão de conteúdo por especialista em segurança e saúde no trabalho.

## Licença

Código sob licença MIT. Os textos normativos pertencem às respectivas fontes públicas oficiais;
os modelos Qwen2.5 usados no experimento são distribuídos sob Apache-2.0.
