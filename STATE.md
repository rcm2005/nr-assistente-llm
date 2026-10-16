# Estado da entrega — 15/10/2026

- Sete NRs oficiais no recorte; manifesto com URLs e SHA-256.
- 720 pares sintéticos validados: 600 treino, 120 teste, split por item normativo.
- Quatro treinos LoRA concluídos na RTX 4060 Ti; resultados e telemetria em `outputs/`.
- Modelo base e quatro adapters avaliados nos 120 casos retidos na GPU; previsões, métricas e telemetria em `results/eval/`.
- Dez perguntas por modelo julgadas adicionalmente por Gemini; registros em `results/eval/`.
- Interface Gradio instanciada na Lightning, com cinco modelos disponíveis e GPU detectada.
- Adapter `q05_r8_lr3e4_e2` incluído para demonstração e código, relatório e gráficos preparados.

**Pendente da sessão gráfica do usuário:** captura real da GPU dentro do VSCode em `docs/img/vscode-gpu.png`. O procedimento está em `docs/gpu.md`. O arquivo não foi simulado.

**Limitação principal:** somente 1/94 citações exatas nas perguntas respondíveis com o melhor adapter; o protótipo não está pronto para consulta normativa autônoma.
