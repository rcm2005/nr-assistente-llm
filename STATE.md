# Estado do projeto

Última atualização: 2026-10-15 19:35 EDT

## Concluído

- Repositório iniciado localmente; nenhum remoto será criado.
- Memória do host verificada: 3,4 GiB disponíveis; trabalho pesado reservado à Lightning.
- Manual operacional e seções 28/28d do histórico consultados.
- Baseline da Lightning: `TRWPhase10` Running, um runner, marker de pausa ausente, quatro consumidores FVM nas duas lanes, sem HOLD/restore ledger/batch owner e 7.489 MiB disponíveis.
- Procedimento de janela definido: criar somente o marker normal de pausa e aguardar o runner ficar Ready; manter FVM ativos. Retomar pelo guard hash-pinned `C:\trw-proc\phase21-start-full.ps1`, nunca por `lightning resume-transcription`.
- Fontes oficiais localizadas na página de NRs vigentes do MTE; licenças Apache-2.0 dos dois Qwen2.5 confirmadas.
- Ambiente criado em `C:\nr-assistente\.venv` com Python 3.11.0 e PyTorch 2.12.1+cu130; `torch.cuda.is_available()` retornou `True` e uma multiplicação CUDA 2048x2048 foi concluída na RTX 4060 Ti. Saídas reais estão em `docs/gpu.md`.
- PDFs oficiais foram baixados com URL e SHA-256 no manifesto. O recorte rápido NR-06/10/12/17/18/33/35 produziu inicialmente 1.023 candidatos e reteve 974 cláusulas após filtrar títulos/colisões de anexos.
- Dataset fast-track concluído: 720 pares aceitos, 144 armadilhas de recusa e nenhuma rejeição local. O split tem 600 exemplos de treino e 120 de teste em 120 itens exclusivos, sem `source_id` compartilhado.
- Geração auditada em 40 lotes pelo `gemini-3.8-flash-medium`: 55 tentativas, 40 sucessos e 15 saídas vazias/JSON inválido descartadas; nenhum fallback sintético local.

## Próximo

1. Sincronizar código/dados para `C:\nr-assistente` e executar os quatro runs 0,5B na janela guardada.
2. Avaliar, gerar gráficos/screenshot e concluir documentação em pt-BR.
3. Após o commit de submissão, abrir `experimento-completo` para ampliar NRs, pares e runs.

## Janelas GPU

| Janela | Início | Fim | Minutos | Uso | Estado final |
|---|---|---|---:|---|---|
| — | — | — | 0 | Nenhuma janela iniciada | Transcrição Running; marker ausente; FVM Running |

Total GPU desta tarefa: **0 min / limite aproximado de 300 min**.

## Bloqueios/segurança

- Nenhum bloqueio do Kaspersky, Smart App Control ou Windows Security observado.
- Não foram tocados serviços, containers, gravação, HOLD, documentos FVM/TRW ou credenciais.
