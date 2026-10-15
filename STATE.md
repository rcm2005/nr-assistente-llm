# Estado do projeto

Última atualização: 2026-10-15 19:05 EDT

## Concluído

- Repositório iniciado localmente; nenhum remoto será criado.
- Memória do host verificada: 3,4 GiB disponíveis; trabalho pesado reservado à Lightning.
- Manual operacional e seções 28/28d do histórico consultados.
- Baseline da Lightning: `TRWPhase10` Running, um runner, marker de pausa ausente, quatro consumidores FVM nas duas lanes, sem HOLD/restore ledger/batch owner e 7.489 MiB disponíveis.
- Procedimento de janela definido: criar somente o marker normal de pausa e aguardar o runner ficar Ready; manter FVM ativos. Retomar pelo guard hash-pinned `C:\trw-proc\phase21-start-full.ps1`, nunca por `lightning resume-transcription`.
- Fontes oficiais localizadas na página de NRs vigentes do MTE; licenças Apache-2.0 dos dois Qwen2.5 confirmadas.

## Próximo

1. Implementar pipeline de download, extração por item, geração/validação e split sem vazamento.
2. Baixar textos vigentes oficiais e gerar 2.000–3.000 pares com Gemini Flash.
3. Preparar e testar pipeline de treino/eval/frontend.
4. Sincronizar para `C:\nr-assistente`, instalar ambiente CUDA compatível e executar pelo menos seis runs em janelas de até 90 min.
5. Avaliar, gerar gráficos/screenshots e concluir documentação em pt-BR.

## Janelas GPU

| Janela | Início | Fim | Minutos | Uso | Estado final |
|---|---|---|---:|---|---|
| — | — | — | 0 | Nenhuma janela iniciada | Transcrição Running; marker ausente; FVM Running |

Total GPU desta tarefa: **0 min / limite aproximado de 300 min**.

## Bloqueios/segurança

- Nenhum bloqueio do Kaspersky, Smart App Control ou Windows Security observado.
- Não foram tocados serviços, containers, gravação, HOLD, documentos FVM/TRW ou credenciais.

