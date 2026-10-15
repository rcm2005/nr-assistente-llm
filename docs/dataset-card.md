# Cartão do dataset

Gerado automaticamente em 2026-10-15T23:34:42.903817+00:00 a partir dos arquivos locais.

## Origem e escopo

Trechos dos PDFs públicos oficiais das Normas Regulamentadoras do MTE. SHA-256 do manifesto de fontes: `11ddc1ec0cbc8c9ab6c3fa1a6b9069069d23f7e59c8ecaf7663468d1a80823a7`.

## Composição medida

O Gemini `gemini-3.8-flash-medium` produziu 720 saídas em 40 lotes, com até 2 chamadas simultâneas. A validação aceitou 720 e rejeitou 0.

| Split | Pares | Recusas esperadas | Itens normativos distintos |
|---|---:|---:|---:|
| Treino | 600 | 118 | 600 |
| Teste | 120 | 26 | 120 |

| NR | Treino | Teste |
|---:|---:|---:|
| 06 | 33 | 4 |
| 10 | 112 | 21 |
| 12 | 145 | 32 |
| 17 | 52 | 11 |
| 18 | 144 | 32 |
| 33 | 65 | 14 |
| 35 | 49 | 6 |

## Metodologia

Os pares foram gerados pelo `gemini-3.8-flash-medium` via `agy`, entre `2026-10-15T23:11:52.868609+00:00` e `2026-10-15T23:34:10.095490+00:00`. Foram 55 tentativas no total (15 repetições após saída inválida ou falha). Depois, os dados foram validados localmente. Exemplos comuns exigem a citação exata do item-fonte; armadilhas exigem recusa e não podem conter citação. O split é feito por `source_id`, portanto um mesmo item normativo não aparece simultaneamente em treino e teste.

## Limitações

Dados sintéticos podem conter formulações pouco naturais ou inferências imperfeitas. A validação local confirma esquema, identidade da fonte, duplicatas, recusas e formato de citação, mas não substitui revisão jurídica humana. O assistente não oferece parecer legal.
