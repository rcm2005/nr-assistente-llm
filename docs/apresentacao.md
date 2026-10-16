# Roteiro de apresentação (5–7 minutos)

1. **Problema:** perguntas em português sobre sete NRs, com citação do item e recusa quando a
   fonte não sustenta a resposta. Mostrar o recorte e o aviso de uso acadêmico.
2. **Modelo e GPU:** Qwen2.5-0.5B-Instruct, licença Apache-2.0, LoRA e RTX 4060 Ti de 8 GB.
   Abrir [gpu.md](gpu.md) e a captura do VSCode, se disponível.
3. **Dados:** PDFs oficiais do MTE, 720 pares sintéticos; 600 de treino, 120 retidos para teste;
   26 perguntas de recusa no teste. Explicar a separação por item.
4. **Experimento:** mostrar a tabela dos quatro runs no [relatório](relatorio.md): ranks 8/32,
   taxas 1e-4/3e-4 e 1/2 épocas. Todos os treinos foram executados na GPU.
5. **Resultados:** comparar o modelo base e os quatro adapters nos mesmos 120 casos. O melhor
   compromisso recuperou 22/26 recusas, mas acertou só 1/94 citações respondíveis. Mostrar
   o gráfico e um exemplo de resposta incorreta.
6. **Demonstração:** iniciar `python app.py`, digitar uma pergunta e trocar o modelo ou a
   temperatura. Conferir a resposta com a NR oficial.
7. **Conclusão:** o ajuste melhorou semelhança e recusa, mas a confiabilidade da citação ainda
   é insuficiente. Propor recuperação de texto oficial e validação de citações como próximos passos.
