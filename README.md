# TrabalhoAnaliseNumerica
Codigos para o trabalho de analise numerica focado em utilizar python e os conhecimentos abordados em aula para resolver o problema da condutividade termica transformando as equaçoes de laplace em sistemas nao lineares

   * Método de Broyden;
5. **Validação:** Comparar a solução numérica com a solução exata $T(x, y) = \sin(\pi x)\sin(\pi y)$ usando a norma do erro máximo ($L_\infty$).
6. **Análise de Convergência:** Analisar a convergência com diferentes tamanhos de malha $N \in \{4, 8, 16, 32\}$ e preencher a tabela de erros.
7. **Desempenho:** Comparar os métodos em termos de número de iterações, tempo de execução e precisão da solução.
8. **Ordem de Convergência:** Verificar se o erro cai como $\mathcal{O}(h^2)$, esperado para diferenças finitas de segunda ordem.
9. **Diferencial:** Implementação sem o uso de bibliotecas de álgebra linear prontas (`numpy.linalg`, `scipy.linalg`, etc.).

---

## Tabela de Erros para Diferentes Malhas

Preencha a tabela abaixo com os erros obtidos por cada método para cada valor de $N$. O erro deve ser medido na norma $L_\infty$ (erro máximo sobre todos os nós internos):

| $N$ | $h$ | Erro (Newton + LU) | Erro (Broyden) | Erro (Newton + CG) | Ordem Estimada ($p$) |
|:---:|:---:|:------------------:|:--------------:|:------------------:|:--------------------:|
|  4  | 1/5 |         ?          |       ?        |         ?          |          -           |
|  8  | 1/9 |         ?          |       ?        |         ?          |          ?           |
| 16  | 1/17|         ?          |       ?        |         ?          |          ?           |
| 32  | 1/33|         ?          |       ?        |         ?          |          ?           |

*A ordem de convergência estimada é calculada como: $p  pprox \log_2(e_N / e_{2N})$, onde $e_N$ é o erro para a malha de tamanho $N$.*

---

## Entrega

* **Código-fonte:** Comentado e estruturado de forma explicativa.
* **Relatório em PDF:** Utilizando o template disponibilizado, contendo:
  * Descrição dos métodos e da discretização;
  * Gráficos da solução numérica e do erro pontual para $N = 16$;
  * Tabela de erros preenchida para $N \in \{4, 8, 16, 32\}$;
  * Gráfico da norma do resíduo $\|F(U^k)\|$ por iteração para cada método;
  * Conclusão sobre a precisão e eficiência de cada método.