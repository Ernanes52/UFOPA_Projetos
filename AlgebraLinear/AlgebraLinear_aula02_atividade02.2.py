# ==============================================================================
# PROGRAMA 3 — REGRESSÃO LINEAR VIA SISTEMAS LINEARES E REGRA DE CRAMER
# Disciplina: Álgebra Linear (UFOPA)
#
# VISÃO GERAL DE COMO O CÓDIGO SE CONECTA (ROTEIRO DA APRESENTAÇÃO):
# 1. ENTRADA (main): Coleta os pontos (x, y) e exibe o gráfico inicial.
# 2. MODELAGEM (funções de regressão): Transforma os pontos em somatórios e
#    monta um Sistema de Equações Lineares na forma matricial A * X = B.
# 3. RESOLUÇÃO (resolver_cramer): Resolve o sistema A * X = B usando
#    determinantes (Regra de Cramer) para descobrir os coeficientes (a, b, c).
# 4. SAÍDA E PREVISÃO (plotar_regressao + lambda modelo): Monta a equação final,
#    desenha a curva ajustada sobre os pontos e permite calcular novos valores.
# ==============================================================================

import numpy as np               # Biblioteca para vetores, matrizes e cálculo de determinantes
import matplotlib.pyplot as plt  # Biblioteca para desenhar os gráficos dos pontos e das curvas


# ==============================================================================
# BLOCO 1: O MOTOR MATEMÁTICO (REGRA DE CRAMER)
# Conexão: É chamado por TODAS as três funções de regressão para resolver AX = B.
# ==============================================================================
def resolver_cramer(matriz_a, vetor_b):
    """
    Resolve um sistema linear AX = B utilizando a Regra de Cramer:
    x_j = det(A_j) / det(A)
    """
    # Passo 1.1: Calcula o determinante da matriz principal dos coeficientes (det(A)).
    det_a = np.linalg.det(matriz_a)

    # Verifica se det(A) é zero (ou muito próximo de zero por arredondamento de ponto flutuante).
    # Se det(A) == 0, o sistema não possui solução única (não dá para dividir por zero).
    if np.isclose(det_a, 0.0):
        raise ValueError("O determinante da matriz dos coeficientes é zero (sistema sem solução única).")

    n = len(vetor_b)  # Descobre quantas incógnitas existem (2 na simples/exp, 3 na quadrática)
    solucao = []      # Lista que vai guardar os coeficientes encontrados [a, b] ou [a, b, c]

    # Passo 1.2: Para cada incógnita j (coluna 0, coluna 1, etc.):
    for j in range(n):
        matriz_j = matriz_a.copy()     # Cria uma cópia de A para não alterar a matriz original
        matriz_j[:, j] = vetor_b       # Substitui toda a coluna j pelo vetor B dos termos independentes
        det_j = np.linalg.det(matriz_j)# Calcula o determinante dessa nova matriz modificada (det_j)
        solucao.append(det_j / det_a)  # Aplica a fórmula de Cramer: x_j = det_j / det(A)

    return solucao  # Devolve os coeficientes calculados para a função de regressão que chamou


# ==============================================================================
# BLOCO 2: OS TRÊS MODELOS DE REGRESSÃO (MONTAGEM DOS SISTEMAS AX = B)
# Conexão: Recebem os vetores xs e ys do main(), montam A e B com somatórios
#          e enviam para resolver_cramer() achar os coeficientes.
# ==============================================================================
def regressao_linear_simples(xs, ys):
    """
    Ajusta uma reta: f(x) = a + bx
    Monta e resolve o sistema linear 2x2:
      | n      sum(x)   | * | a | = | sum(y)   |
      | sum(x) sum(x^2) |   | b |   | sum(x*y) |
    """
    n = len(xs)                # Quantidade de pontos
    sx = np.sum(xs)            # Somatório de x
    sy = np.sum(ys)            # Somatório de y
    sx2 = np.sum(xs ** 2)      # Somatório de x²
    sxy = np.sum(xs * ys)      # Somatório de (x * y)

    # Monta a matriz A (2x2) dos coeficientes e o vetor B (2x1) dos resultados
    matriz_a = np.array([
        [n, sx],
        [sx, sx2]
    ], dtype=float)
    vetor_b = np.array([sy, sxy], dtype=float)

    # Envia A e B para Cramer e recebe os valores exatos de 'a' e 'b'
    a, b = resolver_cramer(matriz_a, vetor_b)
    return a, b


def regressao_quadratica(xs, ys):
    """
    Ajusta uma parábola: f(x) = a + bx + cx^2
    Monta o sistema linear 3x3 e aplica normalização nas linhas para evitar overflow.
    """
    n = len(xs)                     # Quantidade de pontos
    sx = np.sum(xs)                 # Somatório de x
    sx2 = np.sum(xs ** 2)           # Somatório de x²
    sx3 = np.sum(xs ** 3)           # Somatório de x³
    sx4 = np.sum(xs ** 4)           # Somatório de x⁴
    sy = np.sum(ys)                 # Somatório de y
    sxy = np.sum(xs * ys)           # Somatório de (x * y)
    sx2y = np.sum((xs ** 2) * ys)   # Somatório de (x² * y)

    # Monta o sistema 3x3 original antes da normalização
    matriz_a = np.array([
        [n,   sx,  sx2],
        [sx,  sx2, sx3],
        [sx2, sx3, sx4]
    ], dtype=float)
    vetor_b = np.array([sy, sxy, sx2y], dtype=float)

    # PONTO IMPORTANTE PARA A APRESENTAÇÃO (Prevenção de Overflow):
    # Como x^3 e x^4 crescem muito rápido, o determinante pode estourar (overflow).
    # Conforme visto em aula, dividimos a 1ª linha por sum(x^2), a 2ª por sum(x^3)
    # e a 3ª por sum(x^4), deixando a última coluna igual a 1 sem alterar a solução.
    divisores = [sx2, sx3, sx4]
    for i, div in enumerate(divisores):
        if not np.isclose(div, 0.0):
            matriz_a[i] = matriz_a[i] / div
            vetor_b[i] = vetor_b[i] / div

    # Envia o sistema normalizado para Cramer e recebe 'a', 'b' e 'c'
    a, b, c = resolver_cramer(matriz_a, vetor_b)
    return a, b, c


def regressao_exponencial(xs, ys):
    """
    Ajusta uma curva exponencial: f(x) = a * e^(bx)
    Estratégia: Aplica logaritmo natural ln(y) para transformar a curva em uma reta:
    ln(y) = ln(a) + b*x  =>  Y = A + b*x (onde A = ln(a), logo a = e^A).
    """
    # Não existe logaritmo real de zero ou número negativo; valida antes de calcular
    if np.any(ys <= 0):
        raise ValueError("A regressão exponencial exige todos os valores de y > 0 (para calcular ln(y)).")

    n = len(xs)
    ln_y = np.log(ys)             # Lineariza os valores de y aplicando ln(y)
    sx = np.sum(xs)               # Somatório de x
    sx2 = np.sum(xs ** 2)         # Somatório de x²
    slny = np.sum(ln_y)           # Somatório de ln(y)
    sxlny = np.sum(xs * ln_y)     # Somatório de (x * ln(y))

    # Monta o mesmo sistema 2x2 da regressão simples, trocando y por ln(y)
    matriz_a = np.array([
        [n, sx],
        [sx, sx2]
    ], dtype=float)
    vetor_b = np.array([slny, sxlny], dtype=float)

    # Resolve via Cramer para achar A (que é ln(a)) e b
    A, b = resolver_cramer(matriz_a, vetor_b)
    a = np.exp(A)  # Reverte o logaritmo usando a função exponencial (a = e^A)
    return a, b


# ==============================================================================
# BLOCO 3: VISUALIZAÇÃO GRÁFICA (MATPLOTLIB)
# Conexão: plotar_pontos() mostra os dados brutos antes da escolha do modelo;
#          plotar_regressao() desenha a curva matemática sobre esses pontos.
# ==============================================================================
def plotar_pontos(xs, ys, nome_funcao, var_indep):
    """
    Exibe apenas os pontos experimentais digitados pelo usuário.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(xs, ys, color="red", s=60, zorder=3, label="Pontos informados")
    ax.set_title(f"Pontos Experimentais — {nome_funcao}({var_indep})")
    ax.set_xlabel(var_indep)
    ax.set_ylabel(f"{nome_funcao}({var_indep})")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plotar_regressao(xs, ys, funcao_modelo, equacao_str, nome_funcao, var_indep):
    """
    Exibe os pontos originais (em vermelho) junto com a curva ajustada (em azul).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(xs, ys, color="red", s=60, zorder=3, label="Pontos informados")

    # Cria 300 pontos intermediários entre o menor e o maior x (com margem de 10%)
    # para que a linha/curva fique suave no gráfico, mesmo na quadrática e exponencial.
    x_min, x_max = np.min(xs), np.max(xs)
    margem = (x_max - x_min) * 0.1 if x_max != x_min else 1.0
    x_curva = np.linspace(x_min - margem, x_max + margem, 300)

    # Usa a função matemática encontrada (modelo) para calcular o y de cada um dos 300 pontos
    y_curva = funcao_modelo(x_curva)

    ax.plot(x_curva, y_curva, color="blue", linewidth=2, label=f"Regressão: {equacao_str}")
    ax.set_title(f"Ajuste por Regressão — {equacao_str}")
    ax.set_xlabel(var_indep)
    ax.set_ylabel(f"{nome_funcao}({var_indep})")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.show()


# ==============================================================================
# BLOCO 4: ORQUESTRADOR PRINCIPAL (INTERFACE E FLUXO DO PROGRAMA)
# Conexão: Une entrada de dados -> gráficos -> cálculo dos coeficientes -> previsão.
# ==============================================================================
def main():
    try:
        # 1. Coleta a quantidade de pontos e valida o mínimo necessário (2 pontos)
        num_pontos = int(input("Digite o número de pontos: "))
        if num_pontos < 2:
            print("Erro: Informe pelo menos 2 pontos.")
            return

        # Personaliza os rótulos da equação e dos eixos conforme o problema (ex: T(h) ou f(x))
        nome_funcao = input("Digite o nome da função (ex: f, T, n): ").strip() or "f"
        var_indep = input("Digite o nome da variável independente (ex: x, h, d): ").strip() or "x"

        # 2. Lê todos os pares ordenados (x, y) digitados pelo usuário
        xs = []
        ys = []
        for i in range(num_pontos):
            print(f"\nPonto {i + 1}:")
            x_val = float(input(f"{var_indep}{i + 1}: "))
            y_val = float(input(f"y{i + 1}: "))
            xs.append(x_val)
            ys.append(y_val)

        # Converte as listas Python para arrays NumPy (permite operações vetoriais como xs ** 2)
        xs = np.array(xs, dtype=float)
        ys = np.array(ys, dtype=float)

        # 3. Primeira visualização: mostra apenas os pontos para o usuário observar a tendência
        plotar_pontos(xs, ys, nome_funcao, var_indep)

        # 4. Menu de escolha do tipo de regressão
        print("\nOpções de Regressão Linear:")
        print("1 - Linear Simples")
        print("2 - Regressão Quadrática")
        print("3 - Regressão Exponencial")
        opcao = input("Escolha uma opção (1, 2 ou 3): ").strip()

        # 5. Chama a função escolhida, formata o texto da equação e cria a função 'modelo'
        #    Nota: 'modelo' guarda a fórmula matemática pronta para desenhar o gráfico e fazer previsões.
        if opcao == "1":
            a, b = regressao_linear_simples(xs, ys)
            sinal_b = "+" if b >= 0 else "-"
            equacao_str = f"{nome_funcao}({var_indep}) = {a:.4f} {sinal_b} {abs(b):.4f}{var_indep}"
            modelo = lambda x: a + b * x

        elif opcao == "2":
            if num_pontos < 3:
                print("Erro: A regressão quadrática exige pelo menos 3 pontos.")
                return
            a, b, c = regressao_quadratica(xs, ys)
            sinal_b = "+" if b >= 0 else "-"
            sinal_c = "+" if c >= 0 else "-"
            equacao_str = (
                f"{nome_funcao}({var_indep}) = {a:.4f} {sinal_b} {abs(b):.4f}{var_indep} "
                f"{sinal_c} {abs(c):.4f}{var_indep}²"
            )
            modelo = lambda x: a + b * x + c * (x ** 2)

        elif opcao == "3":
            a, b = regressao_exponencial(xs, ys)
            equacao_str = f"{nome_funcao}({var_indep}) = {a:.4f} * e^({b:.4f}{var_indep})"
            modelo = lambda x: a * np.exp(b * x)

        else:
            print("Erro: Opção inválida.")
            return

        # 6. Exibe a equação final no terminal e plota os pontos junto com a curva ajustada
        print(f"\nEquação encontrada:\n{equacao_str}")
        plotar_regressao(xs, ys, modelo, equacao_str, nome_funcao, var_indep)

        # 7. Fase de previsão: permite testar o modelo encontrado com novos valores de entrada
        while True:
            usar = input(f"\nDeseja calcular {nome_funcao}({var_indep}) para um valor de {var_indep}? (s/n): ").strip().lower()
            if usar != "s":
                break
            valor_x = float(input(f"Digite o valor de {var_indep}: "))
            resultado = modelo(valor_x)  # Reutiliza a mesma função 'modelo' usada no gráfico
            print(f"{nome_funcao}({valor_x}) = {resultado:.4f}")

    # Captura erros de digitação (letras no lugar de números) ou erros matemáticos (det == 0, y <= 0)
    except ValueError as erro:
        print(f"Erro: {erro}")


# Ponto de entrada padrão em Python: garante que main() só rode quando este arquivo for executado diretamente
if __name__ == "__main__":
    main()