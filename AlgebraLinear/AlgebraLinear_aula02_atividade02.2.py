import numpy as np
import matplotlib.pyplot as plt


def resolver_cramer(matriz_a, vetor_b):
    det_a = np.linalg.det(matriz_a)
    if np.isclose(det_a, 0.0):
        raise ValueError("O determinante da matriz dos coeficientes é zero (sistema sem solução única).")

    n = len(vetor_b)
    solucao = []
    for j in range(n):
        matriz_j = matriz_a.copy()
        matriz_j[:, j] = vetor_b
        det_j = np.linalg.det(matriz_j)
        solucao.append(det_j / det_a)

    return solucao


def regressao_linear_simples(xs, ys):
    n = len(xs)
    sx = np.sum(xs)
    sy = np.sum(ys)
    sx2 = np.sum(xs ** 2)
    sxy = np.sum(xs * ys)

    matriz_a = np.array([
        [n, sx],
        [sx, sx2]
    ], dtype=float)
    vetor_b = np.array([sy, sxy], dtype=float)

    a, b = resolver_cramer(matriz_a, vetor_b)
    return a, b


def regressao_quadratica(xs, ys):
    n = len(xs)
    sx = np.sum(xs)
    sx2 = np.sum(xs ** 2)
    sx3 = np.sum(xs ** 3)
    sx4 = np.sum(xs ** 4)
    sy = np.sum(ys)
    sxy = np.sum(xs * ys)
    sx2y = np.sum((xs ** 2) * ys)

    matriz_a = np.array([
        [n, sx, sx2],
        [sx, sx2, sx3],
        [sx2, sx3, sx4]
    ], dtype=float)
    vetor_b = np.array([sy, sxy, sx2y], dtype=float)

    divisores = [sx2, sx3, sx4]
    for i in range(3):
        if not np.isclose(divisores[i], 0.0):
            matriz_a[i] = matriz_a[i] / divisores[i]
            vetor_b[i] = vetor_b[i] / divisores[i]

    a, b, c = resolver_cramer(matriz_a, vetor_b)
    return a, b, c


def regressao_exponencial(xs, ys):
    for y in ys:
        if y <= 0:
            raise ValueError("A regressão exponencial exige todos os valores de y > 0 (para calcular ln(y)).")

    n = len(xs)
    ln_y = np.log(ys)
    sx = np.sum(xs)
    sx2 = np.sum(xs ** 2)
    slny = np.sum(ln_y)
    sxlny = np.sum(xs * ln_y)

    matriz_a = np.array([
        [n, sx],
        [sx, sx2]
    ], dtype=float)
    vetor_b = np.array([slny, sxlny], dtype=float)

    A, b = resolver_cramer(matriz_a, vetor_b)
    a = np.exp(A)
    return a, b


def calcular_modelo(opcao, x, a, b, c=0.0):
    if opcao == "1":
        return a + b * x
    elif opcao == "2":
        return a + b * x + c * (x ** 2)
    else:
        return a * np.exp(b * x)


def plotar_pontos(xs, ys, nome_funcao, var_indep):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(xs, ys, color="red", s=60, zorder=3, label="Pontos informados")
    ax.set_title(f"Pontos Experimentais — {nome_funcao}({var_indep})")
    ax.set_xlabel(var_indep)
    ax.set_ylabel(f"{nome_funcao}({var_indep})")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plotar_regressao(xs, ys, opcao, a, b, c, equacao_str, nome_funcao, var_indep):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(xs, ys, color="red", s=60, zorder=3, label="Pontos informados")

    x_min, x_max = np.min(xs), np.max(xs)
    margem = (x_max - x_min) * 0.1 if x_max != x_min else 1.0
    x_curva = np.linspace(x_min - margem, x_max + margem, 300)
    y_curva = calcular_modelo(opcao, x_curva, a, b, c)

    ax.plot(x_curva, y_curva, color="blue", linewidth=2, label=f"Regressão: {equacao_str}")
    ax.set_title(f"Ajuste por Regressão — {equacao_str}")
    ax.set_xlabel(var_indep)
    ax.set_ylabel(f"{nome_funcao}({var_indep})")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.show()


def main():
    try:
        num_pontos = int(input("Digite o número de pontos: "))
        if num_pontos < 2:
            print("Erro: Informe pelo menos 2 pontos.")
            return

        nome_funcao = input("Digite o nome da função (ex: f, T, n): ").strip() or "f"
        var_indep = input("Digite o nome da variável independente (ex: x, h, d): ").strip() or "x"

        xs = []
        ys = []
        for i in range(num_pontos):
            print(f"\nPonto {i + 1}:")
            x_val = float(input(f"{var_indep}{i + 1}: "))
            y_val = float(input(f"y{i + 1}: "))
            xs.append(x_val)
            ys.append(y_val)

        xs = np.array(xs, dtype=float)
        ys = np.array(ys, dtype=float)

        plotar_pontos(xs, ys, nome_funcao, var_indep)

        print("\nOpções de Regressão Linear:")
        print("1 - Linear Simples")
        print("2 - Regressão Quadrática")
        print("3 - Regressão Exponencial")
        opcao = input("Escolha uma opção (1, 2 ou 3): ").strip()

        c = 0.0
        if opcao == "1":
            a, b = regressao_linear_simples(xs, ys)
            sinal_b = "+" if b >= 0 else "-"
            equacao_str = f"{nome_funcao}({var_indep}) = {a:.4f} {sinal_b} {abs(b):.4f}{var_indep}"

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

        elif opcao == "3":
            a, b = regressao_exponencial(xs, ys)
            equacao_str = f"{nome_funcao}({var_indep}) = {a:.4f} * e^({b:.4f}{var_indep})"

        else:
            print("Erro: Opção inválida.")
            return

        print(f"\nEquação encontrada:\n{equacao_str}")
        plotar_regressao(xs, ys, opcao, a, b, c, equacao_str, nome_funcao, var_indep)

        while True:
            usar = input(f"\nDeseja calcular {nome_funcao}({var_indep}) para um valor de {var_indep}? (s/n): ").strip().lower()
            if usar != "s":
                break
            valor_x = float(input(f"Digite o valor de {var_indep}: "))
            resultado = calcular_modelo(opcao, valor_x, a, b, c)
            print(f"{nome_funcao}({valor_x}) = {resultado:.4f}")

    except ValueError as erro:
        print(f"Erro: {erro}")


if __name__ == "__main__":
    main()