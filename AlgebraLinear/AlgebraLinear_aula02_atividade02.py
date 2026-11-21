import numpy as np
import matplotlib.pyplot as plt


def calcular_area_triangulo(x1, y1, x2, y2, x3, y3):
    matriz = np.array([
        [x1, y1, 1],
        [x2, y2, 1],
        [x3, y3, 1]
    ])
    return abs(np.linalg.det(matriz)) / 2


def plotar_triangulos(triangulos, area_total):
    fig, ax = plt.subplots(figsize=(8, 6))

    for idx, (pts, area) in enumerate(triangulos, start=1):
        xs = [pts[0][0], pts[1][0], pts[2][0], pts[0][0]]
        ys = [pts[0][1], pts[1][1], pts[2][1], pts[0][1]]

        linha, = ax.plot(xs, ys, marker="o", linewidth=2, label=f"Triângulo {idx} (Área = {area:.2f})")
        ax.fill(xs, ys, color=linha.get_color(), alpha=0.2)

    ax.set_title(f"Triângulos Informados — Área Total = {area_total:.2f}")
    ax.set_xlabel("Eixo X")
    ax.set_ylabel("Eixo Y")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.show()


def main():
    try:
        num_triangulos = int(input("Digite o número de triângulos: "))
        if num_triangulos <= 0:
            print("Erro: Informe uma quantidade maior que zero.")
            return

        area_total = 0.0
        triangulos = []

        for i in range(num_triangulos):
            print(f"\nTriângulo {i + 1}:")
            x1 = float(input("x1: "))
            y1 = float(input("y1: "))
            x2 = float(input("x2: "))
            y2 = float(input("y2: "))
            x3 = float(input("x3: "))
            y3 = float(input("y3: "))

            area = calcular_area_triangulo(x1, y1, x2, y2, x3, y3)
            area_total += area

            if np.isclose(area, 0.0):
                print(f"Aviso: Os pontos do triângulo {i + 1} são colineares (Área = 0.00).")
            else:
                print(f"Área do triângulo {i + 1}: {area:.2f}")

            triangulos.append(([(x1, y1), (x2, y2), (x3, y3)], area))

        print(f"\nÁrea total dos {num_triangulos} triângulos: {area_total:.2f}")
        plotar_triangulos(triangulos, area_total)

    except ValueError:
        print("Erro: Entrada inválida. Por favor, digite apenas números.")


if __name__ == "__main__":
    main()