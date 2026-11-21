import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# CONEXÃO TEÓRICA E ESTRUTURAL (ÁLGEBRA LINEAR):
# A área de um triângulo no R² pode ser calculada via determinante de uma matriz 3x3.
# Se det(M) = 0, os 3 pontos estão na mesma linha (colineares), formando um 
# "triângulo degenerado" (área zero).
# ==============================================================================

def calcular_area_triangulo(x1, y1, x2, y2, x3, y3):
    """
    Função dedicada exclusivamente ao cálculo matemático.
    Recebe as coordenadas dos 3 vértices e retorna a área e o status de colinearidade.
    """
    # 1. Montagem da Matriz das Coordenadas Ampliada:
    # Agrupamos as coordenadas (x, y) e adicionamos a terceira coluna com 1s 
    # para permitir o cálculo do determinante em 3D correspondente à área no plano 2D.
    matriz = np.array([
        [x1, y1, 1],
        [x2, y2, 1],
    [x3, y3, 1]
    ])
    
    # 2. Cálculo do Determinante (Álgebra Linear):
    # Utiliza a biblioteca NumPy (np.linalg.det) para calcular o determinante da matriz.
    det = np.linalg.det(matriz)
    
    # 3. Cálculo da Área Geométrica:
    # Pela geometria analítica, Área = 1/2 * |det(M)|. Usa-se abs() para garantir valor positivo.
    area = abs(det) / 2
    
    # 4. Verificação de Degeneração (Colinearidade):
    # np.isclose é usado em vez de 'area == 0' devido a pequenas imprecisões de ponto flutuante em computadores.
    # Conecta a matemática teórica (det=0) com o tratamento computacional seguro.
    degenerado = np.isclose(area, 0)
    
    # Retorna uma tupla contendo o valor numérico da área e a flag booleana (True/False)
    return area, degenerado

def main():
    """
    Função principal de controle da execução.
    Faz a ponte entre a interface do usuário (Input), o processamento (calcular_area_triangulo)
    e a saída gráfica/visual (Matplotlib).
    """
    try:
        # --- ETAPA 1: ENTRADA DE DADOS E VALIDAÇÃO INICIAL ---
        # Solicita a quantidade de triângulos para definir quantas vezes o loop irá rodar.
        num_str = input("Digite o número de triângulos: ")
        num = int(num_str)
        
        # Validação de regra de negócio: não podemos processar quantidade <= 0
        if num <= 0:
            print("Por favor, digite um número maior que zero.")
            return

        # Acumulador que somará a área de todos os triângulos processados
        area_total = 0.0

        # --- ETAPA 2: LOOP DE PROCESSAMENTO INDIVIDUAL ---
        # Itera pelo número total de triângulos informado pelo usuário
        for i in range(num):
            print(f"\nTriângulo {i + 1}:")
            
            # Coleta das coordenadas dos 3 vértices do triângulo 'i'
            # Usa list comprehension para ler x1, x2, x3 e y1, y2, y3
            print("Digite as coordenadas dos vértices (x1, x2, x3):")
            x = [float(input(f"x{j+1}: ")) for j in range(3)]
            print("Digite as coordenadas dos vértices (y1, y2, y3):")
            y = [float(input(f"y{j+1}: ")) for j in range(3)]

            # --- ETAPA 3: CONEXÃO COM A FUNÇÃO DE CÁLCULO ---
            # Desempacota a lista de coordenadas x e y e envia para a função de cálculo.
            # Recebe de volta a área calculada e se o triângulo é degenerado.
            area, degenerado = calcular_area_triangulo(x[0], y[0], x[1], y[1], x[2], y[2])

            # Trata o aviso caso os pontos sejam colineares
            if degenerado:
                print(f"Aviso: Triângulo {i+1} é degenerado (os pontos são colineares).")

            # Acumula o valor na área total global
            area_total += area
            print(f"Área do triângulo {i + 1}: {area:.2f}")

            # --- ETAPA 4: PROCESSAMENTO VISUAL E GRÁFICO ---
            # Conecta os dados calculados com a exibição visual usando Matplotlib.
            plt.figure() # Cria uma janela/figura separada para este triângulo
            
            # np.column_stack((x, y)) une os vetores X e Y em pares (x,y) de coordenadas 2D.
            # plt.Polygon desenha o triângulo preenchido usando essas coordenadas.
            tri = plt.Polygon(np.column_stack((x, y)), fill=True, alpha=0.3, label=f"T{i+1}")
            plt.gca().add_patch(tri) # Adiciona o polígono ao eixe atual do gráfico
            
            # Plota os vértices individuais destacados como pontos vermelhos
            plt.scatter(x, y, color='red')
            plt.title(f"Triângulo {i+1} - Área: {area:.2f}")
            plt.grid(True)
            
            # Opcional: plt.show() exibiria a janela. Mantido como log visual.
            print("Gráfico processado (visualização omitida).")

        # --- ETAPA 5: EXIBIÇÃO DOS RESULTADOS FINAIS ---
        print(f"\nÁrea total dos {num} triângulos: {area_total:.2f}")

    except ValueError:
        # Captura erros caso o usuário digite texto/letras onde eram esperados números
        print("Erro: Entrada inválida. Por favor, digite números.")

# --- PONTO DE ENTRADA DO PROGRAMA ---
# Garante que a função main() só seja executada quando o script for rodado diretamente
if __name__ == "__main__":
    main()
