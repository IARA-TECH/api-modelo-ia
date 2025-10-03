import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')


class AbacusEducationalCounterV3:
    def __init__(self, debug=False):
        """
        Sistema otimizado de contagem para ábaco educacional brasileiro
        Versão 3.0 com melhorias na detecção e lógica de agrupamento
        """
        self.debug = debug

        # Parâmetros de detecção ajustados
        self.eps_clustering = 18
        self.min_area_micanga = 50
        self.max_area_micanga = 1000
        self.min_circularity = 0.15  # Mais permissivo para miçangas ovais

        # Parâmetros para análise de movimento
        self.threshold_agrupamento = 25  # Distância para considerar miçangas agrupadas (fallback)
        self.min_micangas_grupo = 2      # Mínimo de miçangas para formar grupo

        # Valores das colunas
        self.valores_coluna = {
            'CENTENA': 100,
            'DEZENA': 10,
            'UNIDADE': 1
        }

        # Cores para cada tipo de miçanga
        self.cores_micangas = {
            'CENTENA': (255, 0, 0),    # Azul em BGR
            'DEZENA': (0, 0, 255),     # Vermelho em BGR  
            'UNIDADE': (0, 255, 255)   # Amarelo em BGR
        }

    # === FUNÇÕES DE DETECÇÃO DE MIÇANGAS ===
    def detectar_micangas_avancado(self, img):
        """
        Detecção avançada de miçangas usando múltiplas abordagens combinadas
        """
        micangas_detectadas = []
        height, width = img.shape[:2]

        # Pré-processamento da imagem
        img_blur = cv2.GaussianBlur(img, (3, 3), 0)
        hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)

        # === MÉTODO 1: Detecção por cores HSV melhorada ===
        ranges_cores = [
            ([85, 40, 40], [135, 255, 255]),     # Azul
            ([0, 80, 50], [15, 255, 255]),       # Vermelho parte 1
            ([165, 80, 50], [180, 255, 255]),    # Vermelho parte 2
            ([8, 80, 80], [45, 255, 255]),       # Amarelo/laranja
            ([35, 40, 40], [85, 255, 255]),      # Verde
            ([140, 40, 40], [170, 255, 255])     # Rosa/magenta
        ]

        mask_cores = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges_cores:
            mask = cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
            mask_cores = cv2.bitwise_or(mask_cores, mask)

        # === MÉTODO 2: Saturação e brilho ===
        s_channel = hsv[:, :, 1]
        v_channel = hsv[:, :, 2]
        _, mask_saturacao = cv2.threshold(s_channel, 70, 255, cv2.THRESH_BINARY)
        mask_brilho = cv2.inRange(v_channel, 40, 240)
        mask_sat_bri = cv2.bitwise_and(mask_saturacao, mask_brilho)

        # === COMBINAÇÃO ===
        mask_combined = cv2.bitwise_or(mask_cores, mask_sat_bri)
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel_small)
        mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, kernel_medium)

        if self.debug:
            cv2.imwrite("debug_mask_combined.jpg", mask_combined)

        # === CONTORNOS ===
        contours_info = cv2.findContours(mask_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]

        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_area_micanga < area < self.max_area_micanga:
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0

                    if (circularity > self.min_circularity and aspect_ratio < 3.0 and w > 5 and h > 5):
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            margin = 20
                            if (margin < cx < width - margin and margin < cy < height - margin):
                                micangas_detectadas.append({
                                    'x': cx, 'y': cy,
                                    'area': area,
                                    'circularity': circularity,
                                    'aspect_ratio': aspect_ratio,
                                    'contour': contour,
                                    'bbox': (x, y, w, h)  # Adicionado para bounding box
                                })

        if self.debug:
            print(f"Miçangas detectadas: {len(micangas_detectadas)}")

        return micangas_detectadas

    # === DETECÇÃO DE ESTRUTURA ===
    def detectar_estrutura_inteligente(self, img):
        """
        Detecta divisor central e seções (centenas, dezenas, unidades)
        """
        height, width = img.shape[:2]
        divisor_central = width // 2

        secoes = [
            {'nome': 'CENTENA', 'x_inicio': 0, 'x_fim': divisor_central // 3, 'cor': self.cores_micangas['CENTENA']},
            {'nome': 'DEZENA', 'x_inicio': divisor_central // 3, 'x_fim': 2 * divisor_central // 3, 'cor': self.cores_micangas['DEZENA']},
            {'nome': 'UNIDADE', 'x_inicio': 2 * divisor_central // 3, 'x_fim': divisor_central, 'cor': self.cores_micangas['UNIDADE']}
        ]

        return {
            'divisor_central': divisor_central,
            'secoes': secoes
        }

    # === AGRUPAMENTO DE MIÇANGAS ===
    def agrupar_micangas_dbscan(self, micangas, divisor_central, eps=None, min_samples=None):
        """
        Agrupamento de miçangas usando DBSCAN
        """
        if not micangas:
            return []

        if eps is None:
            eps = self.eps_clustering
        if min_samples is None:
            min_samples = 1

        X = np.array([[m['x'], m['y']] for m in micangas])
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)

        grupos = []
        for i in range(max(labels) + 1):
            grupo_idx = np.where(labels == i)[0]
            grupo_micangas = [micangas[idx] for idx in grupo_idx]
            if grupo_micangas:
                grupos.append(grupo_micangas)

        return grupos

    # === PROCESSAMENTO ===
    def processar_imagem(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return None, f"Erro: Imagem '{image_path}' não encontrada"

        estrutura = self.detectar_estrutura_inteligente(img)
        micangas = self.detectar_micangas_avancado(img)
        grupos = self.agrupar_micangas_dbscan(micangas, estrutura['divisor_central'])

        # Contagem final
        valor_total = 0
        resultados = []
        micangas_por_secao = {'CENTENA': [], 'DEZENA': [], 'UNIDADE': []}

        for secao in estrutura['secoes']:
            micangas_secao = [m for m in micangas if secao['x_inicio'] <= m['x'] < secao['x_fim']]
            n = len(micangas_secao)
            valor_secao = n * self.valores_coluna[secao['nome']]
            valor_total += valor_secao
            resultados.append(f"{n} {secao['nome']}s -> {valor_secao}")
            micangas_por_secao[secao['nome']] = micangas_secao

        resultado_str = " + ".join(resultados) + f" = {valor_total}"

        return valor_total, resultado_str, micangas_por_secao, estrutura

    # === RELATÓRIO DETALHADO ===
    def relatorio_detalhado(self, image_path):
        print("📋 RELATÓRIO DETALHADO DE ANÁLISE")
        print("=" * 50)

        img = cv2.imread(image_path)
        if img is None:
            msg = f"❌ Erro: Imagem '{image_path}' não encontrada"
            print(msg)
            return None, msg

        print(f"📁 Imagem: {image_path}")
        print(f"📐 Dimensões: {img.shape[1]}x{img.shape[0]} pixels")

        micangas = self.detectar_micangas_avancado(img)
        print(f"\n🔍 Miçangas detectadas: {len(micangas)}")

        estrutura = self.detectar_estrutura_inteligente(img)
        print(f"🏗️  Divisor central em X={estrutura['divisor_central']}")

        valor_total, resultado, micangas_por_secao, _ = self.processar_imagem(image_path)
        print(f"\n🎯 Resultado: {resultado}")

        print("=" * 50)
        return valor_total, resultado

    # === GERAR IMAGEM COM MIÇANGAS COLORIDAS ===
    def gerar_imagem_micangas_coloridas(self, image_path, output_path="micangas_coloridas.jpg"):
        """
        Gera imagem com miçangas coloridas de acordo com sua posição (centena=azul, dezena=vermelho, unidade=amarelo)
        """
        img = cv2.imread(image_path)
        if img is None:
            print("❌ Erro: Imagem não encontrada")
            return None

        # Processar imagem para obter dados
        valor_total, resultado, micangas_por_secao, estrutura = self.processar_imagem(image_path)
        
        # Criar imagem de resultado
        img_resultado = img.copy()
        height, width = img.shape[:2]
        
        # Desenhar divisórias das seções
        for secao in estrutura['secoes']:
            cv2.line(img_resultado, (secao['x_inicio'], 0), (secao['x_inicio'], height), (100, 100, 100), 1)
            # Texto da seção
            cv2.putText(img_resultado, secao['nome'], 
                       (secao['x_inicio'] + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, secao['cor'], 2)
        
        # Desenhar divisor central
        cv2.line(img_resultado, (estrutura['divisor_central'], 0), 
                (estrutura['divisor_central'], height), (0, 0, 255), 2)
        
        # Desenhar miçangas com cores conforme a seção
        contagem_total = 0
        for secao_nome, micangas_secao in micangas_por_secao.items():
            cor = self.cores_micangas[secao_nome]
            for i, micanga in enumerate(micangas_secao):
                contagem_total += 1
                
                # Desenhar contorno colorido
                cv2.drawContours(img_resultado, [micanga['contour']], -1, cor, 3)
                
                # Desenhar bounding box
                x, y, w, h = micanga['bbox']
                cv2.rectangle(img_resultado, (x, y), (x + w, y + h), cor, 2)
                
                # Numerar a miçanga
                cv2.putText(img_resultado, f'{contagem_total}', 
                           (micanga['x'] - 10, micanga['y'] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(img_resultado, f'{contagem_total}', 
                           (micanga['x'] - 10, micanga['y'] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 1)
        
        # Adicionar legenda
        y_legenda = height - 100
        for secao_nome, cor in self.cores_micangas.items():
            contagem = len(micangas_por_secao[secao_nome])
            texto = f"{secao_nome}: {contagem} miçangas"
            cv2.putText(img_resultado, texto, (10, y_legenda), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)
            y_legenda += 25
        
        # Adicionar resultado total
        cv2.putText(img_resultado, f"VALOR TOTAL: {valor_total}", 
                   (width - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Salvar imagem
        cv2.imwrite(output_path, img_resultado)
        print(f"🖼️  Imagem com miçangas coloridas salva em: {output_path}")
        
        # Mostrar imagem
        cv2.imshow("Miçangas Coloridas - Azul=Centena, Vermelho=Dezena, Amarelo=Unidade", img_resultado)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return img_resultado

    # === DEBUG ===
    def gerar_imagem_debug_completa(self, image_path, output_path="debug.jpg"):
        img = cv2.imread(image_path)
        if img is None:
            print("Erro: Imagem não encontrada para debug")
            return

        estrutura = self.detectar_estrutura_inteligente(img)
        micangas = self.detectar_micangas_avancado(img)
        grupos = self.agrupar_micangas_dbscan(micangas, estrutura['divisor_central'])

        # Desenhar divisor
        cv2.line(img, (estrutura['divisor_central'], 0),
                 (estrutura['divisor_central'], img.shape[0]), (0, 0, 255), 2)

        # Desenhar miçangas
        for m in micangas:
            cv2.circle(img, (m['x'], m['y']), 8, (0, 255, 0), 2)

        cv2.imwrite(output_path, img)
        print(f"Imagem de debug salva em {output_path}")

    # === CALIBRAÇÃO AUTOMÁTICA ===
    def calibrar_automaticamente(self, image_path, valor_esperado=None):
        print("🔧 Iniciando calibração automática...")
        img = cv2.imread(image_path)
        if img is None:
            print("❌ Erro: Imagem não encontrada")
            return

        parametros_teste = [
            {'min_area': 40, 'max_area': 800, 'eps': 15},
            {'min_area': 60, 'max_area': 1200, 'eps': 18},
            {'min_area': 80, 'max_area': 1000, 'eps': 20},
            {'min_area': 50, 'max_area': 900, 'eps': 22},
        ]

        resultados = []
        for params in parametros_teste:
            self.min_area_micanga = params['min_area']
            self.max_area_micanga = params['max_area']
            self.eps_clustering = params['eps']

            valor, resultado, _, _ = self.processar_imagem(image_path)
            resultados.append({
                'params': params,
                'valor_total': valor,
                'resultado': resultado
            })

        print("\n📊 Resultados da calibração:")
        for r in resultados:
            print(f"{r['params']} -> {r['valor_total']}")

        if valor_esperado is not None:
            melhor = min(resultados, key=lambda r: abs((r['valor_total'] or 0) - valor_esperado))
            print("\n✅ Melhor configuração encontrada:", melhor)
            return melhor
        return resultados


# --- Classe de compatibilidade (versão original simulada) ---
class AbacusEducationalCounter:
    """Classe simples para comparar com a V3: usa a mesma lógica básica."""
    def __init__(self, debug=False):
        self.impl = AbacusEducationalCounterV3(debug=debug)

    def processar_imagem(self, image_path):
        return self.impl.processar_imagem(image_path)


# Exemplo de execução
if __name__ == "__main__":
    def executar_exemplo_completo():
        print("🎯 === SISTEMA DE CONTAGEM DE ÁBACO COM CORES ===\n")
        contador = AbacusEducationalCounterV3(debug=True)
        imagem = "modelos/contagem/24.jpg"  # Altere para o caminho da sua imagem

        try:
            print("🔧 ETAPA 1: Calibração automática")
            contador.calibrar_automaticamente(imagem)

            print("\n🔍 ETAPA 2: Análise detalhada")
            valor, resultado = contador.relatorio_detalhado(imagem)

            print("\n🎨 ETAPA 3: Gerando imagem com miçangas coloridas")
            img_colorida = contador.gerar_imagem_micangas_coloridas(imagem, "micangas_coloridas_resultado.jpg")

            print("\n✨ RESULTADO FINAL:")
            print("=" * 60)
            print(f"🔢 VALOR CALCULADO: {valor}")
            print(f"📊 DETALHAMENTO: {resultado}")
            print("🎨 CORES: Azul=Centena, Vermelho=Dezena, Amarelo=Unidade")
            print("=" * 60)

        except Exception as e:
            print(f"❌ Erro durante execução: {e}")
            import traceback
            traceback.print_exc()

    # Executar o exemplo completo
    executar_exemplo_completo()