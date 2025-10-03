import cv2
import numpy as np
import matplotlib.pyplot as plt
import json

class RobustAbacusCounter:
    def __init__(self):
        self.color_values = {'blue': 100, 'red': 10, 'yellow': 1}
    
    def load_image(self, image_path):
        """Carrega e prepara a imagem"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Não foi possível carregar a imagem")
        return image

    def enhance_colors(self, image):
        """Realça as cores para melhor detecção"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], 1.2)  # Aumenta saturação
        hsv[:, :, 2] = cv2.multiply(hsv[:, :, 2], 1.1)  # Aumenta brilho
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def detect_circles(self, image, color_name):
        """Detecta círculos coloridos usando múltiplas abordagens"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Definir máscaras de cor
        if color_name == 'red':
            mask1 = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255]))
            mask2 = cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
            mask = cv2.bitwise_or(mask1, mask2)
        elif color_name == 'blue':
            mask = cv2.inRange(hsv, np.array([100, 120, 70]), np.array([130, 255, 255]))
        elif color_name == 'yellow':
            mask = cv2.inRange(hsv, np.array([20, 120, 70]), np.array([30, 255, 255]))
        else:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # Limpeza da máscara
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Encontrar contornos
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        circles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 5000:  # Filtro de tamanho
                # Verificar circularidade
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.6:  # Forma circular
                        # Encontrar centro
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            circles.append({
                                'center': (cx, cy),
                                'contour': contour,
                                'area': area
                            })
        
        return circles

    def analyze_spatial_distribution(self, circles):
        """Analisa a distribuição espacial para identificar grupos"""
        if len(circles) < 2:
            return circles, []  # Todas são separadas se há poucas

        centers = [circle['center'] for circle in circles]
        
        # Calcular matriz de distâncias
        distances = []
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                dist = np.sqrt((centers[i][0] - centers[j][0])**2 + 
                              (centers[i][1] - centers[j][1])**2)
                distances.append(dist)
        
        if not distances:
            return circles, []

        # Usar threshold adaptativo
        median_dist = np.median(distances)
        threshold = median_dist * 0.6  # Threshold adaptativo

        # Identificar componentes conectados
        visited = set()
        clusters = []
        
        for i in range(len(centers)):
            if i not in visited:
                cluster = [i]
                queue = [i]
                visited.add(i)
                
                while queue:
                    current = queue.pop(0)
                    for j in range(len(centers)):
                        if j not in visited:
                            dist = np.sqrt((centers[current][0] - centers[j][0])**2 + 
                                          (centers[current][1] - centers[j][1])**2)
                            if dist < threshold:
                                cluster.append(j)
                                queue.append(j)
                                visited.add(j)
                
                clusters.append(cluster)
        
        # O maior cluster é considerado o amontoado
        if clusters:
            main_cluster = max(clusters, key=len)
            separated_indices = [i for i in range(len(circles)) if i not in main_cluster]
            
            separated = [circles[i] for i in separated_indices]
            clustered = [circles[i] for i in main_cluster]
            
            return separated, clustered
        else:
            return circles, []

    def process_abacus(self, image_path):
        """Processa a imagem completa do ábaco"""
        print("🔄 Iniciando análise do ábaco...")
        
        # Carregar e melhorar imagem
        image = self.load_image(image_path)
        enhanced_image = self.enhance_colors(image)
        debug_image = image.copy()
        
        results = {
            'blue': {'detectadas': 0, 'contabilizadas': 0, 'valor': 0},
            'red': {'detectadas': 0, 'contabilizadas': 0, 'valor': 0},
            'yellow': {'detectadas': 0, 'contabilizadas': 0, 'valor': 0},
            'total_geral': 0
        }
        
        # Processar cada cor
        for color_name in ['red', 'blue', 'yellow']:
            print(f"\n🎯 Analisando miçangas {color_name}...")
            
            # Detectar círculos
            circles = self.detect_circles(enhanced_image, color_name)
            results[color_name]['detectadas'] = len(circles)
            
            if circles:
                print(f"   ✅ Detectadas: {len(circles)} miçangas")
                
                # Analisar distribuição espacial
                separated, clustered = self.analyze_spatial_distribution(circles)
                results[color_name]['contabilizadas'] = len(separated)
                results[color_name]['valor'] = len(separated) * self.color_values[color_name]
                
                print(f"   📊 Contabilizadas: {len(separated)} miçangas")
                
                # Marcar na imagem de debug
                color_bgr = {'red': (0, 0, 255), 'blue': (255, 0, 0), 'yellow': (0, 255, 255)}
                
                # Amontoado em vermelho
                for circle in clustered:
                    center = circle['center']
                    cv2.drawContours(debug_image, [circle['contour']], -1, (0, 0, 255), 3)
                    cv2.putText(debug_image, "X", (center[0]-10, center[1]+10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Separadas em verde
                for circle in separated:
                    center = circle['center']
                    cv2.drawContours(debug_image, [circle['contour']], -1, (0, 255, 0), 3)
                    cv2.putText(debug_image, "✓", (center[0]-10, center[1]+10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                print(f"   ❌ Nenhuma miçanga {color_name} detectada")
        
        # Calcular total
        results['total_geral'] = sum(results[color]['valor'] for color in ['red', 'blue', 'yellow'])
        
        print(f"\n✅ Análise concluída!")
        return results, debug_image

    def display_results(self, results, debug_image):
        """Exibe os resultados de forma clara"""
        print("\n" + "="*50)
        print("📊 RESULTADOS DA CONTAGEM")
        print("="*50)
        
        for color in ['red', 'blue', 'yellow']:
            data = results[color]
            print(f"\n{color.upper():>8}:")
            print(f"   Detectadas: {data['detectadas']}")
            print(f"   Contabilizadas: {data['contabilizadas']}")
            print(f"   Valor: {data['valor']}")
        
        print(f"\n{'='*50}")
        print(f"💰 VALOR TOTAL: {results['total_geral']}")
        print(f"{'='*50}")
        
        # Mostrar imagem com resultados
        plt.figure(figsize=(15, 10))
        plt.imshow(cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB))
        plt.title("ANÁLISE DO ÁBACO - ✓ CONTA | X NÃO CONTA", fontsize=14, pad=20)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

# Interface simples de uso
def analisar_abacus(image_path):
    """
    Função principal para análise de ábaco
    """
    try:
        counter = RobustAbacusCounter()
        results, debug_image = counter.process_abacus(image_path)
        counter.display_results(results, debug_image)
        
        # Salvar resultados
        with open('resultado_abacus.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        cv2.imwrite('debug_abacus.jpg', debug_image)
        print("💾 Resultados salvos em 'resultado_abacus.json'")
        print("🖼️ Imagem de debug salva como 'debug_abacus.jpg'")
        
        return results
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        return None

# Executar
if __name__ == "__main__":
    # Substitua pelo caminho da sua imagem
    image_path = "modelos/contagem/25.jpg"
    analisar_abacus(image_path)