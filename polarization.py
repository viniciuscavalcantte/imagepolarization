import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import os

# Função para calcular entropia local
def local_entropy(channel, kernel_size=5):
    padded = cv2.copyMakeBorder(channel, kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2, cv2.BORDER_REFLECT)
    entropy = np.zeros_like(channel, dtype=np.float32)
    for i in range(channel.shape[0]):
        for j in range(channel.shape[1]):
            patch = padded[i:i+kernel_size, j:j+kernel_size]
            hist = np.histogram(patch, bins=256, range=(0, 255), density=True)[0]
            hist = hist[hist > 0]
            entropy[i, j] = -np.sum(hist * np.log2(hist + 1e-10))
    return entropy / entropy.max()

# Função para calcular PSNR
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

# Função para calcular SSIM com win_size dinâmico
def calculate_ssim(img1, img2):
    min_side = min(img1.shape[0], img1.shape[1])
    win_size = min(7, max(3, min_side // 2 * 2 + 1))  # Ímpar, mínimo 3, máximo 7 ou menor lado
    img1_resized = cv2.resize(img1, (img2.shape[1], img2.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    
    if len(img1.shape) == 2:  # Imagem monocromática
        return ssim(img1_resized, img2, win_size=win_size, data_range=255)
    else:  # Imagem colorida
        return ssim(img1_resized, img2, win_size=win_size, data_range=255, channel_axis=2)

# Função para nitidez simples
def sharpen_image(image):
    blurred = cv2.GaussianBlur(image, (3, 3), 0)  # Kernel menor
    sharpened = cv2.addWeighted(image, 1.2, blurred, -0.2, 0)  # Pesos mínimos
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    print(f"Nitidez: min={sharpened.min()}, max={sharpened.max()}")
    if sharpened.min() == 0 and sharpened.max() == 0:
        print("Aviso: Imagem preta detectada após nitidez!")
    return sharpened

# Função para processar a imagem (colorida ou monocromática)
def process_image(image_path, scale_factor=2):
    # Verificar se o arquivo existe
    if not os.path.exists(image_path):
        raise ValueError(f"Arquivo não encontrado: {image_path}")
    
    # Carregar a imagem
    img_bgr = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img_bgr is None:
        raise ValueError("Não foi possível carregar a imagem.")
    
    # Verificar se a imagem é colorida ou monocromática
    if len(img_bgr.shape) == 3:  # Imagem colorida
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        is_color = True
    else:  # Imagem monocromática
        img_rgb = img_bgr
        is_color = False
    print(f"Imagem original: min={img_rgb.min()}, max={img_rgb.max()}")

    # Passo 1: Denoising
    if is_color:
        img_denoised = cv2.fastNlMeansDenoisingColored(img_rgb, h=8, hColor=8, templateWindowSize=7, searchWindowSize=21)
    else:
        img_denoised = cv2.fastNlMeansDenoising(img_rgb, h=8, templateWindowSize=7, searchWindowSize=21)
    print(f"Denoised: min={img_denoised.min()}, max={img_denoised.max()}")
    if img_denoised.min() == 0 and img_denoised.max() == 0:
        print("Aviso: Imagem preta detectada após denoising!")

    # Separar canais (ou usar canal único para monocromático)
    channels = cv2.split(img_denoised) if is_color else [img_denoised]
    
    # Inicializar listas para armazenar resultados
    edges_channels = []
    clahe_channels = []
    bilateral_channels = []
    
    # Processar cada canal
    for channel in channels:
        # Calcular entropia local para peso adaptativo
        entropy = local_entropy(channel)
        edge_weight = 3 + 3 * entropy.mean()  # Peso muito reduzido
        
        # Passo 2: Detecção de bordas multi-escala (Canny + Sobel)
        edges_canny = cv2.Canny(channel, 50, 120)
        sobelx = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=3)
        edges_sobel = np.sqrt(sobelx**2 + sobely**2)
        edges_sobel = np.clip(edges_sobel, 0, 255).astype(np.uint8)
        edges = cv2.addWeighted(edges_canny, 0.6, edges_sobel, 0.4, 0)
        edges_normalized = edges / 255.0
        print(f"Bordas: min={edges.min()}, max={edges.max()}")
        
        # Passo 3: Aplicar CLAHE com parâmetros dinâmicos
        clip_limit = 0.6 if entropy.mean() > 0.5 else 0.4  # Clip limit bem reduzido
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        clahe_channel = clahe.apply(channel)
        print(f"CLAHE: min={clahe_channel.min()}, max={clahe_channel.max()}")
        if clahe_channel.min() == 0 and clahe_channel.max() == 0:
            print("Aviso: Imagem preta detectada após CLAHE!")
        
        # Passo 4: Aplicar filtro bilateral otimizado
        bilateral = cv2.bilateralFilter(clahe_channel, d=5, sigmaColor=20, sigmaSpace=20)
        print(f"Bilateral: min={bilateral.min()}, max={bilateral.max()}")
        if bilateral.min() == 0 and bilateral.max() == 0:
            print("Aviso: Imagem preta detectada após bilateral!")
        
        # Passo 5: Combinar bordas realçadas com o canal processado
        polarized_channel = np.clip(bilateral + edge_weight * edges_normalized, 0, 255).astype(np.uint8)
        print(f"Polarized channel: min={polarized_channel.min()}, max={polarized_channel.max()}")
        if polarized_channel.min() == 0 and polarized_channel.max() == 0:
            print("Aviso: Imagem preta detectada após combinação de bordas!")
        
        edges_channels.append(edges)
        clahe_channels.append(clahe_channel)
        bilateral_channels.append(polarized_channel)
    
    # Combinar os canais processados de volta
    polarized_img = cv2.merge(bilateral_channels) if is_color else bilateral_channels[0]
    print(f"Polarized image: min={polarized_img.min()}, max={polarized_img.max()}")
    if polarized_img.min() == 0 and polarized_img.max() == 0:
        print("Aviso: Imagem preta detectada após fusão dos canais!")
    
    # Passo 6: Ampliação da imagem original e da imagem polarizada
    new_size = (int(img_rgb.shape[1] * scale_factor), int(img_rgb.shape[0] * scale_factor))
    img_resized = cv2.resize(img_denoised, new_size, interpolation=cv2.INTER_LANCZOS4)
    polarized_resized = cv2.resize(polarized_img, new_size, interpolation=cv2.INTER_LANCZOS4)
    print(f"Resized original: min={img_resized.min()}, max={img_resized.max()}")
    print(f"Resized polarized: min={polarized_resized.min()}, max={polarized_resized.max()}")
    if img_resized.min() == 0 and img_resized.max() == 0:
        print("Aviso: Imagem preta detectada após ampliação original!")
    if polarized_resized.min() == 0 and polarized_resized.max() == 0:
        print("Aviso: Imagem preta detectada após ampliação polarizada!")

    # Passo 7: Aplicar nitidez simples
    img_resized_sharpened = sharpen_image(img_resized)
    polarized_resized_sharpened = sharpen_image(polarized_resized)

    # Calcular métricas de qualidade
    psnr_original = calculate_psnr(img_rgb, cv2.resize(img_resized_sharpened, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_LANCZOS4))
    psnr_polarized = calculate_psnr(img_rgb, cv2.resize(polarized_resized_sharpened, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_LANCZOS4))
    ssim_original = calculate_ssim(img_rgb, img_resized_sharpened)
    ssim_polarized = calculate_ssim(img_rgb, polarized_resized_sharpened)

    # Combinar bordas dos canais para visualização
    edges_combined = cv2.merge(edges_channels) if is_color else edges_channels[0]
    print(f"Edges combined: min={edges_combined.min()}, max={edges_combined.max()}")

    return img_rgb, edges_combined, polarized_img, img_resized_sharpened, polarized_resized_sharpened, psnr_original, psnr_polarized, ssim_original, ssim_polarized

# Função para visualizar os resultados
def display_results(original, edges, polarized, resized_original, resized_polarized, psnr_original, psnr_polarized, ssim_original, ssim_polarized, output_path="resultados_ampliacao_colorida_fixed.png"):
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.title("Imagem Original")
    plt.imshow(original, cmap='gray' if len(original.shape) == 2 else None)
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.title("Bordas Detectadas (Canny + Sobel)")
    plt.imshow(edges, cmap='gray' if len(edges.shape) == 2 else None)
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.title("Imagem com Polarização Simulada")
    plt.imshow(polarized, cmap='gray' if len(polarized.shape) == 2 else None)
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.title(f"Ampliação Original\nPSNR: {psnr_original:.2f} dB, SSIM: {ssim_original:.3f}")
    plt.imshow(resized_original, cmap='gray' if len(resized_original.shape) == 2 else None)
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.title(f"Ampliação com Polarização\nPSNR: {psnr_polarized:.2f} dB, SSIM: {ssim_polarized:.3f}")
    plt.imshow(resized_polarized, cmap='gray' if len(resized_polarized.shape) == 2 else None)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
    print(f"Resultado salvo em: {output_path}")

# Função principal
def main():
    # Definir o caminho da imagem local
    image_path = 'D:\\004 - TRABALHOS_DOCUMENTOS_CC\PDI\project4\images2.jpg'
    scale_factor = 2
    output_path = "resultados_ampliacao_colorida_fixed.png"  # Caminho para salvar o resultado

    try:
        # Processar a imagem
        original, edges, polarized, resized_original, resized_polarized, psnr_original, psnr_polarized, ssim_original, ssim_polarized = process_image(image_path, scale_factor)
        # Exibir os resultados
        display_results(original, edges, polarized, resized_original, resized_polarized, psnr_original, psnr_polarized, ssim_original, ssim_polarized, output_path)
    except Exception as e:
        print(f"Erro: {e}")

if __name__ == "__main__":
    main()