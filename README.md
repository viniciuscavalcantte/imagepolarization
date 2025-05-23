# 🧠 Ampliação de Imagens com Polarização Simulada

Projeto da disciplina de **Processamento Digital de Imagens** (UFAL - Arapiraca) que simula o uso de **técnicas de polarização** aplicadas à **ampliação de imagens digitais**. O objetivo é realçar bordas, texturas e detalhes antes da ampliação, inspirando-se nos efeitos físicos da polarização da luz.

---

## 📌 Objetivo

Investigar e aplicar técnicas computacionais inspiradas em fenômenos de polarização para melhorar a **qualidade de imagens ampliadas**, com foco em detalhes e contraste local.

---

## 🧪 Metodologia

A pipeline implementada segue os seguintes passos:

1. **Leitura e pré-processamento da imagem**
   - Suporte a imagens coloridas ou monocromáticas
   - Denoising com `fastNlMeansDenoising`

2. **Cálculo de entropia local**
   - Serve como base para decisões adaptativas de contraste e bordas

3. **Detecção de bordas multiescala**
   - Combinação de filtros **Canny** e **Sobel**

4. **Equalização adaptativa de histograma (CLAHE)**
   - Ajuste dinâmico com base na entropia

5. **Filtragem bilateral**
   - Suavização com preservação de contornos

6. **Fusão com bordas (simulação de polarização)**
   - Bordas realçadas são somadas aos canais com pesos adaptativos

7. **Ampliação com interpolação Lanczos**
   - Técnica avançada para preservar qualidade na ampliação

8. **Aplicação de nitidez**
   - Realce leve de contornos pós-ampliação

9. **Métricas de qualidade**
   - `PSNR` e `SSIM` comparando as versões originais e polarizadas

---

## 🛠️ Ferramentas e Bibliotecas

- Python 3.x  
- [OpenCV](https://opencv.org/)  
- [NumPy](https://numpy.org/)  
- [Matplotlib](https://matplotlib.org/)  
- [scikit-image (SSIM)](https://scikit-image.org/)

---

## 📁 Estrutura do Projeto

```
.
├── polarization.py                  # Código principal
├── images2.jpg                      # Imagem de entrada (exemplo local)
├── resultados_ampliacao_colorida_fixed.png  # Resultado salvo
├── README.md
```

---

## 📊 Resultados

A técnica simulada melhora o realce de detalhes ao ampliar a imagem. As bordas são reforçadas, e o contraste local é adaptado conforme a complexidade visual (entropia). Métricas obtidas:

| Versão                  | PSNR (dB) | SSIM   |
|------------------------|-----------|--------|
| Ampliação tradicional  | 33.98     | 0.916  |
| Ampliação polarizada   | 29.94     | 0.887  |

> *Valores exatos são exibidos ao executar o script.*

---

## 📸 Exemplo de Execução

```bash
python polarization.py
```

Isso gera e salva a imagem `resultados_ampliacao_colorida_fixed.png`, com todas as comparações lado a lado.

---

## 🎓 Conclusão

A abordagem baseada em **polarização digital simulada** demonstrou resultados promissores para a **ampliação de imagens com preservação de detalhes**. Essa integração entre técnicas físicas e computacionais reforça a importância de abordagens híbridas em PDI.

---

## 📎 Referência

Trabalho prático apresentado na disciplina de **Processamento Digital de Imagens (PDI)**  
**Universidade Federal de Alagoas – Campus Arapiraca**  

## 👥 Desenvolvedores
- **José Carlos Silva Santos**   
- **José Vinicius Cavalcante Soares** 
- **Jorge Lucas Firmino Silva de Sá**
- **Liedson Da Silva Santos**
- **Samuel Jonas Cavalcante Lima**
