# 1. Definição da lista de imagens (baseado nos arquivos que você enviou)
# Se as imagens estiverem em uma pasta específica, adicione o caminho (ex: '/content/1.png')
lista_de_imagens = ['1.png', '2.png', '3.png', '4.png', '5.png', '6.png']

# 2. Definir Parâmetros Customizados
# Exemplo: Queremos ignorar os 5% mais escuros e 5% mais claros (P5 e P95)
min_p = 10
max_p = 80

# Exemplo: Thresholds de classe personalizados
class_th_1 = 60   # Sem turbidez
class_th_2 = 120  # Turbidez baixa
class_th_3 = 180  # Turbidez média
class_th_4 = 240  # Separação alta

# 3. Executar a função
results = image_classification_for_hsp_method_v01_63(
    image_paths=lista_de_imagens,
    show_plots=True, # Irá gerar as duas figuras (Objeto 1 e Objeto 2)
    minimum_percentil=min_p,
    maximum_percentil=max_p,
    maximum_pixel_intensity_for_class_1=class_th_1,
    maximum_pixel_intensity_for_class_2=class_th_2,
    maximum_pixel_intensity_for_class_3=class_th_3,
    maximum_pixel_intensity_for_class_4=class_th_4
)
