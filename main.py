import cv2
import numpy as np
from ultralytics import YOLO

# Carregar modelo YOLOv8
model = YOLO("yolov8n.pt")

# Calcula o mapa de disparidade
def compute_disparity_map(imgL, imgR):
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    stereo = cv2.StereoBM_create(numDisparities=1024, blockSize=11)
    disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0
    return disparity

# Estima a distância a partir da disparidade
def estimate_distance(focal_length, baseline, disparity):
    with np.errstate(divide='ignore'):
        distance = (focal_length * baseline) / disparity
        distance[disparity <= 0] = 0
    return distance

# Converte a distância focal da lente em milímetros para pixels
def convert_focal_length(img_width):
    focal_length_mm = int(input("Informe a distância focal em mm: "))
    sensor_width = float(input("Informe a largura do sensor em mm: "))
    return (focal_length_mm/sensor_width)*img_width

# Função principal com salvamento automático
def process_stereo_images(imgL_path, imgR_path):
    imgL = cv2.imread(imgL_path)
    imgR = cv2.imread(imgR_path)
    img_width = imgL.shape[1]
    focal_length = convert_focal_length(img_width)
    baseline=float(input("Informe o baseline em metros: "))

    # Calcula disparidade e profundidade
    disparity = compute_disparity_map(imgL, imgR)
    distance_map = estimate_distance(focal_length, baseline, disparity)

    # Cria mapa de profundidade visual (colorido)
    disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disp_vis = np.uint8(disp_vis)
    disp_vis_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

    # Detecção com YOLO
    results = model(imgL)[0]
    img_with_boxes = imgL.copy()

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        label = model.names[cls]

        object_distance = distance_map[y1:y2, x1:x2]
        valid_dist = object_distance[object_distance > 0]
        if len(valid_dist) == 0:
            dist = "N/A"
        else:
            dist = f"{np.median(valid_dist):.2f} m"

        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 20)
        cv2.putText(img_with_boxes, f"{label} - {dist}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 255, 255), 20)

    # Salvar resultados separadamente
    cv2.imwrite("deteccao_e_distancias.png", img_with_boxes)
    cv2.imwrite("mapa_de_profundidade.png", disp_vis_color)

    print("✅ Imagens salvas:")
    print(" → deteccao_e_distancias.png")
    print(" → mapa_de_profundidade.png\n")

if __name__ == '__main__':
    process_stereo_images("esquerda.jpg", "direita.jpg")