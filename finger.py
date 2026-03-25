import cv2
import mediapipe as mp
import pygame
from ultralytics import YOLO

# ====== SOM ======
pygame.init()
pygame.mixer.init()
pygame.mixer.music.load("alerta.mp3")

# ====== MEDIAPIPE (MÃO) ======
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

# ====== DETECTOR DE ROSTO ======
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ====== MODELO DE GÊNERO ======
gender_net = cv2.dnn.readNetFromCaffe("gender_deploy.prototxt", "gender_net.caffemodel")
GENDER_LIST = ["Masculino", "Feminino"]
MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# ====== YOLOV8 ======
yolo = YOLO("yolov8n.pt")

# Classes mapeadas para categorias
HUMANOS = {"person"}
ANIMAIS = {
    "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe"
}

def classificar(label):
    if label in HUMANOS:
        return "Human"
    elif label in ANIMAIS:
        return "Animal"
    else:
        return "Objeto"

# Cores por categoria
CORES = {
    "Human":  (0, 255, 0),
    "Animal": (255, 165, 0),
    "Objeto": (200, 200, 200),
}

# ====== FILTRO DE TAMANHO ======
PROPORCAO_MIN = 0.01   # ignora menor que 1% da tela (ruído)
PROPORCAO_MAX = 0.20   # ignora maior que 20% da tela (objeto grande)

# ====== CÂMERA ======
cap = cv2.VideoCapture(0)

# ====== CONTROLE ======
tempo_sem_rosto = 0
limite = 5
alerta_ativo = False
frame_count = 0
resultados_yolo = []

# ====== LOOP PRINCIPAL ======
while True:
    success, img = cap.read()
    if not success:
        break

    frame_count += 1
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    altura_img, largura_img = img.shape[:2]
    area_total = altura_img * largura_img

    # ====== DETECÇÃO DA MÃO ======
    #results_hands = hands.process(img_rgb)
    #if results_hands.multi_hand_landmarks:
        #for handLms in results_hands.multi_hand_landmarks:
            #mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    # ====== YOLOV8: roda só a cada 2 frames ======
    if frame_count % 2 == 0:
        resultados_yolo = yolo(img, verbose=False)[0].boxes

    # ====== DESENHA RESULTADOS DO YOLO ======
    for box in resultados_yolo:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label_yolo = yolo.names[cls_id]
        categoria = classificar(label_yolo)
        cor = CORES[categoria]

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Filtro de tamanho
        area_box = (x2 - x1) * (y2 - y1)
        proporcao = area_box / area_total

        if proporcao < PROPORCAO_MIN or proporcao > PROPORCAO_MAX:
            continue

        if label_yolo != "person":
            cv2.rectangle(img, (x1, y1), (x2, y2), cor, 2)
            texto = f"{categoria}"
            cv2.putText(img, texto, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.7, cor, 2)

    # ====== DETECÇÃO DE ROSTO + GÊNERO ======
    faces = face_cascade.detectMultiScale(gray, 1.2, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), CORES["Human"], 2)

        rosto = img[y:y + h, x:x + w]
        blob = cv2.dnn.blobFromImage(rosto, 1.0, (227, 227), MEAN_VALUES, swapRB=False)
        gender_net.setInput(blob)
        preds = gender_net.forward()
        genero = GENDER_LIST[preds[0].argmax()]

        cor_genero = (255, 0, 0) if genero == "Masculino" else (180, 105, 255)
        cv2.putText(img, f"Human | {genero}", (x, y - 10),
                    cv2.FONT_HERSHEY_TRIPLEX, 0.8, cor_genero, 2)

    # ====== FOCO / DISTRAÇÃO ======
    if len(faces) > 0:
        tempo_sem_rosto = 0
        alerta_ativo = False
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
        cv2.putText(img, "FOCADO", (20, 50),
                    cv2.FONT_HERSHEY_TRIPLEX, 1.2, (0, 255, 0), 3)
    else:
        tempo_sem_rosto += 0.1
        cv2.putText(img, "DISTRAIDO", (20, 50),
                    cv2.FONT_HERSHEY_TRIPLEX, 1.2, (0, 0, 255), 3)
        if tempo_sem_rosto > limite:
            alerta_ativo = True

    # ====== ALERTA SONORO ======
    if alerta_ativo:
        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.play(-1)

    # ====== FPS NA TELA ======
    #cv2.putText(img, f"Frame: {frame_count}", (20, 90),
                #cv2.FONT_HERSHEY_TRIPLEX, 0.6, (150, 150, 150), 1)

    # ====== EXIBIÇÃO ======
    cv2.imshow("Monitoramento de Foco", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# ====== FINALIZAÇÃO ======
cap.release()
cv2.destroyAllWindows()
pygame.mixer.music.stop()