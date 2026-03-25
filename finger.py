import cv2
import mediapipe as mp
import pygame

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

# ====== CÂMERA ======
cap = cv2.VideoCapture(0)

# ====== CONTROLE ======
tempo_sem_rosto = 0
limite = 5
alerta_ativo = False

# ====== LOOP PRINCIPAL ======
while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ====== DETECÇÃO DA MÃO ======
    results = hands.process(img_rgb)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    # ====== DETECÇÃO DE ROSTO ======
    faces = face_cascade.detectMultiScale(gray, 1.2, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # ====== DETECÇÃO DE GÊNERO ======
        rosto = img[y:y+h, x:x+w]
        blob = cv2.dnn.blobFromImage(rosto, 1.0, (227, 227), MEAN_VALUES, swapRB=False)
        gender_net.setInput(blob)
        preds = gender_net.forward()
        genero = GENDER_LIST[preds[0].argmax()]

        cor_genero = (255, 0, 0) if genero == "Masculino" else (180, 105, 255)
        cv2.putText(img, genero, (x, y - 10),
            cv2.FONT_HERSHEY_TRIPLEX, 0.9, cor_genero, 2)

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

    # ====== EXIBIÇÃO ======
    cv2.imshow("Monitoramento de Foco", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# ====== FINALIZAÇÃO ======
cap.release()
cv2.destroyAllWindows()
pygame.mixer.music.stop()