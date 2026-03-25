import cv2
import numpy as np

fontes = [
    ("SIMPLEX", cv2.FONT_HERSHEY_SIMPLEX),
    ("PLAIN", cv2.FONT_HERSHEY_PLAIN),
    ("DUPLEX", cv2.FONT_HERSHEY_DUPLEX),
    ("COMPLEX", cv2.FONT_HERSHEY_COMPLEX),
    ("TRIPLEX", cv2.FONT_HERSHEY_TRIPLEX),
    ("COMPLEX_SMALL", cv2.FONT_HERSHEY_COMPLEX_SMALL),
    ("SCRIPT_SIMPLEX", cv2.FONT_HERSHEY_SCRIPT_SIMPLEX),
    ("SCRIPT_COMPLEX", cv2.FONT_HERSHEY_SCRIPT_COMPLEX),
]

img = np.zeros((500, 700, 3), dtype=np.uint8)

for i, (nome, fonte) in enumerate(fontes):
    y = 50 + i * 55
    cv2.putText(img, f"{nome}: Masculino / Feminino", (20, y),
                fonte, 0.8, (0, 255, 255), 2)

cv2.imshow("Fontes disponíveis", img)
cv2.waitKey(0)
cv2.destroyAllWindows()