import cv2
import easyocr
from googletrans import Translator
import numpy as np

# Crear lector de OCR
reader = easyocr.Reader(['en'])
translator = Translator()

# Captura de cámara
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = frame.copy()

    # Detectar texto con EasyOCR
    results = reader.readtext(frame)

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    traducciones = []

    for (bbox, text, prob) in results:
        if prob > 0.5:
            pts = np.array(bbox, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
            try:
                traduccion = translator.translate(text, src='en', dest='es').text
            except:
                traduccion = text
            traducciones.append((pts, traduccion))

    # Borrar texto original
    img_sin_texto = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)

    # Escribir traducciones
    for pts, traduccion in traducciones:
        x, y = pts[0]
        # Fondo para texto para mejorar legibilidad
        cv2.rectangle(img_sin_texto,
                      (int(x), int(y-30)),
                      (int(x+200), int(y)),
                      (255,255,255), -1)
        cv2.putText(img_sin_texto,
                    traduccion,
                    (int(x), int(y-5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0,0,0),
                    2,
                    cv2.LINE_AA)

    cv2.imshow("Traductor IA Magico", img_sin_texto)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
        break

cap.release()
cv2.destroyAllWindows()
