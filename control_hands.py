import cv2
import mediapipe as mp
import json
import asyncio
import websockets
import asyncio

# Configuração do MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Função para enviar comando JSON via WebSocket
async def send_command(ws, angle, speed):
    command = {
        "angulo": angle,
        "velocidade": speed
    }
    await ws.send(json.dumps(command))
    print(f"Enviado comando: {command}")

# Função para verificar se o dedo está aberto
def is_finger_open(landmarks, finger_tip, finger_mcp):
    return landmarks[finger_tip].y < landmarks[finger_mcp].y

# Função principal de controle com timeout aumentado
async def main():
    uri = "ws://192.168.4.1/ws"  # Endereço WebSocket do robô

    try:
        async with asyncio.wait_for(websockets.connect(uri), timeout=10) as ws:
            cap = cv2.VideoCapture(0)
            try:
                while cap.isOpened():
                    success, image = cap.read()
                    if not success:
                        print("Ignorando frame vazio da câmera.")
                        continue

                    # Converte a imagem para RGB e processa com MediaPipe
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = hands.process(image_rgb)

                    # Verifica se há uma mão detectada
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            # Desenha os pontos da mão para visualização
                            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                            # Extrai pontos dos dedos
                            landmarks = hand_landmarks.landmark
                            thumb_open = is_finger_open(landmarks, mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.THUMB_IP)
                            pinky_open = is_finger_open(landmarks, mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_MCP)
                            index_open = is_finger_open(landmarks, mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP)
                            middle_open = is_finger_open(landmarks, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP)
                            ring_open = is_finger_open(landmarks, mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP)

                            # Define os gestos e envia o comando correspondente
                            if all([thumb_open, index_open, middle_open, ring_open, pinky_open]):
                                await send_command(ws, 0, 0)  # Mão aberta - Parar
                            elif not any([thumb_open, index_open, middle_open, ring_open, pinky_open]):
                                await send_command(ws, 0, 50)  # Mão fechada - Andar
                            elif thumb_open and not pinky_open:
                                await send_command(ws, 45, 50)  # Polegar aberto - Andar para a direita
                            elif pinky_open and not thumb_open:
                                await send_command(ws, 135, 50)  # Mindinho aberto - Andar para a esquerda
                            elif thumb_open and pinky_open:
                                await send_command(ws, 0, 0)  # Polegar e mindinho abertos - Parar

                    # Exibe o feed da câmera com os pontos de referência da mão
                    cv2.imshow("Controle do Robô com MediaPipe", image)
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
            finally:
                cap.release()
                cv2.destroyAllWindows()
    except asyncio.TimeoutError:
        print("Tempo limite para conexão ao WebSocket esgotado. Verifique o IP e a conectividade do robô.")

# Executa o controle assíncrono
asyncio.run(main())
