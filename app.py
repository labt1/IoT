from flask import Flask, Response, render_template, request, jsonify
import cv2
import threading
import time
from process import *
import random
from werkzeug.utils import secure_filename
import os
import psycopg2
from psycopg2 import sql
from datetime import datetime
import requests, json

app = Flask(__name__)

ort_model = init_resources()
random.seed(42)
CLASS_COLORS = [[random.randint(0, 255) for _ in range(3)] for _ in range(80)]

# Configuración de conexión
DB_CONFIG = {
    'dbname': 'detections',    # Nombre de la base de datos
    'user': 'admin',           # Usuario
    'password': 'admin123',    # Contraseña
    'host': '192.168.1.100',       # Dirección del servicio PostgreSQL
    'port': 5432              # Puerto configurado en el NodePort
}

# Lista de URLs de las cámaras

camera_urls = [
    #"http://158.58.130.148/mjpg/video.mjpg",
    #"http://158.58.130.148/mjpg/video.mjpg",
    #"http://61.211.241.239/nphMotionJpeg?Resolution=320x240&Quality=Standard",
    "rtsp://localhost:554/stream",
    #"rtsp://localhost:554/stream",
    #"rtsp://localhost:554/stream",
    #"http://158.58.130.148/mjpg/video.mjpg",
    #"rtsp://localhost:554/stream",
    #"http://158.58.130.148/mjpg/video.mjpg"
    #"static/uploads/camera_1.mp4",
    #"http://158.58.130.148/mjpg/video.mjpg",
    #"http://61.211.241.239/nphMotionJpeg?Resolution=320x240&Quality=Standard"
]

# Diccionario para manejar threads y frames
streams = {}

def draw_bboxes(frame, bboxes, labels):
    for label, bbox in zip(labels, bboxes):
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), CLASS_COLORS[label], 2)
    return frame

def fetch_stream():
    url = 'http://192.168.1.102:80/detect-stream/'
    data = {
        "stream_url": "rtsp://192.168.1.101:8554/stream"
    }
    
    try:
        response = requests.post(url, json=data, stream=True)
        if response.status_code == 200:
            print("Conectado al stream. Recibiendo detecciones...")
            # Procesar el flujo de datos
            for line in response.iter_lines():
                if line:
                    try:
                        message = json.loads(line.decode('utf-8'))
                        print(message)
                    except json.JSONDecodeError as e:
                        print(f"Error al decodificar JSON: {e}")
        else:
            print(f"Error al conectar al servicio: {response.status_code}")
    except Exception as e:
        print(f"Ocurrió un error: {e}")

# Clase para manejar cada stream de cámara
class CameraStream:
    def __init__(self, camera_id, url=None):
        self.camera_id = camera_id
        self.url = url
        self.cap = None
        self.frame = None
        self.labels = None
        self.running = False
        self.lock = threading.Lock()

    def set_url(self, url):
        self.url = url
        self.lock = threading.Lock()
        self.cap = None
        self.frame = None
        self.running = False

    def start(self):
        if not self.running:
            self.running = True
            threading.Thread(target=self._capture, daemon=True).start()

    def _capture(self):
        self.cap = cv2.VideoCapture(self.url)
        
        if not self.cap.isOpened():
            print(f"Error: No se pudo abrir la cámara {self.camera_id}")
            self.running = False
            return

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print(f"Error: No se pudo leer el frame de la cámara {self.camera_id}")
                self.running = False
                break

            labels, scores, boxes = process_frame(frame, ort_model)

            draw_bboxes(frame, boxes, labels)

            with self.lock:
                self.labels = labels
                self.frame = frame

            time.sleep(1 / 30)
        self.cap.release()

    def get_frame(self):
        with self.lock:
            if self.frame is None:
                return None
            ret, buffer = cv2.imencode('.jpg', self.frame)
            if not ret:
                return None
            return buffer.tobytes()
        
    def get_detection(self):
        with self.lock:
            if self.labels is None:
                return None
            
            #ret, buffer = cv2.imencode('.jpg', self.frame)
            #if not ret:
            #    return None
            
            return self.labels

    def stop(self):
        self.running = False

#for i in range(0,9):
#    streams[i] = CameraStream(i)
#streams[0] = CameraStream(1)

# Inicializar streams

# Inicializar streams
#for i in range(0, 6):
#    streams[i] = CameraStream(i)
    #streams[i].start()

"""for i, url in enumerate(camera_urls):
    streams[i] = CameraStream(3, url)
    #streams[i].set_url(url)
    streams[i].start()"""


# Generador para el stream MJPEG
def generate_frames(camera_id):
    stream = streams.get(camera_id)
    if not stream:
        print(f"Error: Cámara {camera_id} no encontrada")
        return
    
    print("Camara con datos", camera_id)

    try:
        # Conectar a la base de datos
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

    except Exception as e:
        print(f"Error al guardar detecciones: {e}")
    #finally:
    #    if conn:
    #        cursor.close()
    #        conn.close()


    while True:
        frame = stream.get_frame()
        labels = stream.get_detection()

        if not labels is None:
            # Consulta para insertar detecciones
            insert_query = sql.SQL("""
                INSERT INTO detections (camera_id, frame_number, labels, timestamp)
                VALUES (%s, %s, %s, %s)
            """)
            
            timestamp = datetime.now()

            for i in labels:
                cursor.execute(insert_query, (camera_id+1, 1, int(i), timestamp))
                conn.commit()

        if frame is None:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# Endpoint para cada feed de cámara
@app.route('/video_feed/<int:camera_id>')
def video_feed(camera_id):
    print("Camera ID", camera_id)
    if camera_id not in streams:
        return f"Error: Cámara {camera_id} no configurada", 404
    return Response(generate_frames(camera_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Ruta para cargar el video y asociarlo a un slot
@app.route('/upload_video', methods=['POST'])
def upload_video():
    video = request.files.get('video')
    camera_id = int(request.form.get('camera_id'))

    if video and 1 <= camera_id <= 9:
        filename = secure_filename(video.filename)
        video_path = os.path.join('./static/uploads', filename)
        video.save(video_path)

        print("TEST", camera_id)
        # Detener el stream actual
        if camera_id - 1 in streams:
            streams[camera_id - 1].stop()

        # Reiniciar el stream con el nuevo video
        streams[camera_id - 1] = CameraStream(camera_id - 1, video_path)
        streams[camera_id - 1].start()

        thread = threading.Thread(target=fetch_stream)
        thread.start()
        print("Hilo de streaming iniciado")

        return jsonify({'status': 'success', 'message': f'Video cargado y procesado en el slot {camera_id}.'})
    else:
        return jsonify({'status': 'error', 'message': 'No se cargó el video o ID de cámara inválido.'})

@app.route('/search_detections', methods=['GET'])
def search_detections():
    object_label = request.args.get('object_label')
    time_interval = str(request.args.get('time_interval'))
    print(time_interval)

    if not object_label:
        return jsonify({'error': 'Se requiere un label para buscar.'}), 400

    try:
        # Conexión a la base de datos
        connection = psycopg2.connect(**DB_CONFIG)
        cursor = connection.cursor()

        # Query de búsqueda para obtener solo una detección por segundo
        query = """
            WITH ranked_detections AS (
                SELECT
                    camera_id,
                    frame_number,
                    labels,
                    timestamp,
                    ROW_NUMBER() OVER (PARTITION BY camera_id, labels, FLOOR(EXTRACT(EPOCH FROM timestamp) / %s) ORDER BY timestamp DESC) AS row_num
                FROM detections
                WHERE labels = %s
            )
            SELECT
                camera_id,
                frame_number,
                labels,
                timestamp
            FROM ranked_detections
            WHERE row_num = 1
            ORDER BY timestamp DESC
            LIMIT 10;
        """
        cursor.execute(query, (time_interval, object_label))
        results = cursor.fetchall()

        # Formatear los resultados
        response = [
            {
                'camera_id': row[0],
                'frame_number': row[1],
                'label': row[2],
                'timestamp': row[3].isoformat()
            }
            for row in results
        ]

        return jsonify(response)

    except Exception as e:
        print(str(e))
        return jsonify({'error': str(e)}), 500
    finally:
        if connection:
            cursor.close()
            connection.close()

# Página principal
@app.route('/')
def index():
    return render_template('index.html', cameras=len(camera_urls))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
