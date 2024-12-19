import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from threading import Thread
import datetime
import time

class Grabador:
    def __init__(self, metodo):
        self.metodo = metodo
        self.grabando = False
        self.out = None
        self.fps = 20.0

    def iniciar(self, cuadro, ancho, altura):
        if not self.grabando:
            self.grabando = True
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            cuatrocc = cv2.VideoWriter_fourcc(*"mp4v") 
            self.out = cv2.VideoWriter(f"grabaciones/grabado_{self.metodo.lower()}_{timestamp}.mp4", cuatrocc, self.fps, (ancho, altura))

    def grabar(self, cuadro):
        if self.grabando and self.out is not None:
            self.out.write(cuadro)

    def detener(self):
        if self.grabando:
            self.grabando = False
            if self.out:
                self.out.release()
                print("Grabación detenida.")

class Detector:
    def __init__(self, fuente, metodo):
        self.fuente = fuente
        self.metodo = metodo
        self.grabador = Grabador(metodo)
        self.tiempo_ultima_deteccion = time.time()
        self.tiempo_maximo_sin_deteccion = 30  

        if metodo == "YOLO":
            self.config = "Model/Yolo/yolov3.cfg"
            self.weights = "Model/Yolo/yolov3.weights"
            self.etiquetas = open("Model/Yolo/coco.names").read().strip().split("\n")
            self.colores = np.random.randint(0, 255, size=(len(self.etiquetas), 3), dtype="uint8")

            self.red_neuronal = cv2.dnn.readNetFromDarknet(self.config, self.weights)
            self.red_neuronal.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)  # Usar la GPU para la inferencia

        else:  
            self.clasificador_rostros = cv2.CascadeClassifier('Model/Haar/ReconocimientoFacial.xml')
            self.clasificador_torso = cv2.CascadeClassifier('Model/Haar/ReconocimientoTorso.xml')

    def procesar(self):
        cap = cv2.VideoCapture(self.fuente)

        if not cap.isOpened():
            messagebox.showerror("Error", f"No se pudo abrir la fuente de video: {self.fuente}.")
            return

        while True:
            ret, cuadro = cap.read()
            if not ret:
                break

            if self.metodo == "YOLO":
                self.procesar_yolo(cuadro)
            else:
                self.procesar_harcascade(cuadro)

            cv2.imshow(f"Detección - {self.fuente}", cuadro)

            if cv2.waitKey(1) & 0xFF == ord('s'):
                break

        cap.release()
        self.grabador.detener()
        cv2.destroyAllWindows()

    def procesar_yolo(self, cuadro):
        altura, ancho, _ = cuadro.shape
        blob = cv2.dnn.blobFromImage(cuadro, 1 / 255.0, (416, 416), swapRB=True, crop=False)

        ln = self.red_neuronal.getLayerNames()
        ln = [ln[i - 1] for i in self.red_neuronal.getUnconnectedOutLayers()]
        self.red_neuronal.setInput(blob)
        salidas = self.red_neuronal.forward(ln)

        cajas, confidencias, ids_clase = [], [], []

        for salida in salidas:
            for deteccion in salida:
                puntajes = deteccion[5:]
                id_clase = np.argmax(puntajes)
                confianza = puntajes[id_clase]

                if confianza > 0.3:  
                    caja = deteccion[:4] * np.array([ancho, altura, ancho, altura])
                    (x_centro, y_centro, w, h) = caja.astype("int")
                    x = int(x_centro - (w / 2))
                    y = int(y_centro - (h / 2))
                    cajas.append([x, y, w, h])
                    confidencias.append(float(confianza))
                    ids_clase.append(id_clase)

        idx = cv2.dnn.NMSBoxes(cajas, confidencias, 0.5, 0.4)

        if len(idx) > 0:
            self.tiempo_ultima_deteccion = time.time()  
            self.grabador.iniciar(cuadro, ancho, altura)

        self.grabador.grabar(cuadro)

        if time.time() - self.tiempo_ultima_deteccion > self.tiempo_maximo_sin_deteccion:
            self.grabador.detener()

        if len(idx) > 0:
            self.dibujar_cajas(cuadro, cajas, idx, confidencias, ids_clase)

    def procesar_harcascade(self, cuadro):
        gris = cv2.cvtColor(cuadro, cv2.COLOR_BGR2GRAY)

        rostros = self.clasificador_rostros.detectMultiScale(gris, 1.3, 5)
        if len(rostros) > 0:
            self.tiempo_ultima_deteccion = time.time()  
            self.grabador.iniciar(cuadro, cuadro.shape[1], cuadro.shape[0])

        torsos = self.clasificador_torso.detectMultiScale(gris, 1.1, 5)
        if len(torsos) > 0:
            self.tiempo_ultima_deteccion = time.time()  
            self.grabador.iniciar(cuadro, cuadro.shape[1], cuadro.shape[0])

        self.grabador.grabar(cuadro)

        if time.time() - self.tiempo_ultima_deteccion > self.tiempo_maximo_sin_deteccion:
            self.grabador.detener()

        for (x, y, w, h) in rostros:
            cv2.rectangle(cuadro, (x, y), (x + w, y + h), (255, 0, 0), 2)

        for (x, y, w, h) in torsos:
            cv2.rectangle(cuadro, (x, y), (x + w, y + h), (0, 255, 0), 2)

    def dibujar_cajas(self, cuadro, cajas, idx, confidencias, ids_clase):
        for i in idx.flatten():
            (x, y) = (cajas[i][0], cajas[i][1])
            (w, h) = (cajas[i][2], cajas[i][3])
            color = self.colores[ids_clase[i]].tolist()
            texto = "{}: {:.3f}".format(self.etiquetas[ids_clase[i]], confidencias[i])
            cv2.rectangle(cuadro, (x, y), (x + w, y + h), color, 2)
            cv2.putText(cuadro, texto, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

class App:
    def __init__(self, master):
        self.master = master
        self.master.title("Detección de Objetos")
        self.frame = tk.Frame(master)
        self.frame.pack(pady=20)

        self.cam_frame = tk.Frame(self.frame)
        self.cam_frame.pack(pady=10)

        self.camara_label = tk.Label(self.cam_frame, text="Dirección IP de Cámara:")
        self.camara_label.grid(row=0, column=0)
        self.camara_entry = tk.Entry(self.cam_frame)
        self.camara_entry.grid(row=0, column=1)
        self.camara_entry.insert(0, "rtsp://usuario:contraseña@IP_CAMARA")  
        self.camara_button = tk.Button(self.cam_frame, text="Abrir Cámara IP", command=self.abrir_camara)
        self.camara_button.grid(row=0, column=2)

        self.local_camera_label = tk.Label(self.cam_frame, text="Cámara Local (índice):")
        self.local_camera_label.grid(row=1, column=0)
        self.local_camera_index_entry = tk.Entry(self.cam_frame)
        self.local_camera_index_entry.grid(row=1, column=1)
        self.local_camera_index_entry.insert(0, "0")  
        self.local_camera_button = tk.Button(self.cam_frame, text="Abrir Cámara Local", command=self.abrir_camara_local)
        self.local_camera_button.grid(row=1, column=2)

        self.video_button = tk.Button(self.frame, text="Abrir Video", command=self.abrir_video)
        self.video_button.pack(pady=10)

        self.metodo_var = tk.StringVar(value="Haar Cascade")  
        self.yolo_radio = tk.Radiobutton(self.frame, text="YOLO", variable=self.metodo_var, value="YOLO")
        self.yolo_radio.pack(side=tk.LEFT, padx=10)

        self.haar_radio = tk.Radiobutton(self.frame, text="Haar Cascade", variable=self.metodo_var, value="Haar Cascade")
        self.haar_radio.pack(side=tk.LEFT, padx=10)

    def abrir_video(self):
        archivo = filedialog.askopenfilename(title="Selecciona un video", filetypes=(("Archivos de video", "*.mp4;*.avi"), ("Todos los archivos", "*.*")))
        if archivo:
            metodo = self.metodo_var.get()
            self.iniciar_detector(archivo, metodo)

    def abrir_camara(self):
        direccion_ip = self.camara_entry.get()
        metodo = self.metodo_var.get()
        self.iniciar_detector(direccion_ip, metodo)

    def abrir_camara_local(self):
        indice_camara = int(self.local_camera_index_entry.get())  
        metodo = self.metodo_var.get()
        self.iniciar_detector(indice_camara, metodo)

    def iniciar_detector(self, fuente, metodo):
        detector = Detector(fuente, metodo)
        hilo = Thread(target=detector.procesar)
        hilo.daemon = True
        hilo.start()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
