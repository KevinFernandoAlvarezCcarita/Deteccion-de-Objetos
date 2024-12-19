import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from threading import Thread
import datetime
import time

class Detector:
    def __init__(self, fuente, metodo):
        self.fuente = fuente
        self.metodo = metodo
        self.grabando = False
        self.out = None
        self.tiempo_ultima_deteccion = time.time()
        self.tiempo_maximo_sin_deteccion = 30  
        self.fps = 20.0

        if metodo == "YOLO":
            self.config = "model/yolov3.cfg"
            self.weights = "model/yolov3.weights"
            self.etiquetas = open("model/coco.names").read().strip().split("\n")
            self.colores = np.random.randint(0, 255, size=(len(self.etiquetas), 3), dtype="uint8")
            self.red_neuronal = cv2.dnn.readNetFromDarknet(self.config, self.weights)
        else:  
            self.clasificador_rostros = cv2.CascadeClassifier('model/reconocimientofacial.xml')

    def procesar(self):
        cap = cv2.VideoCapture(self.fuente)

        if not cap.isOpened():
            messagebox.showerror("Error", f"No se pudo abrir la fuente de video: {self.fuente}.")
            return

        cuatrocc = cv2.VideoWriter_fourcc(*"mp4v")  

        while True:
            ret, cuadro = cap.read()
            if not ret:
                break

            if self.metodo == "YOLO":
                self.procesar_yolo(cuadro, cuatrocc)
            else:
                self.procesar_harcascade(cuadro, cuatrocc)

            cv2.imshow(f"Detección - {self.fuente}", cuadro)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        if self.grabando:
            self.out.release()
        cv2.destroyAllWindows()

    def procesar_yolo(self, cuadro, cuatrocc):
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

                if id_clase == 0 and confianza > 0.3:  
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
            if not self.grabando:
                self.iniciar_grabacion(cuadro, cuatrocc, ancho, altura)

        if self.grabando:
            self.out.write(cuadro)

        if time.time() - self.tiempo_ultima_deteccion > self.tiempo_maximo_sin_deteccion:
            self.detener_grabacion()

        self.dibujar_cajas(cuadro, cajas, idx, confidencias)

    def procesar_harcascade(self, cuadro, cuatrocc):
        gris = cv2.cvtColor(cuadro, cv2.COLOR_BGR2GRAY)
        rostros = self.clasificador_rostros.detectMultiScale(gris, 1.4, 4)

        if len(rostros) > 0:
            self.tiempo_ultima_deteccion = time.time()  
            if not self.grabando:
                self.iniciar_grabacion(cuadro, cuatrocc, cuadro.shape[1], cuadro.shape[0])

        if self.grabando:
            self.out.write(cuadro)

        if time.time() - self.tiempo_ultima_deteccion > self.tiempo_maximo_sin_deteccion:
            self.detener_grabacion()

        for (x, y, w, h) in rostros:
            cv2.rectangle(cuadro, (x, y), (x + w, y + h), (255, 0, 0), 2)

    def iniciar_grabacion(self, cuadro, cuatrocc, ancho, altura):
        self.grabando = True
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.out = cv2.VideoWriter(f"grabado_{self.metodo.lower()}_{timestamp}.mp4", cuatrocc, self.fps, (ancho, altura))

    def detener_grabacion(self):
        if self.grabando:
            self.grabando = False
            self.out.release()
            print("Grabación detenida.")

    def dibujar_cajas(self, cuadro, cajas, idx, confidencias):
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

        self.video_button = tk.Button(self.frame, text="Abrir Videos", command=self.abrir_videos_multiples)
        self.video_button.pack(pady=10)

        self.metodo_var = tk.StringVar(value="YOLO")
        self.yolo_radio = tk.Radiobutton(self.frame, text="YOLO", variable=self.metodo_var, value="YOLO")
        self.yolo_radio.pack(side=tk.LEFT, padx=10)

        self.haar_radio = tk.Radiobutton(self.frame, text="Haar Cascade", variable=self.metodo_var, value="Haar Cascade")
        self.haar_radio.pack(side=tk.LEFT, padx=10)

    def abrir_videos_multiples(self):
        rutas_archivos = filedialog.askopenfilenames()
        if rutas_archivos:
            for ruta_archivo in rutas_archivos:
                metodo = self.metodo_var.get()
                detector = Detector(ruta_archivo, metodo)
                thread = Thread(target=detector.procesar)
                thread.start()

    def abrir_camara(self):
        url = self.camara_entry.get()  
        metodo = self.metodo_var.get()
        detector = Detector(url, metodo)  
        thread = Thread(target=detector.procesar)
        thread.start()

    def abrir_camara_local(self):
        try:
            indice_camara = int(self.local_camera_index_entry.get())
            metodo = self.metodo_var.get()
            detector = Detector(indice_camara, metodo)  
            thread = Thread(target=detector.procesar)
            thread.start()
        except ValueError:
            messagebox.showerror("Error", "Por favor, introduce un número válido para el índice de la cámara.")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
