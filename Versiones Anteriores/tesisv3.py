import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from threading import Thread
import time
import cv2  

class Grabador:
    def __init__(self, metodo):
        self.metodo = metodo
        self.grabando = False

    def iniciar(self):
        if not self.grabando:
            self.grabando = True
            print("Grabación iniciada")

    def grabar(self, cuadro):
        if self.grabando:
            print("Grabando cuadro...")

    def detener(self):
        if self.grabando:
            self.grabando = False
            print("Grabación detenida.")

class Detector:
    def __init__(self, fuente, metodo):
        self.fuente = fuente
        self.metodo = metodo
        self.grabador = Grabador(metodo)
        self.tiempo_ultima_deteccion = time.time()
        self.tiempo_maximo_sin_deteccion = 30  

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

        while True:
            ret, cuadro = cap.read()
            if not ret:
                break

            if self.metodo == "YOLO":
                self.procesar_yolo(cuadro)
            else:
                self.procesar_harcascade(cuadro)
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
            self.grabador.iniciar()

        self.grabador.grabar(cuadro)

        if time.time() - self.tiempo_ultima_deteccion > self.tiempo_maximo_sin_deteccion:
            self.grabador.detener()

    def procesar_harcascade(self, cuadro):
        gris = cv2.cvtColor(cuadro, cv2.COLOR_BGR2GRAY)
        rostros = self.clasificador_rostros.detectMultiScale(gris, 1.4, 4)

        if len(rostros) > 0:
            self.tiempo_ultima_deteccion = time.time()
            self.grabador.iniciar()

        self.grabador.grabar(cuadro)

        if time.time() - self.tiempo_ultima_deteccion > self.tiempo_maximo_sin_deteccion:
            self.grabador.detener()

class App:
    def __init__(self, master):
        self.master = master
        self.master.title("Detección de Objetos")
        self.frame = tk.Frame(master)
        self.frame.pack(pady=20)

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

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
