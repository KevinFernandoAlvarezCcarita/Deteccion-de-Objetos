import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from threading import Thread
import datetime
import time

class Grabador:
    def __init__(self, metodo):
        self.metodo = metodo
        self.grabando = False
        self.out = None
        self.fps = 30.0

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
    def __init__(self, fuente, metodo, tiempo_maximo):
        self.fuente = fuente
        self.metodo = metodo
        self.grabador = Grabador(metodo)
        self.tiempo_ultima_deteccion = time.time()
        self.tiempo_maximo_sin_deteccion = tiempo_maximo

        if metodo == "YOLO":
            self.config = "model/Yolo/yolov3.cfg"
            self.weights = "model/Yolo/yolov3.weights"
            self.etiquetas = open("model/Yolo/coco.names").read().strip().split("\n")
            self.colores = np.random.randint(0, 255, size=(len(self.etiquetas), 3), dtype="uint8")
            self.red_neuronal = cv2.dnn.readNetFromDarknet(self.config, self.weights)
        else:
            self.clasificador_rostros = cv2.CascadeClassifier('model/Haar/reconocimientofacial.xml')

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
            self.grabador.iniciar(cuadro, ancho, altura)

        self.grabador.grabar(cuadro)

        if time.time() - self.tiempo_ultima_deteccion > self.tiempo_maximo_sin_deteccion:
            self.grabador.detener()

        if len(idx) > 0:
            self.dibujar_cajas(cuadro, cajas, idx, confidencias, ids_clase)

    def procesar_harcascade(self, cuadro):
        gris = cv2.cvtColor(cuadro, cv2.COLOR_BGR2GRAY)
        rostros = self.clasificador_rostros.detectMultiScale(gris, 1.4, 4)

        if len(rostros) > 0:
            self.tiempo_ultima_deteccion = time.time()
            self.grabador.iniciar(cuadro, cuadro.shape[1], cuadro.shape[0])

        self.grabador.grabar(cuadro)

        if time.time() - self.tiempo_ultima_deteccion > self.tiempo_maximo_sin_deteccion:
            self.grabador.detener()

        for (x, y, w, h) in rostros:
            cv2.rectangle(cuadro, (x, y), (x + w, y + h), (255, 0, 0), 2)

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

        self.metodo_label = tk.Label(self.frame, text="Método de Detección:")
        self.metodo_label.pack()

        self.metodo_combobox = ttk.Combobox(self.frame, values=["Haar Cascade", "YOLO"], state="readonly")
        self.metodo_combobox.set("Haar Cascade")
        self.metodo_combobox.pack()

        self.tiempo_label = tk.Label(self.frame, text="Tiempo sin Detección:")
        self.tiempo_label.pack()

        self.tiempo_entry = tk.Entry(self.frame)
        self.tiempo_entry.insert(0, "30")
        self.tiempo_entry.pack()

        self.cam_label = tk.Label(self.cam_frame, text="Cámara:")
        self.cam_label.pack()

        self.camaras_combobox = ttk.Combobox(self.cam_frame, state="readonly")
        self.camaras_combobox.pack()

        self.ip_entry = tk.Entry(self.cam_frame, state="normal")
        self.ip_entry.pack()

        self.actualizar_camaras_button = tk.Button(self.cam_frame, text="Actualizar Cámaras", command=self.actualizar_camaras)
        self.actualizar_camaras_button.pack()

        self.iniciar_button = tk.Button(self.cam_frame, text="Iniciar", command=self.abrir_seleccion_camara)
        self.iniciar_button.pack()

        self.video_button = tk.Button(self.cam_frame, text="Abrir Video", command=self.abrir_video)
        self.video_button.pack()

        self.actualizar_camaras()

    def detectar_camaras_disponibles(self):
        camaras = []
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                camaras.append(f"Cámara {i}")
                cap.release()
        return camaras

    def actualizar_camaras(self):
        camaras = self.detectar_camaras_disponibles()
        camaras.append("Manual (IP)")
        self.camaras_combobox["values"] = camaras
        self.camaras_combobox.set("Seleccionar")

    def abrir_seleccion_camara(self):
        seleccion = self.camaras_combobox.get()
        if seleccion == "Manual (IP)":
            fuente = self.ip_entry.get()
        else:
            fuente = int(seleccion.split(" ")[1])

        metodo = self.metodo_combobox.get()
        tiempo_maximo = int(self.tiempo_entry.get())
        detector = Detector(fuente, metodo, tiempo_maximo)
        thread = Thread(target=detector.procesar, daemon=True)
        thread.start()

    def abrir_video(self):
        ruta_video = filedialog.askopenfilename()
        if ruta_video:
            metodo = self.metodo_combobox.get()
            tiempo_maximo = int(self.tiempo_entry.get())
            detector = Detector(ruta_video, metodo, tiempo_maximo)
            thread = Thread(target=detector.procesar, daemon=True)
            thread.start()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
