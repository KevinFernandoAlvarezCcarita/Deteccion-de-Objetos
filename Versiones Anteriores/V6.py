import os
import locale
import numpy as np
import cv2
import datetime
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from threading import Thread
import json

locale.setlocale(locale.LC_ALL, 'es_ES.UTF-8')  

def cargar_modelos_haar():
    modelos = []
    ruta_modelos = "model/Haar/"
    for archivo in os.listdir(ruta_modelos):
        if archivo.endswith(".xml"):
            modelos.append(archivo)
    return modelos

def guardar_ip_camara(ip):
    try:
        with open("camara_ip.json", "w") as archivo:
            json.dump({"ip_camara": ip}, archivo)
    except Exception as e:
        print(f"Error al guardar la IP de la cámara: {e}")

def cargar_ip_camara():
    try:
        with open("camara_ip.json", "r") as archivo:
            datos = json.load(archivo)
            return datos.get("ip_camara", "")
    except FileNotFoundError:
        return ""
    except Exception as e:
        print(f"Error al cargar la IP de la cámara: {e}")
        return ""

class Grabador:
    def __init__(self, metodo, modelo_haar):
        self.metodo = metodo
        self.modelo_haar = modelo_haar
        self.grabando = False
        self.out = None
        self.fps = 24.0
        self.video_path = None
        self.drive = self.inicializar_drive()

    def inicializar_drive(self):
        gauth = GoogleAuth()
        gauth.LoadCredentialsFile("credenciales.txt")
        if not gauth.credentials:
            gauth.LocalWebserverAuth()
        elif gauth.access_token_expired:
            gauth.Refresh()
        gauth.SaveCredentialsFile("credenciales.txt")
        return GoogleDrive(gauth)

    def iniciar(self, cuadro, ancho, altura):
        if not self.grabando:
            self.grabando = True
            timestamp = datetime.datetime.now().strftime("%d_de_%B_de_%Y_hora_%H_%M_%S")  # Incluyendo segundos
            self.video_path = f"grabaciones/grabado_{timestamp}.mp4"
            cuatrocc = cv2.VideoWriter_fourcc(*"mp4v")
            self.out = cv2.VideoWriter(self.video_path, cuatrocc, self.fps, (ancho, altura))

    def grabar(self, cuadro):
        if self.grabando and self.out is not None:
            self.out.write(cuadro)

    def detener(self):
        if self.grabando:
            self.grabando = False
            if self.out:
                self.out.release()
                print(f"Grabación detenida. Video guardado en {self.video_path}")
                self.subir_a_google_drive(self.video_path)

    def subir_a_google_drive(self, file_path):
        try:
            file_drive = self.drive.CreateFile({'title': os.path.basename(file_path)})
            file_drive.SetContentFile(file_path)
            file_drive.Upload()
            print(f"Video subido exitosamente a Google Drive: {file_drive['title']}")
        except Exception as e:
            print(f"Error al subir el video a Google Drive: {e}")

    def borrar_credenciales(self):
        try:
            if os.path.exists("credenciales.txt"):
                os.remove("credenciales.txt")
            print("Credenciales borradas exitosamente.")
        except Exception as e:
            print(f"Error al borrar las credenciales: {e}")

class Detector:
    def __init__(self, fuente, metodo, tiempo_maximo, modelo_haar):
        self.fuente = fuente
        self.metodo = metodo
        self.modelo_haar = modelo_haar
        self.grabador = Grabador(metodo, modelo_haar)
        self.tiempo_ultima_deteccion = time.time()
        self.tiempo_maximo_sin_deteccion = tiempo_maximo

        if metodo == "YOLO":
            self.config = "model/Yolo/yolov3.cfg"
            self.weights = "model/Yolo/yolov3.weights"
            self.etiquetas = open("model/Yolo/coco.names").read().strip().split("\n")
            self.colores = np.random.randint(0, 255, size=(len(self.etiquetas), 3), dtype="uint8")
            self.red_neuronal = cv2.dnn.readNetFromDarknet(self.config, self.weights)
        else:
            self.clasificador_rostros = cv2.CascadeClassifier(f'model/Haar/{self.modelo_haar}')

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

    def procesar_harcascade(self, cuadro):
        gris = cv2.cvtColor(cuadro, cv2.COLOR_BGR2GRAY)
        rostros = self.clasificador_rostros.detectMultiScale(gris, 1.4, 4)

        if len(rostros) > 0:
            self.tiempo_ultima_deteccion = time.time()
            self.grabador.iniciar(cuadro, cuadro.shape[1], cuadro.shape[0])

        self.grabador.grabar(cuadro)

        if time.time() - self.tiempo_ultima_deteccion > self.tiempo_maximo_sin_deteccion:
            self.grabador.detener()

modelos_haar = cargar_modelos_haar()

class App:
    def __init__(self, master):
        self.master = master
        self.master.title("Detección de Objetos")
        self.frame = tk.Frame(master)
        self.frame.pack(pady=20)

        self.metodo_label = tk.Label(self.frame, text="Método de Detección:")
        self.metodo_label.pack()

        self.metodo_combobox = ttk.Combobox(self.frame, values=["Haar Cascade", "YOLO"], state="readonly")
        self.metodo_combobox.set("Haar Cascade")
        self.metodo_combobox.pack()

        self.modelo_haar_label = tk.Label(self.frame, text="Seleccionar Modelo Haar:")
        self.modelo_haar_label.pack()

        self.modelo_haar_combobox = ttk.Combobox(self.frame, values=modelos_haar, state="readonly")
        self.modelo_haar_combobox.set(modelos_haar[0] if modelos_haar else "No hay modelos")
        self.modelo_haar_combobox.pack()

        self.ip_label = tk.Label(self.frame, text="Cámara IP:")
        self.ip_label.pack()

        self.ip_entry = tk.Entry(self.frame)
        self.ip_entry.pack()

        ip_guardada = cargar_ip_camara()
        if ip_guardada:
            self.ip_entry.insert(0, ip_guardada)

        self.tiempo_label = tk.Label(self.frame, text="Tiempo sin Detección (segundos):")
        self.tiempo_label.pack()

        self.tiempo_entry = tk.Entry(self.frame)
        self.tiempo_entry.insert(0, "30")
        self.tiempo_entry.pack()

        self.iniciar_button = tk.Button(self.frame, text="Iniciar Detección", command=self.iniciar_deteccion)
        self.iniciar_button.pack()

        self.video_button = tk.Button(self.frame, text="Abrir Video", command=self.abrir_video)
        self.video_button.pack(pady=10)

        self.borrar_credenciales_button = tk.Button(self.frame, text="Borrar Credenciales", command=self.borrar_credenciales)
        self.borrar_credenciales_button.pack(pady=10)

        self.metodo_combobox.bind("<<ComboboxSelected>>", self.toggle_harcascade_options)

        self.toggle_harcascade_options()

    def toggle_harcascade_options(self, event=None):
        if self.metodo_combobox.get() == "YOLO":
            self.modelo_haar_combobox.config(state="disabled")
            self.modelo_haar_label.config(state="disabled")
        else:
            self.modelo_haar_combobox.config(state="normal")
            self.modelo_haar_label.config(state="normal")

    def iniciar_deteccion(self):
        metodo_deteccion = self.metodo_combobox.get()
        modelo_haar = self.modelo_haar_combobox.get()
        ip_camara = self.ip_entry.get()
        tiempo_maximo_sin_deteccion = int(self.tiempo_entry.get())

        if ip_camara:
            guardar_ip_camara(ip_camara)

        fuente_video = ip_camara if ip_camara else "video.mp4"
        
        detector = Detector(fuente_video, metodo_deteccion, tiempo_maximo_sin_deteccion, modelo_haar)
        thread = Thread(target=detector.procesar, daemon=True)
        thread.start()

    def abrir_video(self):
        ruta_video = filedialog.askopenfilename()
        if ruta_video:
            self.ip_entry.delete(0, tk.END)
            self.ip_entry.insert(0, ruta_video)
            self.iniciar_deteccion()

    def borrar_credenciales(self):
        grabador = Grabador("", "")
        grabador.borrar_credenciales()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
