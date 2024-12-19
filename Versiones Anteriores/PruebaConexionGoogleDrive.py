from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import os

gauth = GoogleAuth()
gauth.LocalWebserverAuth()  

drive = GoogleDrive(gauth)

file1 = drive.CreateFile({'title': 'video.mp4'})  
file1.Upload()  print("Archivo subido exitosamente")
