# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 11:07:20 2024

@author: kristina

NEW images in image_rc : pyrcc5 -o image_rc.py image.qrc
location: D:\ALL_PROJECT\_ABc\py39y7_talk2me\yolov7
activate py39y7

10.12.24: webcam zu Frame  zu text in 3 Threads  in einer pyqt5 gui  wunderbar.. das geht morgen zu github
Features:
    Live Webcam Feed: Captures video frames in real-time.
    Ollama Integration: Processes frames to generate natural language descriptions of the scene.
    Text-to-Speech (TTS): Reads out the descriptions in real-time using a TTS engine.
    Thread Synchronization: Efficient management of webcam, processing, and TTS tasks using a custom status manager (tts_done, tts_go, tts_busy).
    PyQt5 GUI: Intuitive graphical user interface (GUI) to display the webcam feed and generated text.

******************************
pip freeze > requirements.txt
besser:
    pipreqs --force    erzeugt eine kompakte   requirement.txt!!
******************************

"""


from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot
from PyQt5 import QtCore,  QtWidgets, uic
from PyQt5.QtCore import  QObject
from PyQt5.QtGui import QImage, QPixmap

import image_rc  #   

import os
import cv2
import sys


import ollama

#für tts und sound text vorlesen
import wave
import sounddevice as sd
import io

# import all the modules that we will need to use
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer

# stuff for YOLOv7
from collections import Counter
import torch
import numpy as np
# Load YOLOv7 model
#model = torch.hub.load('WongKinYiu/yolov7', 'custom', 'yolov7.pt')  # Update with path to your weights
model = torch.hub.load('WongKinYiu/yolov7', 'custom', 'yolov7-e6.pt')  # Update with path to your weights


from threading import Event

from toolsClass import tools 
tools = tools()   # class ocr

_BaseDir = os.getcwd()
_UI_FILE = os.path.join(_BaseDir,"_img2speachUI.ui" )

tts_done_event = Event()



#-- ALL 4 YOLO ---------------------------------------------------------------------------------
def get_scores(results):
  # translate boxes data from a Tensor to the List of boxes info lists
  boxes_list = results.pandas().xyxy[0]
  boxes_list = boxes_list.values.tolist()
  #columns = ['x_min', 'y_min', 'x_max', 'y_max', 'confidence', 'class_id', 'className']
  # iterate through the list of boxes info and make some formatting  
  count = 0
  if boxes_list:
      for i in boxes_list:        
        print ("box: " + str(count) + " Obj: " + str(i[6])  + " ( " + str(i[5]) + " ) "  +  "  score: "  + str(round( i[4],2)))
        count+=1


def generate_text_from_counts(counts):
    """
    Generiert einen beschreibenden Text basierend auf den gezählten Objekten.
    """
    if not counts:
        return "Es wurden keine Objekte erkannt."

    object_texts = [f"{count} {name}" for name, count in counts.items()]
    text = "Wir sehen " + ", ".join(object_texts[:-1])
    if len(object_texts) > 1:
        text += " und " + object_texts[-1]
    else:
        text += object_texts[0]
    return text


def render_filtered_results(frame, filtered_results):
    """
    Zeichnet nur die gefilterten Bounding Boxes und Labels auf das Bild.
    
    :param frame: Originalbild (NumPy-Array).
    :param filtered_results: Gefilterter DataFrame aus YOLO-Ergebnissen.
    :return: Bild mit gefilterten Bounding Boxes.
    """
    for _, row in filtered_results.iterrows():
        # Extrahiere Bounding Box und Label
        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        label = f"{row['name']} ({row['confidence']:.2f})"

        # Zeichne die Bounding Box
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  # Grün
        # Füge das Label hinzu
        cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame

# Anwendung mit YOLO
def process_frame_with_yolo_and_render(model, frame, score_threshold=0.5):
    # Anwenden des Modells
    results = model(frame)
    
    # Ergebnisse filtern
    filtered_results = filter_results_by_score(results, score_threshold)
    
    # Textbeschreibung generieren (optional)
    counts = Counter(filtered_results['name'])
    description = generate_text_from_counts(counts)

    # Gefilterte Visualisierung erstellen
    rendered_frame = render_filtered_results(frame.copy(), filtered_results)

    return rendered_frame, description

def get_scores(results):
  # translate boxes data from a Tensor to the List of boxes info lists
  boxes_list = results.pandas().xyxy[0]
  boxes_list = boxes_list.values.tolist()
  #columns = ['x_min', 'y_min', 'x_max', 'y_max', 'confidence', 'class_id', 'className']
  # iterate through the list of boxes info and make some formatting
  
  count = 0
  if boxes_list:
      for i in boxes_list:        
        print ("box: " + str(count) + " Obj: " + str(i[6])  + " ( " + str(i[5]) + " ) "  +  "  score: "  + str(round( i[4],2)))
        count+=1
        
def filter_results_by_score(results, score_threshold=0.5):
    """
    Filtert die YOLO-Ergebnisse basierend auf einem minimalen Confidence-Score.    
    :param results: YOLO-Ergebnisse als Detections-Objekt.
    :param score_threshold: Der minimale Confidence-Score, um ein Objekt zu berücksichtigen.
    :return: Gefilterte Liste der erkannten Objekte.
    """
    # Konvertiere Ergebnisse in einen DataFrame
    df = results.pandas().xyxy[0]    
    # Filtere nach Confidence-Score
    filtered_df = df[df['confidence'] >= score_threshold]    
    # Optional: Konvertiere zurück in eine Liste oder arbeite mit dem DataFrame
    return filtered_df  

def combine_yolo_and_ollama(yolo_description, ollama_response):
    """
    Kombiniert die YOLO-Beschreibung mit der Ollama-Interpretation.
    """
    combined_description = (
        f"YOLO erkennt folgende Gegenstände: {yolo_description}. "
        f"Ollama interpretiert das Bild wie folgt: {ollama_response}"
    )
    return combined_description

#-----------------------------------------------------------------------------------------------


class StatusManager(QObject):
    status_changed = pyqtSignal(str)  # Signal, das den Status weitergibt

    def __init__(self):
        super().__init__()
        self._status = "tts_done"  # Initialstatus

    def set_status(self, value: str):
        self._status = value
        self.status_changed.emit(value)  # Signal senden

    def get_status(self) -> str:
        return self._status  # Status abfragen



class WebcamThread(QThread):
    frameCaptured = pyqtSignal(np.ndarray)  # Signal to emit captured frames

    def run(self):
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set 720p resolution
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 10)            # Reduce FPS for efficiency
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)      # Avoid buffering delays

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                
                self.frameCaptured.emit(frame)  # Emit the captured frame --> ollama
            else:
                break
        cap.release()


class OllamaThread(QThread):    
    #Das sendet der thead für das TTS und für den TextEditor
    frameProcessed = pyqtSignal(str)  # Signal to emit Ollama response - zur anzeige im pyqt textEditor
       
    # Das empfängt der thread zur weiterverarbeitung
    frameCaptured = pyqtSignal(np.ndarray)  # Signal to emit captured frames from webcam
    
    #direkt die pyqt felder beschreiben geht nicht
    log_signal = pyqtSignal(str)  # Signal zum Senden von Log-Nachrichten
    
    frameCaptured4Yolo = pyqtSignal(np.ndarray) # emit yolo object detection frame 

    
    def __init__(self,name, tts_done_event, status_manager):
        super().__init__()
        self.name = name
        self.frame = None
        self.running = True
        self.tts_done_event = tts_done_event
        self.status_manager = status_manager
        
        
    def set_thread_status(self, status):
        new_status = f"{status}"        
        self.log_signal.emit(f"ollama /  Setze Status auf {new_status}") 
        self.status_manager.set_status(new_status)
        
    

    def set_frame(self, frame):
       if self.status_manager.get_status() == "tts_done":
           self.frame = frame

    def run(self):
        while self.running:            
            
            if  self.frame is not None and self.status_manager.get_status() == "tts_done" :     
                
                frameC = self.frame.copy()
                
                current_status = self.status_manager.get_status()
                self.log_signal.emit(f"ollama: TTS meldet: {current_status}") 
                
                resized_frame = cv2.resize(self.frame, (384, 384))
                #frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                self.frame = resized_frame
                
                # Convert frame to bytes
                _, buffer = cv2.imencode('.jpg', self.frame)
                frame_bytes = buffer.tobytes()
                
                print("Ollama: Verarbeite neues Bild")
                self.log_signal.emit('Ollama: Verarbeite neues Bild')  # Signal senden
                     
                
                #---------- show the img which is currently in ollama analysis
                # Convert bytes to a NumPy array
                np_arrayFrame = np.frombuffer(frame_bytes, np.uint8)
                self.frameCaptured.emit(np_arrayFrame )  # Emit the captured frame  --> img als np array
                #--------------------------------------------------------------
                            
                
                # Call Ollama for image description - SLOW --------------------
                try:
                    res = ollama.chat(
                        model="llava",
                        messages=[
                            {
                                "role": "user",
                                "content": "Was ist auf diesem Bild zu sehen? (in Deutsch)",
                                "images": [frame_bytes],
                            }
                        ],
                    )
                    ollamaResponse = res["message"]["content"]
                except Exception as e:
                    ollamaResponse= f"Ollama error: {str(e)}"    
                
                #---- YOLO make detections and send img emit ------------------                
               
                # Verarbeite den Frame
                yFrame, yoloDescription = process_frame_with_yolo_and_render(model, frameC, score_threshold=0.6)
                # Textbeschreibung ausgeben
                self.log_signal.emit(yoloDescription)
                
                self.frameCaptured4Yolo.emit(yFrame )  # Emit the captured frame  --> img als np array         
                
                combined_description = combine_yolo_and_ollama(yoloDescription, ollamaResponse)
    

                
               
                #--------------------------------------------------------------   
                
                print("Ollama:Jetzt wird response zum TTS übergebn.....")
                self.log_signal.emit("Ollama: Jetzt wird response zum TTS übergebn.....")  # Signal senden
                
                self.set_thread_status("tts_go")
                self.frameProcessed.emit( combined_description )  # Emit the processed response  --> ollama text-ergebnis
                
                print("Ollama: Warten auf TTS...")
                self.log_signal.emit("Ollama: Warten auf TTS. - tts_done_event.wait()  ")  # Signal senden
               
                self.tts_done_event.wait()  # Warten, bis TTS fertig ist
                self.tts_done_event.clear()  # Zurücksetzen des Flags
                
                self.log_signal.emit("Ollama: Warten auf TTS. - tts_done_event.clear()  ")  # Signal senden
                
                # Frame zurücksetzen & response None
                ollamaResponsee = None
                yoloDescription = None
                res = None
                self.frame = None
               
                
class text2SpeachThread (QThread): # TTS
    #direkt die pyqt felder beschreiben geht nicht
    log_signal = pyqtSignal(str)  # Signal zum Senden von Log-Nachrichten

        
    def __init__(self, name, tts_done_event, status_manager):
        super().__init__()
        self.name = name
        self.imgText = None
        self.running = True
        self.tts_done_event = tts_done_event
        self.status_manager = status_manager

    def set_imgText(self, imgText):
        self.log_signal.emit("TTS: set_imgText()")         
        self.imgText = imgText
        
        current_status = self.status_manager.get_status()
        self.log_signal.emit(f"tts: current state::  {current_status}") 
       

    def set_thread_status(self, status):
        new_status = f"{status}"
        self.log_signal.emit(f"TTS /  Setze Status auf {new_status}") 
        self.status_manager.set_status(new_status)
        
    

    def run(self):
        while self.running:
            if self.imgText is not None and self.status_manager.get_status() == "tts_go" :                
                
                try:
                    
                    self.set_thread_status("tts_busy")
                    
                    print("HALLO: sind in der text2Speach method.")
                    # Pfade zu den Dateien
                    path = "modelsTTS/.models.json"

                    model_manager = ModelManager(path)
                    #https://mbarnig.github.io/TTS-Models-Comparison/
                    model_path, config_path, model_item = model_manager.download_model("tts_models/de/thorsten/tacotron2-DDC")
                    voc_path, voc_config_path, _ = model_manager.download_model(model_item["default_vocoder"])
                    print("Sprachdatei wurde offline erstellt.")
                    self.log_signal.emit("TTS: Sprachdatei wurde offline erstellt.")    
                    
                    syn = Synthesizer(
                        tts_checkpoint=model_path,
                        tts_config_path=config_path,
                        vocoder_checkpoint=voc_path,
                        vocoder_config=voc_config_path
                    )

                    #text = "das ist ein deutscher text - vorgelesen vom computer"
                    #text = res ['message']['content']

                    outputs = syn.tts(self.imgText)
                    
                    syn.save_wav(outputs, "audio-1.wav")
                    print("HALLO: Die AUDIO datei ist erstellt! audio-1.wav")
                    self.log_signal.emit("TTS: Die AUDIO datei ist erstellt! audio-1.wav")    
                    
                    # Play the audio file
                    # Beispiel: WAV-Daten in einer Variable speichern
                    with open("audio-1.wav", "rb") as wav_file:
                        wav_data = wav_file.read()
                        # WAV-Daten direkt abspielen
                        # Erstelle einen Datei-ähnlichen Stream aus den Bytes
                        wav_stream = io.BytesIO(wav_data)

                        # Öffne den Stream mit der wave-Bibliothek
                        with wave.open(wav_stream, 'rb') as wav_file:
                            # Extrahiere Parameter und Daten
                            sample_rate = wav_file.getframerate()
                            audio_data = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype=np.int16)

                            # Spiele das Audio ab
                            print("TTS: Spricht...")
                            self.log_signal.emit("TTS: Spricht...")   
                            sd.play(audio_data, samplerate=sample_rate)
                            sd.wait()  # Warte, bis die Wiedergabe abgeschlossen ist
                            print("TTS: Audio wiedergabe ist beendet!")
                            self.log_signal.emit("TTS: Audio wiedergabe ist beendet!")    
                            
                    self.log_signal.emit("TTS: Fertig, sende Signal an Ollama. - self.tts_done_event.set()")   
                    print("TTS: Fertig, sende Signal an Ollama")
                    self.tts_done_event.set()  # Signal an Ollama senden    
                    self.log_signal.emit("TTS: setze tts_done ")   
                    
                    self.set_thread_status("tts_done")
                    
                        
                except Exception as e:
                    response = f"text2SpeachThread - ERROR: {str(e)}"
                    print(response)
                    print("ERR TTS: Fertig, sende Signal an Ollama")
                    self.tts_done_event.set()  # Signal an Ollama senden   

    
    

class MainWindow_ALI(QtWidgets.QMainWindow):    
    def __init__(self):
        #super(MainWindow_ALI,self).__init__()
        super().__init__()

       
        # load ui file
        try:  
            uic.loadUi( _UI_FILE, self)
        except Exception as e:
            print(e)
       
        # Status-Manager
        status_manager = StatusManager()
        
        
        # Create threads
        self.webcamThread = WebcamThread()
        self.ollamaThread = OllamaThread("ollama", tts_done_event, status_manager)
        self.text2SpeachThread  = text2SpeachThread("tts", tts_done_event, status_manager) # TTS

        # Connect signals
        self.webcamThread.frameCaptured.connect(self.display_frame)     # webcam auf der GUI
        self.webcamThread.frameCaptured.connect(self.send_to_ollama)
       
        self.ollamaThread.frameProcessed.connect(self.update_response)   # GUI   textBrowser_txtFromOllama
        self.ollamaThread.frameCaptured.connect(self.display_analyseFrame)      # self.label_img4OllamaAnanlsys / img zeigt das zu analysierende bild
        self.ollamaThread.frameCaptured4Yolo.connect(self.display_yoloFrame) # display label_yoloImg
        self.ollamaThread.log_signal.connect(self.append_log)               # Verbinde das Signal mit einem Slot
       
        self.text2SpeachThread.log_signal.connect(self.append_log) # Verbinde das Signal mit einem Slot/LOG
        
        # Thread Pros
        
        self.webcamThread.setPriority(QThread.HighPriority)
        self.ollamaThread.setPriority(QThread.NormalPriority)
        self.text2SpeachThread.setPriority(QThread.LowPriority)

        
        # Start threads
        self.webcamThread.start()
        self.ollamaThread.start()
        self.text2SpeachThread.start()
       
        # set control_bt callback clicked  function
       
        self.pushButton_StartStopCam.hide() # brauchen start cam nicht mehr
                      
        self.textBrowser_log.clear()
        self.textBrowser_log.setText(tools.dt() + 'Welcome in Image 2 Speach' ) #Append text to the GUI
        self.textBrowser_log.append('1. Please start Webcam' )        
        
        self.textBrowser_txtFromOllama.clear()
        self.textBrowser_txtFromOllama.append('Text from Ollama...' ) 
        
    
        
    @pyqtSlot(np.ndarray)
    def display_analyseFrame(self, frame):        # Das ist der Frame der von Ollama analysiert wird!
        # Decode the NumPy array to an image
        frameFromOllama = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        rgb_image = cv2.cvtColor(frameFromOllama, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        pixmap = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = pixmap.scaled(400,400, QtCore.Qt.KeepAspectRatio)   
        self.label_img4OllamaAnanlsys.setPixmap(QPixmap.fromImage(pixmap))   
               
        
        
    @pyqtSlot(np.ndarray)
    def display_frame(self, frame):   #   --------------> WEBCAM
        # Convert frame to QImage for display
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        pixmap = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = pixmap.scaled(600,600, QtCore.Qt.KeepAspectRatio) 
        self.lbl_webcam.setPixmap(QPixmap.fromImage(pixmap))
        
       
    @pyqtSlot(np.ndarray)
    def display_yoloFrame(self, frame):   #   --------------> YOLO FRAME
        
        # Verarbeite den Frame
        '''
        yFrame, description = process_frame_with_yolo_and_render(model, frame, score_threshold=0.6)
        # Textbeschreibung ausgeben
        print(description)       
        self.textBrowser_log.append(description) 
        '''
        
        # Convert frame to QImage for display
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        pixmap = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = pixmap.scaled(600,600, QtCore.Qt.KeepAspectRatio) 
        self.label_yoloImg.setPixmap(QPixmap.fromImage(pixmap))
        
        
        

    @pyqtSlot(np.ndarray)
    def send_to_ollama(self, frame):
        self.ollamaThread.set_frame(frame)

    @pyqtSlot(str)
    def update_response(self, response):
        self.textBrowser_txtFromOllama.append(response ) 
        self.textBrowser_txtFromOllama.append("------------- Next Image --- Next Response ------------------\n") 
        self.text2SpeachThread.set_imgText(response) 
       
          
    @pyqtSlot(str)
    def append_log(self, message):
        self.textBrowser_log.append(message)  # GUI-Element im Hauptthread aktualisieren    

    def closeEvent(self, event):
        # Stop threads on close
        self.webcamThread.terminate()
        self.ollamaThread.running = False
        self.ollamaThread.quit()
        self.text2SpeachThread.running = False
        self.text2SpeachThread.quit()
        event.accept()
   
            

#THIS sector is needed for stand alone mode 
        
def app():
    app = QtWidgets.QApplication(sys.argv)        
    win = MainWindow_ALI()
    win.show()    
    sys.exit(app.exec_())

app()   

 
        