import tkinter as tk
from tkinter import font
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from datetime import datetime
import argparse
import time
import cv2
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from keras_facenet import FaceNet
import gspread
from openpyxl import Workbook
from oauth2client.service_account import ServiceAccountCredentials
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def update_time():
    current_time = time.strftime('%H.%M.%S')
    time_label.config(text=current_time)
    root.after(1000, update_time)
    if current_time == "21.00.00":
        date = datetime.now().strftime("%d %B %Y")
        data = spreadsheet.get_all_values()
        workbook = Workbook()
        sheet_excel = workbook.active
        for row in data:
            sheet_excel.append(row)
        workbook.save(f'{date}.xlsx')
        spreadsheet.batch_clear(['C5:I14'])
def terjemahan_hari():
    hari_mapping = {
     "Monday": "Senin",
     "Tuesday": "Selasa",
     "Wednesday": "Rabu",
     "Thursday": "Kamis",
     "Friday": "Jumat",
     "Saturday": "Sabtu",
     "Sunday": "Minggu"}
    day_string = time.strftime("%A")
    return hari_mapping.get(day_string, day_string)

def update_date():
    day = terjemahan_hari()
    date = datetime.now().strftime("%d %B %Y")
    date_label.config(text=f"{day}, {date}")

def set_button_state(button, state, color):
    button.config(state=state, bg=color)

def datang():
    jam = int(time.strftime("%H"))
    menit = int(time.strftime("%M"))
    lanjut = False
    video_capture = cv2.VideoCapture(0)
    time.sleep(2.0)
    start_time = time.time()
    #cek masker
    while True:
        _, frame = video_capture.read()
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred
            if mask > withoutMask:
                label = "Buka masker"
                color = (0, 255, 255)
            else:
                lanjut = True
                break
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 4)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 7)
            remark_label.config(text="Buka masker", fg="black")
        if lanjut:
            break
        cv2.imshow("attendence system", frame)
        if time.time() - start_time > 4:
            current_time = time.strftime('%H.%M.%S')
            time_in_label.config(text=current_time)
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    if not lanjut:
        video_capture.release()
        cv2.destroyAllWindows()
        set_button_state(arrival_button, tk.DISABLED, '#16753D')
        set_button_state(clear_button, tk.NORMAL, '#455a64')
        set_button_state(departure_button, tk.DISABLED, '#0D4577')
    #proses absensi
    else:
        start_time = time.time()
        isi = True
        while True:
            _, frame = video_capture.read()
            rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)
            for x, y, w, h in faces:
                img = rgb_img[y:y + h, x:x + w]
                img = cv2.resize(img, (128, 128))
                img = np.expand_dims(img, axis=0)
                xpred = facenet.embeddings(img)
                distances = np.linalg.norm(X - xpred, axis=1)
                min_distance_index = np.argmin(distances)
                min_distance = distances[min_distance_index]
                predicted_label = Y[min_distance_index]
                threshold = 0.6
                if min_distance <= threshold:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 7)
                    cv2.putText(frame, predicted_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4, cv2.LINE_AA)
                    name_label.config(text=predicted_label)
                    if jam < 8 or (jam == 8 and menit <= 55):
                        remark_label.config(text="Lebih awal", fg="#1E9C43")
                        data_absen = [predicted_label, time.strftime("%H.%M.%S"), "Lebih awal"]
                    elif (jam == 8 and menit > 55) or (jam == 9 and menit <= 5):
                        remark_label.config(text="Tepat waktu", fg="blue")
                        data_absen = [predicted_label, time.strftime("%H.%M.%S"), "Tepat waktu"]
                    else:
                        remark_label.config(text="Terlambat", fg="red")
                        data_absen = [predicted_label, time.strftime("%H.%M.%S"), "Terlambat"]
                    baris = 5
                    while isi:
                        cell_value = spreadsheet.acell(f'C{baris}').value
                        if cell_value == "None":
                            spreadsheet.update([data_absen], f"C{baris}:E{baris}")
                            isi = False
                            break
                        elif cell_value == predicted_label:
                            remark_label.config(text="Sudah absen", fg="black")
                            break
                        else:
                            baris = baris + 1
                else:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 7)
                    cv2.putText(frame, "Tidak dikenali", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4, cv2.LINE_AA)
                    remark_label.config(text="Tidak dikenali", fg="black")
            current_time = time.strftime('%H.%M.%S')

            time_in_label.config(text=current_time)
            cv2.imshow("attendence system", frame)
            if time.time() - start_time > 4:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        video_capture.release()
        cv2.destroyAllWindows()
        set_button_state(arrival_button, tk.DISABLED, '#16753D')
        set_button_state(clear_button, tk.NORMAL, '#455a64')
        set_button_state(departure_button, tk.DISABLED, '#0D4577')

def bersihkan():
    name_label.config(text="")
    time_in_label.config(text="")
    remark_label.config(text="")

    set_button_state(clear_button, tk.DISABLED, '#1A2225')
    set_button_state(arrival_button, tk.NORMAL, '#00c853')
    set_button_state(departure_button, tk.NORMAL, '#1e88e5')

def pulang():
    jam = int(time.strftime("%H"))
    menit = int(time.strftime("%M"))
    lanjut = False
    video_capture = cv2.VideoCapture(0)
    time.sleep(2.0)
    start_time = time.time()
    #cek masker
    while True:
        _, frame = video_capture.read()
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred
            if mask > withoutMask:
                label = "Buka masker"
                color = (0, 255, 255)
            else:
                lanjut = True
                break
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 4)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 7)
            remark_label.config(text="Buka masker", fg="black")
        if lanjut:
            break
        cv2.imshow("attendence system", frame)
        if time.time() - start_time > 4:
            current_time = time.strftime('%H.%M.%S')
            time_in_label.config(text=current_time)
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    if not lanjut:
        video_capture.release()
        cv2.destroyAllWindows()
        set_button_state(departure_button, tk.DISABLED, '#0D4577')
        set_button_state(clear_button, tk.NORMAL, '#455a64')
        set_button_state(arrival_button, tk.DISABLED, '#16753D')
    #proses absen pulang
    else:
        start_time = time.time()
        isi = True
        while True:
            _, frame = video_capture.read()
            rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)
            for x, y, w, h in faces:
                img = rgb_img[y:y + h, x:x + w]
                img = cv2.resize(img, (128, 128))
                img = np.expand_dims(img, axis=0)
                xpred = facenet.embeddings(img)
                distances = np.linalg.norm(X - xpred, axis=1)
                min_distance_index = np.argmin(distances)
                min_distance = distances[min_distance_index]
                predicted_label = Y[min_distance_index]
                threshold = 0.6
                if min_distance <= threshold:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 7)
                    cv2.putText(frame, predicted_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4,cv2.LINE_AA)
                    name_label.config(text=predicted_label)
                    if jam < 16 or (jam == 16 and menit <= 59):
                        remark_label.config(text="Lebih awal", fg="red")
                        data_pulang = [predicted_label, time.strftime("%H.%M.%S")]
                    elif jam == 17 and menit <= 59:
                        remark_label.config(text="Tepat waktu", fg="blue")
                        data_pulang = [predicted_label, time.strftime("%H.%M.%S")]
                    else:
                        remark_label.config(text="Lembur", fg="#1E9C43")
                        data_pulang = [predicted_label, time.strftime("%H.%M.%S")]
                    baris = 0
                    for i in range(5, 15):
                        nama = spreadsheet.acell(f"C{i}").value
                        if data_pulang[0] == nama:
                            baris = i
                            break
                    cell_value = spreadsheet.acell(f"F{baris}").value
                    datang = spreadsheet.acell(f"D{baris}").value
                    format_waktu = "%H.%M.%S"
                    datang = datetime.strptime(datang, format_waktu)
                    pulang = time.strftime('%H.%M.%S')
                    pulang = datetime.strptime(pulang, format_waktu)
                    total = pulang - datang
                    jam_total, sisa_detik = divmod(total.seconds, 3600)
                    menit_total, detik_total = divmod(sisa_detik, 60)
                    total_jam_kerja = f"{jam_total:02}.{menit_total:02}.{detik_total:02}"
                    if jam_total < 8:
                        keterangan = "Lebih awal"
                    elif jam_total == 8:
                        keterangan = "Tepat waktu"
                    else:
                        keterangan = "Lembur"
                    if cell_value == "None":
                        spreadsheet.update([[data_pulang[1], total_jam_kerja, keterangan]], f"F{baris}:H{baris}")
                        isi = False
                    elif cell_value != "None" and isi:
                        remark_label.config(text="Sudah pulang", fg="black")
                else:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 7)
                    cv2.putText(frame, "Tidak dikenali", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4, cv2.LINE_AA)
                    remark_label.config(text="Tidak dikenali", fg="black")
            current_time = time.strftime('%H.%M.%S')

            time_in_label.config(text=current_time)
            cv2.imshow("attendence system", frame)
            if time.time() - start_time > 4:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        video_capture.release()
        cv2.destroyAllWindows()
        set_button_state(departure_button, tk.DISABLED, '#0D4577')
        set_button_state(clear_button, tk.NORMAL, '#455a64')
        set_button_state(arrival_button, tk.DISABLED, '#16753D')

def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()
    faces = []
    locs = []
    preds = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > args["confidence"]:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)
            faces.append(face)
            locs.append((startX, startY, endX, endY))
    if len(faces) > 0:
        preds = maskNet.predict(faces)
    return (locs, preds)

#Memuat embeddings
facenet = FaceNet()
faces_embeddings = np.load("faces_embeddings_done_4classes.npz")
X = faces_embeddings['arr_0']
Y = faces_embeddings['arr_1']

encoder = LabelEncoder()
encoder.fit(Y)
haarcascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str, default="face_detector", help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str, default="mask_detector.model", help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"], "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
maskNet = load_model(args["model"])

scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive.file', 'https://www.googleapis.com/auth/drive']
credentials = ServiceAccountCredentials.from_json_keyfile_name('hadir.json', scope) #change with your own json
gc = gspread.authorize(credentials)
spreadsheet_name = 'data kehadiran' #change with yours
spreadsheet = gc.open(spreadsheet_name).sheet1

day_string = terjemahan_hari()
current_date = datetime.now().strftime("%d %B %Y")
tanggal = f"{day_string}, {current_date}"
spreadsheet.update_cell(3, 3, tanggal)

root = tk.Tk()
root.title("Absen Pegawai")
root.geometry("800x600")
root.configure(bg='#000000')

title_font = font.Font(family='Helvetica', size=40, weight='bold')
label_font = font.Font(family='Helvetica', size=18)
time_font = font.Font(family='Helvetica', size=40)
date_font = font.Font(family='Helvetica', size=24)
button_font = font.Font(family='Helvetica', size=14, weight='bold')

header_frame = tk.Frame(root, bg='#383838', height=50)
header_frame.pack(fill='x')
title_label = tk.Label(header_frame, text="Absen", font=title_font, bg='#383838', fg='white')
title_label.pack(pady=10)
time_label = tk.Label(root, text="", font=time_font, bg='#000000', fg='white')
time_label.pack(pady=(20, 5))
date_label = tk.Label(root, text="", font=date_font, bg='#000000', fg='white')
date_label.pack()

input_frame = tk.Frame(root, bg='#000000')
input_frame.pack(pady=20)

name_label_text = tk.Label(input_frame, text="Nama", font=label_font, bg='#000000', fg='white')
name_label_text.grid(row=0, column=0, sticky='w', padx=20, pady=(0, 20))
name_label = tk.Label(input_frame, text="", font=label_font, bg='#FFFFFF', fg='black', width=30, anchor='w')
name_label.grid(row=0, column=1, padx=20, pady=(0, 20))

time_in_label_text = tk.Label(input_frame, text="Waktu Absen", font=label_font, bg='#000000', fg='white')
time_in_label_text.grid(row=1, column=0, sticky='w', padx=20, pady=(0, 20))
time_in_label = tk.Label(input_frame, text="", font=label_font, bg='#FFFFFF', fg='black', width=30, anchor='w')
time_in_label.grid(row=1, column=1, padx=20, pady=(0, 20))

remark_label_text = tk.Label(input_frame, text="Keterangan", font=label_font, bg='#000000', fg='white')
remark_label_text.grid(row=2, column=0, sticky='w', padx=20, pady=(0, 20))
remark_label = tk.Label(input_frame, text="", font=label_font, bg='#FFFFFF', fg='black', width=30, anchor='w')
remark_label.grid(row=2, column=1, padx=20, pady=(0, 20))

button_frame = tk.Frame(root, bg='#000000')
button_frame.pack(pady=20)

arrival_button = tk.Button(button_frame, text="Datang", font=button_font, bg='#00c853', fg='white', command=datang, height=1, width=13)
arrival_button.grid(row=0, column=0, padx=(0, 28), sticky='w')
clear_button = tk.Button(button_frame, text="Bersihkan", font=button_font, bg='#1A2225', fg='white', command=bersihkan, height=1, width=13, state = tk.DISABLED)
clear_button.grid(row=0, column=1, padx=28)
departure_button = tk.Button(button_frame, text="Pulang", font=button_font, bg='#1e88e5', fg='white', command=pulang, height=1, width=13)
departure_button.grid(row=0, column=2, padx=(28, 0), sticky='e')

update_time()
update_date()
root.mainloop()
