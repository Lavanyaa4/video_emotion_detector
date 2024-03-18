import tkinter as tk 
from tkinter import filedialog
from tkinter import * 

from tensorflow.keras.models import model_from_json   
from PIL import Image, ImageTk
import numpy as np 
import cv2

#importing the haarcascade_frontface_default.xml file from github


def FacialExpressionModel(json_file, weights_file):
    with open(json_file, "r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)

    model.load_weights(weights_file)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

top = tk.Tk()
top.geometry('800x600')
top.title('Video Emotion Detector')
top.configure(background='#CDCDCD')

label1 = Label(top, background='#CDCDCD', font=('times new roman', 16, 'italic'))
sign_image = Label(top)

facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
model = FacialExpressionModel("model.json", "model_weights.h5")

EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

stop_button = Button(top, text="STOP DETECTING", command=top.destroy, padx=10, pady=5, bg="#364156", fg='red', font=('times new roman', 20, 'italic'))  
stop_button.pack(side='bottom', pady=10)


def Detect(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray_image, 1.3, 5)
    try:
        for (x, y, w, h) in faces:
            fc = gray_image[y:y+h, x:x+w]
            roi = cv2.resize(fc, (48, 48))
            pred = EMOTIONS_LIST[np.argmax(model.predict(roi[np.newaxis,:,:,np.newaxis]))]
            print("The Emotion is " + pred)
            label1.configure(foreground="#011638", text=pred)
    except: 
        label1.configure(foreground="#011638", text="Unable to detect")

def show_Detect_button():
    detect_b = Button(top, text="Detect the Emotion", command=detect_emotion, padx=10, pady=5)
    detect_b.configure(background="#364156", foreground='white', font=('times new roman', 10, 'italic bold'))
    detect_b.place(relx=0.79, rely=0.46)
    
def upload_video():
    try:
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi")])
        detect_emotion_from_video(file_path)
    except Exception as e:
        print("Error:", e)
        
def detect_emotion_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        Detect(frame)
        cv2.imshow('Video Emotion Detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()

upload = Button(top, text="Upload a video", command=upload_video, padx=10, pady=5)
upload.configure(background="#364156", foreground='white', font=('times new roman', 20, 'italic bold'))  
upload.pack(side='bottom', pady=50)
sign_image.pack(side='bottom', expand='True')
label1.pack(side='bottom', expand='True')
heading = Label(top, text='Video Emotion Detector', pady=20, font=('times new roman', 25, 'italic bold'))
heading.configure(background='#CDCDCD', foreground='#364156')
heading.pack()

top.mainloop()