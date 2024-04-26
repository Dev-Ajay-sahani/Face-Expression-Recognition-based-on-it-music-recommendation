from customtkinter import * 
from PIL import Image , ImageTk
import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout
import face_recognition
from tkinter import filedialog
import pygame
import os
# Music Player
def load_songs(folder_path):
    return [file for file in os.listdir(folder_path) if file.endswith(".mp3")]

def play_song(folder_path, song_list, current_song_index):
    song_path = os.path.join(folder_path, song_list[current_song_index])
    pygame.mixer.music.load(song_path)
    pygame.mixer.music.play()

def pause_song():
    pygame.mixer.music.pause()

def stop_song():
    pygame.mixer.music.stop()

def play_next_song(folder_path, song_list, current_song_index):
    current_song_index = (current_song_index + 1) % len(song_list)
    stop_song()
    play_song(folder_path, song_list, current_song_index)
    return current_song_index

def play_previous_song(folder_path, song_list, current_song_index):
    current_song_index = (current_song_index - 1) % len(song_list)
    stop_song()
    play_song(folder_path, song_list, current_song_index)
    return current_song_index

def main(Emotion,folder_path):
    root = tk.Tk()
    root.title(" Song's Music Player")
    root.geometry("500x300")
    root.configure(background="white")

    song_list = load_songs(folder_path)
    current_song_index = 0

    # Initialize pygame
    pygame.init()

    # Create GUI elements
    label = tk.Label(root, text="Music Player")
    label.pack(pady=10)

    music_listbox = tk.Listbox(root)
    music_listbox.pack(expand=True, fill=tk.BOTH, padx=10, pady=5)
    for song in song_list:
        music_listbox.insert(tk.END, song)

    play_pause_button = tk.Button(root, text="Play", command=lambda: play_song(folder_path, song_list, current_song_index))
    play_pause_button.pack(pady=5)

    backward_button = tk.Button(root, text="Previous", command=lambda: play_previous_song(folder_path, song_list, current_song_index))
    backward_button.pack(side=tk.LEFT, padx=5)

    forward_button = tk.Button(root, text="Next", command=lambda: play_next_song(folder_path, song_list, current_song_index))
    forward_button.pack(side=tk.RIGHT, padx=5)

    stop_button = tk.Button(root, text="Stop", command=stop_song)
    stop_button.pack(pady=5)

    root.mainloop()
def scan():
    face_exp_model = Sequential([
        Conv2D(64, (5, 5), activation='relu', input_shape=(48, 48, 1)),
        MaxPooling2D(pool_size=(5, 5), strides=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        AveragePooling2D(pool_size=(3, 3), strides=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        Conv2D(128, (3, 3), activation='relu'),
        AveragePooling2D(pool_size=(3, 3), strides=(2, 2)),
        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.2),
        Dense(1024, activation='relu'),
        Dropout(0.2),
        Dense(7, activation='softmax')
    ])
    face_exp_model.load_weights('facial_expression_model_weights.h5')

    emotions_label = ('angry','disgust','fear','happy','sad','surprise','neutral')
    all_face_location = []
    webcam_video_stream = cv2.VideoCapture(0)

    while True:
        ret, current_frame = webcam_video_stream.read()
        current_frame_small = cv2.resize(current_frame, (0, 0), fx=0.25, fy=0.25)
        all_face_location = face_recognition.face_locations(current_frame_small, model='hog')
        
        for index, current_face_location in enumerate(all_face_location):
            top_pos, right_pos, bottom_pos, left_pos = current_face_location
            top_pos = top_pos * 4
            bottom_pos = bottom_pos * 4
            right_pos = right_pos * 4
            left_pos = left_pos * 4
            print('found face {} at top {}, right {}, bottom {}, left {}'.format(index + 1, top_pos, right_pos, bottom_pos, left_pos))
            
            cv2.rectangle(current_frame, (left_pos, top_pos), (right_pos, bottom_pos), (0, 0, 255), 2)
            
            current_face_image = current_frame[top_pos:bottom_pos, left_pos:right_pos]
            current_face_image = cv2.cvtColor(current_face_image, cv2.COLOR_BGR2GRAY)
            current_face_image = cv2.resize(current_face_image, (48, 48))
            
            img_pixels = image.img_to_array(current_face_image)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255
            
            # Prediction
            exp_predictions = face_exp_model.predict(img_pixels)
            max_index = np.argmax(exp_predictions[0])
            predicted_emotions_label = emotions_label[max_index]
            
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(current_frame, predicted_emotions_label, (left_pos, bottom_pos), font, 1, (255, 255, 255), 1)
            if predicted_emotions_label == 'happy':
                cv2.imshow("Webcam", current_frame)
                cv2.waitKey(0)
                main(predicted_emotions_label,"Music\Happy_Songs")
            elif predicted_emotions_label == 'angry':
                cv2.imshow("Webcam", current_frame)
                cv2.waitKey(0)
                main(predicted_emotions_label,"Music\Angry_Songs")
            elif predicted_emotions_label == 'fear':
                cv2.imshow("Webcam", current_frame)
                cv2.waitKey(0)
                main(predicted_emotions_label,"Music\Fear_Songs")
            elif predicted_emotions_label == 'sad':
                cv2.imshow("Webcam", current_frame)
                cv2.waitKey(0)
                main(predicted_emotions_label,"Music\Sad_Songs")
            elif predicted_emotions_label == 'surprise':
                cv2.imshow("Webcam", current_frame)
                cv2.waitKey(0)
                main(predicted_emotions_label,"Music\Surprise_Songs")
                
        cv2.imshow("Webcam", current_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('e'):
            break
    webcam_video_stream.release()
    cv2.destroyAllWindows()
app = CTk()
app.geometry("900x640")
app.iconbitmap("assets/adgips.ico")
app.resizable(height=False,width=False)
app.title("Music Recommender Based on Facial Emotion Recognition")
img = Image.open("Screenshot 2024-04-23 153327.png").resize((1150,800))
image_tk = ImageTk.PhotoImage(img)
Start_img = Image.open("assets/frame0/Play.png")
image_tk = ImageTk.PhotoImage(img)
Stop_img = Image.open("Screenshot 2024-04-23 153327.png")
image_tk = ImageTk.PhotoImage(img)
label1=CTkLabel(app,text="",image= image_tk)
label1.pack()
btn = CTkButton(master=app,
                text="START SCAN",corner_radius=32,hover_color="#4158D0",fg_color="#c850c0",
                image=CTkImage(dark_image=Start_img),
                font=("Helvetica", 32),# Increase font size
                width=20,
                command=scan)
btn.place(relx=0.5,rely=0.7,anchor ="center")
label_text = "Press E to Exit The Program"
label = CTkLabel(master=app,
                 text=label_text,
                 font=("Helvetica", 32),
                 fg_color="#c850c0")
label.place(relx=0.5, rely=0.8, anchor="center")
app.mainloop()
