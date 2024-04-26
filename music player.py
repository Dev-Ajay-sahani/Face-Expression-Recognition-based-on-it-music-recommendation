import os
import tkinter as tk
from tkinter import filedialog
import pygame

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
main("happy","Music\Happy_Songs")
