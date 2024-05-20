# Music Recommender Based on Facial Emotion Recognition

This project is a Python application that recommends music based on real-time facial emotion recognition. It uses a convolutional neural network (CNN) to detect emotions from a webcam feed and plays music corresponding to the detected emotion.

## Features

- **Real-time Facial Emotion Recognition**: Uses a pre-trained CNN model to identify emotions from a live webcam feed.
- **Music Recommendation**: Recommends and plays music based on the detected emotion (e.g., happy, sad, angry).
- **Simple GUI**: Provides a user-friendly interface using `tkinter` and `customtkinter`.

## Installation

### Prerequisites

- Python 3.6 or higher
- Virtual environment (recommended)

### Required Libraries

Install the required libraries using pip:

```bash
pip install customtkinter Pillow opencv-python numpy keras face_recognition pygame
```

### Clone the Repository

```bash
git clone https://github.com/yourusername/music-recommender-emotion.git
cd music-recommender-emotion
```

### Model Weights

Download the pre-trained model weights and place them in the `assets` directory:

- [facial_expression_model_weights.h5](https://link-to-your-model-weights)

## Usage

Run the application:

```bash
python main.py
```

### GUI Elements

- **Start Scan Button**: Starts the webcam feed for emotion detection.
- **Music Player Controls**: Play, pause, stop, and navigate through recommended songs.

### Keyboard Controls

- **Press `E`**: Exit the webcam feed and the program.

## Directory Structure

```
music-recommender-emotion/
├── assets/
│   ├── adgips.ico
│   ├── bg.png
│   ├── frame0/
│   │   └── Play.png
│   └── facial_expression_model_weights.h5
├── Music/
│   ├── Happy_Songs/
│   ├── Sad_Songs/
│   ├── Angry_Songs/
│   ├── Fear_Songs/
│   └── Surprise_Songs/
├── main.py
└── README.md
```

- **assets**: Contains icons, background images, and model weights.
- **Music**: Contains folders with songs categorized by emotions.

## Code Overview

- **main.py**: Contains the complete implementation of the application.
  - **load_songs(folder_path)**: Loads songs from a specified folder.
  - **play_song(folder_path, song_list, current_song_index)**: Plays a selected song.
  - **pause_song()**: Pauses the current song.
  - **stop_song()**: Stops the current song.
  - **play_next_song(folder_path, song_list, current_song_index)**: Plays the next song in the list.
  - **play_previous_song(folder_path, song_list, current_song_index)**: Plays the previous song in the list.
  - **main(Emotion, folder_path)**: Initializes the music player GUI.
  - **scan()**: Captures webcam feed, detects faces, predicts emotions, and launches the music player based on the predicted emotion.

## Contributing

Feel free to fork this repository, make changes, and submit pull requests. Any contributions are welcome!

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
