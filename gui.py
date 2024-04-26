import cv2
import numpy as np
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout
import face_recognition

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
    detected_emotion = None

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
            
            # Check if the detected emotion is the desired one
            if predicted_emotions_label == 'happy':
                detected_emotion = current_frame.copy()
                break
        
        cv2.imshow("Webcam", current_frame)
        
        if detected_emotion is not None:
            cv2.imshow("Detected Emotion", detected_emotion)
            cv2.waitKey(0)  # Display the frame until a key is pressed
            break
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    webcam_video_stream.release()
    cv2.destroyAllWindows()

scan()
