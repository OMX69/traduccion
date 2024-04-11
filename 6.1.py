import cv2
import mediapipe as mp
import numpy as np
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.uix.button import Button
from kivy.core.window import Window

# Cargar modelo entrenado y etiquetas
import pickle
import os
print(os.listdir())

model = pickle.load(open('model.p', 'rb'))['model']
labels_dict = {0: 'A', 1: 'B', 2: 'L'}

# Inicializar detección de manos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

class HandSignTranslatorApp(App):
    def build(self):
        # Colores
        primary_color = (0.247, 0.317, 0.71, 1)  # Azul oscuro
        secondary_color = (0.31, 0.71, 0.89, 1)  # Azul claro

        # Establecer color de fondo azul oscuro
        Window.clearcolor = primary_color

        layout = BoxLayout(orientation='vertical')

        # Relleno para espaciar el contenido de la parte superior de la pantalla
        layout.add_widget(Label(size_hint=(1, 0.05)))

        # Título del proyecto
        title_label = Label(text='Traductor de Señas', font_size='24sp', color=(1, 1, 1, 1))
        layout.add_widget(title_label)

        # Cámara
        self.camera_image = Image(size_hint=(1, 0.65))
        layout.add_widget(self.camera_image)

        # Letras predichas
        self.predicted_character_label = Label(text='', font_size='48sp', color=(1, 1, 1, 1))
        layout.add_widget(self.predicted_character_label)

        # Botón de encendido/apagado de la cámara
        self.camera_button = Button(text='Encender cámara', size_hint=(1, 0.1), background_color=secondary_color)
        self.camera_button.bind(on_press=self.toggle_camera)
        layout.add_widget(self.camera_button)

        Clock.schedule_interval(self.update_camera, 1.0 / 30.0)

        return layout

    def on_start(self):
        self.adjust_background_color()

    def adjust_background_color(self):
        # Cambiar color de fondo de la ventana
        self.root_window.background_color = (0.247, 0.317, 0.71, 1)

    def update_camera(self, dt):
        if hasattr(self, 'cap') and self.cap is not None:
            ret, frame = self.cap.read()

            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        data_aux = []
                        x_ = []
                        y_ = []

                        for i in range(len(hand_landmarks.landmark)):
                            x = hand_landmarks.landmark[i].x
                            y = hand_landmarks.landmark[i].y

                            x_.append(x)
                            y_.append(y)

                        for i in range(len(hand_landmarks.landmark)):
                            x = hand_landmarks.landmark[i].x
                            y = hand_landmarks.landmark[i].y
                            data_aux.append(x - min(x_))
                            data_aux.append(y - min(y_))

                        prediction = model.predict([np.asarray(data_aux)])

                        predicted_character = labels_dict[int(prediction[0])]
                        self.predicted_character_label.text = predicted_character

                frame = cv2.flip(frame, 1)
                buf1 = cv2.flip(frame, 0)
                buf = buf1.tobytes()  # Cambio aquí
                texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
                texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
                self.camera_image.texture = texture

    def toggle_camera(self, instance):
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
            self.cap = None
            self.camera_button.text = 'Encender cámara'
        else:
            self.cap = cv2.VideoCapture(0)
            self.camera_button.text = 'Apagar cámara'

if __name__ == '__main__':
    HandSignTranslatorApp().run()
