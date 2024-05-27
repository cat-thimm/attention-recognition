import queue
import csv
import cv2
import time
import threading
import tkinter as tk
from insightface.app import FaceAnalysis
from PIL import Image, ImageDraw, ImageTk
from datetime import datetime
import numpy as np

class AttentionModel:
    def __init__(self, allow_gpu=True):
        provider = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if allow_gpu else ['CPUExecutionProvider']
        self.__detection = FaceAnalysis(name='buffalo_sc', allowed_modules=['detection'], providers=provider)
        self.__detection.prepare(ctx_id=0, det_thresh=0.60, det_size=(640, 480))

    def infer_attention(self, frame):
        faces = self.__detection.get(frame)
        if faces:
            for face in faces:
                box = face['bbox'].astype(int)
                keypoints = face['kps']

                left_eye = keypoints[0]
                right_eye = keypoints[1]
                nose = keypoints[2]
                left_mouth = keypoints[3]
                right_mouth = keypoints[4]

                # Calculate the center points
                eye_center = (left_eye + right_eye) / 2
                mouth_center = (left_mouth + right_mouth) / 2

                # Calculate the horizontal and vertical distances
                horizontal_distance = eye_center[0] - nose[0]
                vertical_distance = eye_center[1] - nose[1]
                vertical_distance_mouth = mouth_center[1] - nose[1]

                print("Horizontal ", horizontal_distance, "Vertikal ", vertical_distance)

                # Determine attention based on relative positions
                if abs(horizontal_distance) < 10 and abs(vertical_distance) < 40 :
                    attention_state = "Attentive"
                    color = "green"
                elif horizontal_distance > 15:
                    attention_state = "Not Attentive: Looking Left"
                    color = "red"
                elif horizontal_distance < -15:
                    attention_state = "Not Attentive: Looking Right"
                    color = "red"
                elif vertical_distance > -40:
                    attention_state = "Not Attentive: Looking Up"
                    color = "red"
                elif vertical_distance < -15:
                    attention_state = "Not Attentive: Looking Down"
                    color = "red"
                else:
                    attention_state = "Not Attentive"
                    color = "red"

                return attention_state, box, color
        return "Not Attentive", None, "red"


class AttentionView(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.__presenter = AttentionPresenter()

        self.wm_title("Attention Classification")
        self.container = tk.Frame(self)
        self.container.pack(side="top", fill="both", expand=True)
        self.panel = None
        self.record_button = tk.Button(self, text="Start Recording", command=self.record_pressed)
        self.record_button.pack(side="bottom", fill="both", pady=10, padx=10)
        self.protocol("WM_DELETE_WINDOW", self.close_application)

        self.__load_frame()

    def __load_frame(self):
        if not self.__presenter.img_queue.empty():
            image = ImageTk.PhotoImage(self.__presenter.img_queue.get())
            if self.panel is None:
                self.panel = tk.Label(self.container, image=image)
                self.panel.image = image
                self.panel.pack(side="top")
            else:
                self.panel.configure(image=image)
                self.panel.image = image
        self.after(5, self.__load_frame)

    def record_pressed(self):
        if self.__presenter.recording:
            self.record_button.config(text="Start Recording")
        else:
            self.record_button.config(text="Stop Recording")
        self.__presenter.switch_recording()

    def close_application(self):
        self.__presenter.release()
        self.destroy()

class AttentionPresenter:
    def __init__(self):
        self.img_queue = queue.Queue()
        self.recording = False
        self.__model = AttentionModel()
        self.__vc = cv2.VideoCapture(0)
        self.__worker_thread = threading.Thread(target=self.__update_frame)
        self.__worker_thread.start()

    def __update_frame(self):
        while not self.recording:
            start_time = time.time()
            _, frame = self.__vc.read()
            frame_draw = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(frame_draw)

            attention_state, box, color = self.__model.infer_attention(frame)
            if box is not None:
                draw.rectangle(box.tolist(), outline=color, width=6)
                draw.text((box[0], box[1] - 10), attention_state, fill=color)

            fps = 1.0 / (time.time() - start_time)
            draw.text((0, 0), str(int(fps)), fill=color)

            self.img_queue.put(frame_draw)
        return


    def __record_attention(self):
        with open(datetime.now().strftime("%d_%m_%Y %Hh%Mm%Ss") + ".csv", 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            while self.recording:
                start_time = time.time()
                _, frame = self.__vc.read()
                attention_state, _ = self.__model.infer_attention(frame)
                writer.writerow([datetime.now(), attention_state])
                elapsed_time = time.time() - start_time
                time.sleep(max(0, 0.1 - elapsed_time))
            return

    def switch_recording(self):
        self.recording = not self.recording
        self.__worker_thread.join()
        if self.recording:
            self.__worker_thread = threading.Thread(target=self.__record_attention)
        else:
            self.__worker_thread = threading.Thread(target=self.__update_frame)
        self.__worker_thread.start()

    def release(self):
        self.recording = not self.recording
        self.__worker_thread.join()
        self.__vc.release()

if __name__ == '__main__':
    AttentionView().mainloop()
