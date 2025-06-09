import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import threading
import torch
import warnings
import pathlib

warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda.amp")


class AnimalDetect:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv5 Animal detection")
        self.root.geometry("900x600")

        try:
            temp = pathlib.PosixPath
            pathlib.PosixPath = pathlib.WindowsPath

            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='qulinda-project-v1.pt')
            self.model.conf = 0.6
        except Exception as e :
            print(e)
            exit()

        self.streams = {
            "Savanna": "https://zssd-kijami.hls.camzonecdn.com/CamzoneStreams/zssd-kijami/chunklist.m3u8",
            "Elephants": "https://elephants.hls.camzonecdn.com/CamzoneStreams/elephants/Playlist.m3u8",
            "Giraffe": "https://cha-fi1-prd-vid-str-002.epbfi.com/live/giraffe2.stream/playlist.m3u8",
        }

        self.current_cap = None
        self.running = False
        self.thread = None

        self.create_gui()

        self.start_stream(list(self.streams.keys())[0])

    def create_gui(self):
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)

        for name in self.streams.keys():
            btn = tk.Button(btn_frame, text=name, width=15, height=2,
                            command=lambda n=name: self.start_stream(n))
            btn.pack(side=tk.LEFT, padx=5)

        stop_btn = tk.Button(btn_frame, text="STOP", width=15, height=2,
                             bg='red', fg='white', command=self.stop_stream)
        stop_btn.pack(side=tk.LEFT, padx=5)

        img_btn = tk.Button(btn_frame, text="Your own", width=15, height=2,
                            command=self.test_image)
        img_btn.pack(side=tk.LEFT, padx=5)

        self.display_label = tk.Label(self.root, bg='black', fg='white')
        self.display_label.pack(expand=True, fill='both')

    def start_stream(self, stream_name):
        self.running = False
        if self.current_cap:
            self.current_cap.release()
            self.current_cap = None
        url = self.streams[stream_name]

        self.display_label.config(text=f"Changing to {stream_name}...", image="")
        self.running = True

        self.thread = threading.Thread(target=self.process_stream, args=(url,))
        self.thread.daemon = True
        self.thread.start()

    def process_stream(self, url):
        try:
            self.current_cap = cv2.VideoCapture(url)

            self.current_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not self.current_cap.isOpened():
                self.display_label.config(text="Failed to open stream")
                return

            while self.running:
                ret, frame = self.current_cap.read()
                if not ret:
                    continue
                self.show_frame(frame)

            if self.current_cap:
                self.current_cap.release()
                self.current_cap = None

        except Exception as e:
            self.display_label.config(text=f"Stream error: {str(e)[:50]}")

    def test_image(self):
        if not self.model:
            self.display_label.config(text="YOLO model not loaded!")
            return

        file_path = filedialog.askopenfilename(
            title="Choose a photo or a video",
            filetypes=[("File types", "*.jpg *.jpeg *.png *.mp4 *.avi *.mov")]
        )

        if not file_path:
            return

        try:
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                image = cv2.imread(file_path)
                if image is None:
                    self.display_label.config(text="Failed to load image")
                    return

                self.show_frame(image)

            elif file_path.lower().endswith(('.mp4', '.avi', '.mov')):
                self.thread = threading.Thread(target=self.process_video_file, args=(file_path,))
                self.thread.daemon = True
                self.thread.start()

            else:
                self.display_label.config(text="Unsupported file format")

        except Exception as e:
            self.display_label.config(text=f"Processing error: {str(e)[:50]}")

    def process_video_file(self, file_path):
        cap = cv2.VideoCapture(file_path)

        if not cap.isOpened():
            self.display_label.config(text="Could not open the video file")
            return

        self.running = True

        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            try:
                self.show_frame(frame)

            except Exception as e:
                print(e)
                continue

        cap.release()

    def show_frame(self, frame):
        try:
            height, width = frame.shape[:2]
            max_widht = 900
            max_height = 540

            scale = min(max_widht/width, max_height/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results = self.model(frame)
                frame = results.render()[0]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(image=pil_image)

            self.display_label.config(image=photo, text="")
            self.display_label.image = photo

        except:
            pass

    def stop_stream(self):
        self.running = False
        self.display_label.config(text="Stopped", image="")


def main():
    root = tk.Tk()
    app = AnimalDetect(root)
    root.mainloop()


if __name__ == "__main__":
    main()