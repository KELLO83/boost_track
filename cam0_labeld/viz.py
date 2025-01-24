import cv2
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
# GUI로 영상 재생
frame_width, frame_height = 1920, 1080  
output_video = "output.avi"
fps = 15
class VideoPlayer:
    def __init__(self, root, video_path):
        self.root = root
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.canvas = tk.Canvas(root, width=frame_width, height=frame_height)
        self.canvas.pack()

        self.slider = ttk.Scale(root, from_=0, to=self.total_frames - 1, orient="horizontal", command=self.on_slider)
        self.slider.pack(fill="x")

        self.playing = False
        self.current_frame = 0

        self.play_button = ttk.Button(root, text="Play", command=self.toggle_play)
        self.play_button.pack(side="left")

        self.pause_button = ttk.Button(root, text="Pause", command=self.pause)
        self.pause_button.pack(side="left")

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.update_frame()

    def on_slider(self, val):
        self.current_frame = int(float(val))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        self.update_frame()
        self.update_title()

    def toggle_play(self):
        self.playing = not self.playing
        if self.playing:
            self.play_button.config(text="Pause")
            self.play_video()
        else:
            self.play_button.config(text="Play")

    def pause(self):
        self.playing = False
        self.play_button.config(text="Play")

    def play_video(self):
        if self.playing and self.cap.isOpened():
            self.current_frame += 1
            if self.current_frame >= self.total_frames:
                self.current_frame = 0
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.slider.set(self.current_frame)
            self.update_frame()
            self.root.after(int(1000 / fps), self.play_video)

    def update_frame(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (frame_width, frame_height))
                self.photo = tk.PhotoImage(data=cv2.imencode('.png', frame)[1].tobytes())
                self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
                self.update_title()

    def update_title(self):
        self.root.title(f"Video Player - Frame {self.current_frame + 1} / {self.total_frames}")

    def on_close(self):
        self.cap.release()
        self.root.destroy()

# GUI 실행
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Video Player")
    player = VideoPlayer(root, output_video)
    root.mainloop()