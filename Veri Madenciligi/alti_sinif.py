import os
import sys
import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk, ImageDraw
from ultralytics import YOLO

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


class DentalAIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dental Röntgen Analiz Sistemi - V2 (6 Sınıf)")
        self.root.geometry("1200x800")
        self.root.configure(bg="#1e1e1e")

        self.model_path = resource_path('alti/best.pt')
        print(f"Model yükleniyor: {self.model_path}")

        self.model = YOLO(self.model_path)

        self.classes = {
            0: 'Crown (Kaplama)',
            1: 'Filling (Dolgu)',
            2: 'Implant',
            3: 'Root Canal (Kanal Td.)',
            4: 'Root Piece (Kök Parçası)',
            5: 'Impacted (Gömülü Diş)'
        }

        self.colors = ["#33ff33", "#3333ff", "#ffff33", "#33ffff", "#ff9933", "#9933ff"]

        self.image_path = None
        self.original_pil_img = None
        self.predictions = []
        self.active_classes = list(self.classes.keys())

        self.scale = 1.0
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.img_x, self.img_y = 0, 0

        self.setup_ui()

    def setup_ui(self):
        self.left_panel = tk.Frame(self.root, bg="#2d2d2d", width=250)
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        tk.Button(self.left_panel, text="📁 Resim Yükle", command=self.load_image, bg="#4CAF50", fg="white",
                  font=("Arial", 12, "bold")).pack(fill=tk.X, pady=5)
        self.run_btn = tk.Button(self.left_panel, text="⚙️ Analiz Et", command=self.run_analysis, bg="#2196F3",
                                 fg="white", font=("Arial", 12, "bold"), state=tk.DISABLED)
        self.run_btn.pack(fill=tk.X, pady=5)

        tk.Label(self.left_panel, text="--- FİLTRELER ---", bg="#2d2d2d", fg="white").pack(pady=10)

        tk.Button(self.left_panel, text="👁️ Hepsini Göster", command=self.show_all, bg="#555555", fg="white").pack(
            fill=tk.X, pady=2)
        tk.Button(self.left_panel, text="Kapat (Sıfırla)", command=self.hide_all, bg="#555555", fg="white").pack(
            fill=tk.X, pady=2)

        self.filter_buttons = {}
        for cls_id, cls_name in self.classes.items():
            color = self.colors[cls_id % len(self.colors)]
            btn = tk.Button(self.left_panel, text=cls_name, bg=color, fg="black", font=("Arial", 10),
                            command=lambda id=cls_id: self.toggle_class(id))
            btn.pack(fill=tk.X, pady=1)
            self.filter_buttons[cls_id] = btn

        self.canvas_frame = tk.Frame(self.root, bg="#1e1e1e")
        self.canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.canvas = tk.Canvas(self.canvas_frame, bg="#121212", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<MouseWheel>", self.zoom)
        self.canvas.bind("<ButtonPress-1>", self.start_pan)
        self.canvas.bind("<B1-Motion>", self.pan)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Resim Dosyaları", "*.jpg *.jpeg *.png")])
        if not file_path: return
        self.image_path = file_path
        self.original_pil_img = Image.open(file_path).convert("RGB")
        self.predictions = []
        self.run_btn.config(state=tk.NORMAL)
        self.update_canvas()

    def run_analysis(self):
        if not self.image_path: return
        self.run_btn.config(text="Analiz Ediliyor...", state=tk.DISABLED)
        self.root.update()
        results = self.model.predict(source=self.image_path, conf=0.25, verbose=False, device='cpu')
        self.predictions = results[0].boxes.data.cpu().numpy() if len(results[0].boxes) > 0 else []
        self.run_btn.config(text="⚙️ Analiz Et", state=tk.NORMAL)
        self.show_all()

    def toggle_class(self, cls_id):
        if cls_id in self.active_classes:
            self.active_classes.remove(cls_id)
        else:
            self.active_classes.append(cls_id)
        self.update_canvas()

    def show_all(self):
        self.active_classes = list(self.classes.keys())
        self.update_canvas()

    def hide_all(self):
        self.active_classes = []
        self.update_canvas()

    def update_canvas(self):
        if self.original_pil_img is None: return
        img_copy = self.original_pil_img.copy()
        draw = ImageDraw.Draw(img_copy)
        for pred in self.predictions:
            x1, y1, x2, y2, conf, cls = pred
            cls_id = int(cls)
            if cls_id in self.active_classes:
                color = self.colors[cls_id % len(self.colors)]
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                label = f"{self.classes[cls_id]} {conf:.2f}"
                draw.text((x1 + 2, y1 - 15), label, fill=color)

        new_w, new_h = int(img_copy.width * self.scale), int(img_copy.height * self.scale)
        if new_w > 0 and new_h > 0:
            resized = img_copy.resize((new_w, new_h), Image.Resampling.LANCZOS)
            self.tk_image = ImageTk.PhotoImage(resized)
            self.canvas.delete("all")
            self.canvas.create_image(self.img_x, self.img_y, anchor=tk.NW, image=self.tk_image)

    def zoom(self, event):
        self.scale *= 1.1 if event.delta > 0 else 0.9
        self.update_canvas()

    def start_pan(self, event):
        self.pan_start_x, self.pan_start_y = event.x, event.y

    def pan(self, event):
        self.img_x += event.x - self.pan_start_x
        self.img_y += event.y - self.pan_start_y
        self.pan_start_x, self.pan_start_y = event.x, event.y
        self.update_canvas()


if __name__ == "__main__":
    root = tk.Tk()
    app = DentalAIApp(root)
    root.mainloop()