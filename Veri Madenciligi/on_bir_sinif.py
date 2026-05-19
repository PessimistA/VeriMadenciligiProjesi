import tkinter as tk
from tkinter import ttk, filedialog
import cv2
import glob
import os
import sys
from PIL import Image, ImageTk, ImageDraw
from ultralytics import YOLO

if getattr(sys, 'frozen', False):
    os.chdir(os.path.dirname(sys.executable))
else:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
class DentalAIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dental Röntgen Analiz Sistemi")
        self.root.geometry("1200x800")
        self.root.configure(bg="#1e1e1e")

        self.available_models = [path.replace("\\", "/") for path in glob.glob("*/*.pt")]
        self.default_model = 'on_bir/best.pt'


        if self.default_model not in self.available_models:
            self.available_models.insert(0, self.default_model)

        self.current_model_path = self.default_model

        print(f"Varsayılan model yükleniyor: {self.current_model_path}")
        self.model = YOLO(self.current_model_path)

        self.classes = {
            0: 'Caries (Çürük)', 1: 'Crown (Kaplama)', 2: 'Filling (Dolgu)',
            3: 'Implant', 4: 'Missing teeth (Eksik Diş)', 5: 'Periapical lesion (Lezyon)',
            6: 'Root Canal (Kanal Td.)', 7: 'Root Piece (Kök Parçası)',
            8: 'Impacted (Gömülü)', 9: 'Bone Loss (Kemik Kaybı)', 10: 'Primary teeth (Süt Diş)'
        }


        self.colors = ["#ff3333", "#33ff33", "#3333ff", "#ffff33", "#ff33ff",
                       "#33ffff", "#ff9933", "#9933ff", "#33ff99", "#ff3399", "#99ff33"]

        self.image_path = None
        self.original_pil_img = None
        self.current_pil_img = None
        self.predictions = []
        self.active_classes = list(self.classes.keys())

        self.scale = 1.0
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.img_x = 0
        self.img_y = 0

        self.setup_ui()

    def setup_ui(self):

        self.left_panel = tk.Frame(self.root, bg="#2d2d2d", width=250)
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        tk.Label(self.left_panel, text="🧠 Model Seçimi:", bg="#2d2d2d", fg="white", font=("Arial", 10, "bold")).pack(
            fill=tk.X, pady=(0, 2))
        self.model_combo = ttk.Combobox(self.left_panel, values=self.available_models, state="readonly")
        self.model_combo.set(self.current_model_path)
        self.model_combo.pack(fill=tk.X, pady=(0, 10))
        self.model_combo.bind("<<ComboboxSelected>>", self.change_model)

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
        self.canvas.bind("<Button-4>", self.zoom)
        self.canvas.bind("<Button-5>", self.zoom)
        self.canvas.bind("<ButtonPress-1>", self.start_pan)
        self.canvas.bind("<B1-Motion>", self.pan)

    def change_model(self, event):
        selected_model = self.model_combo.get()
        if selected_model == self.current_model_path:
            return

        print(f"Model değiştiriliyor: {selected_model}")
        self.run_btn.config(text="Model Yükleniyor...", state=tk.DISABLED)
        self.root.update()

        try:
            self.model = YOLO(selected_model)
            self.current_model_path = selected_model
            print("Model başarıyla yüklendi.")
        except Exception as e:
            print(f"Model yüklenirken hata oluştu: {e}")
            self.model_combo.set(self.current_model_path)

        state = tk.NORMAL if self.image_path else tk.DISABLED
        self.run_btn.config(text="⚙️ Analiz Et", state=state)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Resim Dosyaları", "*.jpg *.jpeg *.png")])
        if not file_path: return

        self.image_path = file_path
        self.original_pil_img = Image.open(file_path).convert("RGB")
        self.predictions = []
        self.run_btn.config(state=tk.NORMAL)

        c_width = self.canvas.winfo_width()
        c_height = self.canvas.winfo_height()
        i_width, i_height = self.original_pil_img.size
        self.scale = min(c_width / i_width, c_height / i_height) * 0.9
        self.img_x, self.img_y = 20, 20

        self.update_canvas()

    def run_analysis(self):
        if not self.image_path: return
        self.run_btn.config(text="Analiz Ediliyor...", state=tk.DISABLED)
        self.root.update()

        results = self.model.predict(source=self.image_path, conf=0.25, verbose=False, device='cpu')

        if len(results[0].boxes) > 0:
            self.predictions = results[0].boxes.data.cpu().numpy()
        else:
            self.predictions = []

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
                draw.rectangle([x1, y1 - 15, x1 + len(label) * 6, y1], fill=color)
                draw.text((x1 + 2, y1 - 15), label, fill="black")

        new_width = int(img_copy.width * self.scale)
        new_height = int(img_copy.height * self.scale)
        if new_width > 0 and new_height > 0:
            resized_img = img_copy.resize((new_width, new_height), Image.Resampling.LANCZOS)

            self.tk_image = ImageTk.PhotoImage(resized_img)
            self.canvas.delete("all")
            self.canvas.create_image(self.img_x, self.img_y, anchor=tk.NW, image=self.tk_image)

    def zoom(self, event):
        if event.num == 4 or event.delta > 0:
            self.scale *= 1.1
        elif event.num == 5 or event.delta < 0:
            self.scale *= 0.9
        self.update_canvas()

    def start_pan(self, event):
        self.pan_start_x = event.x
        self.pan_start_y = event.y

    def pan(self, event):
        dx = event.x - self.pan_start_x
        dy = event.y - self.pan_start_y
        self.img_x += dx
        self.img_y += dy
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        self.update_canvas()


if __name__ == "__main__":
    root = tk.Tk()
    app = DentalAIApp(root)
    root.mainloop()