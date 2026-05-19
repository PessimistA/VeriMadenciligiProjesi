import os
import cv2
import shutil

BASE_PATH = '/content/Dental_Veri/YOLO/YOLO' 
TARGET_PATH = '/content/Dental_Veri/YOLO/Dental_Veri_Temiz' 

subfolders = ['train', 'valid', 'test']
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
global_counter = 1

print("İşlem başlıyor... SSD hızında işleme yapıyoruz!\n")

for sub in subfolders:
    actual_sub = sub
    if sub == 'valid' and not os.path.exists(os.path.join(BASE_PATH, 'valid')):
        if os.path.exists(os.path.join(BASE_PATH, 'val')):
            actual_sub = 'val'

    img_dir = os.path.join(BASE_PATH, actual_sub, 'images')
    lbl_dir = os.path.join(BASE_PATH, actual_sub, 'labels')

    if not os.path.exists(img_dir):
        print(f"Uyarı: {img_dir} bulunamadı, atlanıyor...")
        continue

    os.makedirs(os.path.join(TARGET_PATH, sub, 'images'), exist_ok=True)
    os.makedirs(os.path.join(TARGET_PATH, sub, 'labels'), exist_ok=True)

    images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    print(f"--- {sub.upper()} işleniyor ({len(images)} resim) ---")

    for img_name in images:
        img_path = os.path.join(img_dir, img_name)
        lbl_path = os.path.join(lbl_dir, os.path.splitext(img_name)[0] + '.txt')

        if not os.path.exists(lbl_path): continue

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None: continue

        enhanced = clahe.apply(img)

        h, w = enhanced.shape
        enhanced[0:int(h*0.15), :] = 0
        enhanced[int(h*0.85):h, :] = 0

        new_name = f"{global_counter:06d}"

        cv2.imwrite(os.path.join(TARGET_PATH, sub, 'images', new_name + '.jpg'), enhanced)
        shutil.copy(lbl_path, os.path.join(TARGET_PATH, sub, 'labels', new_name + '.txt'))

        global_counter += 1

    print(f"{sub} tamamlandı.")

print(f"\nBÜTÜN İŞLEMLER BİTTİ!")
print(f"Toplam {global_counter-1} resim temizlendi, maskelendi ve anonimleştirildi.")
print(f"Yeni veri setin burada: {TARGET_PATH}")


import yaml

base_path = '/content/Dental_Veri/YOLO/Dental_Veri_Temiz'

data_config = {
    'train': f'{base_path}/train/images',
    'val': f'{base_path}/valid/images',
    'test': f'{base_path}/test/images',
    'nc': 31, 
    'names': {
        0: 'Caries', 1: 'Crown', 2: 'Filling', 3: 'Implant', 4: 'Malaligned',
        5: 'Mandibular Canal', 6: 'Missing teeth', 7: 'Periapical lesion',
        8: 'Retained root', 9: 'Root Canal Treatment', 10: 'Root Piece',
        11: 'impacted tooth', 12: 'maxillary sinus', 13: 'Bone Loss',
        14: 'Fracture teeth', 15: 'Permanent Teeth', 16: 'Supra Eruption',
        17: 'TAD', 18: 'abutment', 19: 'attrition', 20: 'bone defect',
        21: 'gingival former', 22: 'metal band', 23: 'orthodontic brackets',
        24: 'permanent retainer', 25: 'post - core', 26: 'plating',
        27: 'wire', 28: 'Cyst', 29: 'Root resorption', 30: 'Primary teeth'
    }
}

yaml_save_path = '/content/Dental_Veri/YOLO/Dental_Veri_Temiz/dental_data.yaml'

with open(yaml_save_path, 'w') as f:
    yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)

print(f"Yeni dental_data.yaml başarıyla oluşturuldu!\nKonum: {yaml_save_path}")
from ultralytics import YOLO

model = YOLO('yolov8s.pt')


results = model.train(
    data='/content/Dental_Veri/YOLO/Dental_Veri_Temiz/dental_data.yaml',
    epochs=5,
    imgsz=640,
    batch=16,
    device=0,  
    workers=4,  
    project='Dis_Projesi',
    name='v8s_deneme'
)