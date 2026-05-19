
import os
import cv2

keep_classes = {
    0: 'Caries', 1: 'Crown', 2: 'Filling', 3: 'Implant',
    6: 'Missing teeth', 7: 'Periapical lesion',
    9: 'Root Canal Treatment', 10: 'Root Piece',
    11: 'impacted tooth', 13: 'Bone Loss',
    30: 'Primary teeth'
}
old_to_new_id = {old_id: new_id for new_id, old_id in enumerate(keep_classes.keys())}

BASE_PATH = '/content/Dental_Veri/YOLO/YOLO'
TARGET_PATH = '/content/Dental_Veri/YOLO/Dental_Veri_Temiz_Filtreli' 

subfolders = ['train', 'valid', 'test']
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
global_counter = 1
atlanilan_resim_sayisi = 0 

print("İşlem başlıyor... Gereksiz sınıflar filtreleniyor ve SSD hızında işleniyor!\n")

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

        with open(lbl_path, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts: continue
            old_id = int(parts[0])

            if old_id in old_to_new_id:
                parts[0] = str(old_to_new_id[old_id]) 
                new_lines.append(' '.join(parts) + '\n')

        if not new_lines:
            atlanilan_resim_sayisi += 1
            continue

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None: continue

        enhanced = clahe.apply(img)
        h, w = enhanced.shape
        enhanced[0:int(h*0.15), :] = 0
        enhanced[int(h*0.85):h, :] = 0

        new_name = f"{global_counter:06d}"

        cv2.imwrite(os.path.join(TARGET_PATH, sub, 'images', new_name + '.jpg'), enhanced)

        with open(os.path.join(TARGET_PATH, sub, 'labels', new_name + '.txt'), 'w') as f:
            f.writelines(new_lines)

        global_counter += 1

    print(f"{sub} tamamlandı.")

print(f"\nBÜTÜN İŞLEMLER BİTTİ!")
print(f"Toplam {global_counter-1} resim temizlendi, filtrelendi ve anonimleştirildi.")
print(f"İçinde hedef sınıflardan hiçbiri olmayan {atlanilan_resim_sayisi} 'çöp' resim veri setinden elendi.")
print(f"Yeni, odaklanmış veri setin burada: {TARGET_PATH}")

import yaml

base_path = '/content/Dental_Veri/YOLO/Dental_Veri_Temiz_Filtreli'

data_config = {
    'train': f'{base_path}/train/images',
    'val': f'{base_path}/valid/images',
    'test': f'{base_path}/test/images',
    'nc': 11,  
    'names': {
        0: 'Caries',
        1: 'Crown',
        2: 'Filling',
        3: 'Implant',
        4: 'Missing teeth',
        5: 'Periapical lesion',
        6: 'Root Canal Treatment',
        7: 'Root Piece',
        8: 'impacted tooth',
        9: 'Bone Loss',
        10: 'Primary teeth'
    }
}

yaml_save_path = f'{base_path}/dental_data_filtered.yaml'

with open(yaml_save_path, 'w') as f:
    yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)

print(f"Yeni dental_data_filtered.yaml başarıyla oluşturuldu!\nKonum: {yaml_save_path}")
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

results = model.train(
    data='/content/Dental_Veri/YOLO/Dental_Veri_Temiz_Filtreli/dental_data_filtered.yaml',
    epochs=40,
    imgsz=640,
    batch=16,
    device=0,  
    workers=2,  
    project='Dis_Projesi4',
    name='v8n_deneme5'
)