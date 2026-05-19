import os
import cv2
import yaml
import shutil
from datetime import datetime
from ultralytics import YOLO

keep_classes = {
    0: 'Caries', 1: 'Crown', 2: 'Filling', 3: 'Implant',
    6: 'Missing teeth', 7: 'Periapical lesion',
    9: 'Root Canal Treatment', 10: 'Root Piece',
    11: 'impacted tooth', 13: 'Bone Loss',
    30: 'Primary teeth'
}
old_to_new_id = {old_id: new_id for new_id, old_id in enumerate(keep_classes.keys())}

BASE_PATH = '/content/Dental_Veri/YOLO/YOLO'
TARGET_PATH = '/content/Dental_Veri/YOLO/Dental_Veri_V2_Hizli'

subfolders = ['train', 'valid', 'test']
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
global_counter = 1
atlanilan_resim_sayisi = 0

print("1. AŞAMA: Veri Ön İşleme Başlıyor (CLAHE & Kırpma)...")

for sub in subfolders:
    actual_sub = 'val' if sub == 'valid' and not os.path.exists(os.path.join(BASE_PATH, 'valid')) else sub
    img_dir = os.path.join(BASE_PATH, actual_sub, 'images')
    lbl_dir = os.path.join(BASE_PATH, actual_sub, 'labels')

    if not os.path.exists(img_dir): continue

    os.makedirs(os.path.join(TARGET_PATH, sub, 'images'), exist_ok=True)
    os.makedirs(os.path.join(TARGET_PATH, sub, 'labels'), exist_ok=True)

    images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    for img_name in images:
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

        img_path = os.path.join(img_dir, img_name)
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

print(f"Veri işleme tamam. {global_counter-1} resim hazır, {atlanilan_resim_sayisi} çöp resim elendi.")

print("\n2. AŞAMA: YAML Dosyası Oluşturuluyor...")
data_config = {
    'train': f'{TARGET_PATH}/train/images',
    'val': f'{TARGET_PATH}/valid/images',
    'test': f'{TARGET_PATH}/test/images',
    'nc': 11,
    'names': {i: name for i, name in enumerate(keep_classes.values())}
}

yaml_save_path = f'{TARGET_PATH}/dental_data_v2.yaml'
with open(yaml_save_path, 'w') as f:
    yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)

print("\n3. AŞAMA: YOLO11 Eğitimi Başlıyor...")
model = YOLO('yolo11n.pt') 

results = model.train(
    data=yaml_save_path,
    epochs=40,
    imgsz=640,
    batch=32,      
    device=0,
    workers=2,     
    cache=True,    
    project='Dis_Projesi_V11',
    name='v11n_deneme1'
)

print("\n4. AŞAMA: Sonuçlar Drive'a Yedekleniyor...")
zaman = datetime.now().strftime("%Y%m%d_%H%M")
yedek_yolu = f'/content/drive/MyDrive/Veri_Madenciligi/YOLO_{zaman}'
os.makedirs(yedek_yolu, exist_ok=True)

kaynak_klasor = '/content/Dis_Projesi_V11/v11n_deneme1'

if os.path.exists(kaynak_klasor):
    shutil.copytree(kaynak_klasor, yedek_yolu, dirs_exist_ok=True)
    print(f"✅ Başarılı! Tüm analizler şuraya yedeklendi:\n{yedek_yolu}")
else:
    print("❌ Hata: Kaynak klasör bulunamadı. Lütfen yolu kontrol et!")