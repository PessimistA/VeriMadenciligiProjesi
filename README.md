# Dental Röntgen Analiz Sistemi — Veri Madenciliği Projesi

Panoramik diş röntgenlerindeki bulguları **YOLO11** nesne tespiti modelleriyle otomatik
olarak işaretleyen bir masaüstü uygulaması ve onu besleyen veri hazırlama / model
eğitim hattı.

Kullanıcı bir röntgen görüntüsü yükler, **Analiz Et** der; uygulama tespit edilen
çürükleri, dolguları, implantları, gömülü dişleri ve diğer bulguları renkli kutularla
güven skoruyla birlikte gösterir. Her sınıf tek tıkla açılıp kapatılabilir, görüntü
fare tekerleğiyle yakınlaştırılıp sürüklenebilir.

---

## İçindekiler

- [Projenin amacı](#projenin-amacı)
- [Üç sürüm, üç sınıf kümesi](#üç-sürüm-üç-sınıf-kümesi)
- [Arayüz](#arayüz)
- [Veri hazırlama ve model eğitim hattı](#veri-hazırlama-ve-model-eğitim-hattı)
- [Deney günlüğü](#deney-günlüğü)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Model ağırlıkları](#model-ağırlıkları)
- [Klasör yapısı](#klasör-yapısı)
- [Sınırlar ve tıbbi uyarı](#sınırlar-ve-tıbbi-uyarı)
- [Lisans](#lisans)

---

## Projenin amacı

Ham dental veri kümesi 31 farklı etikete sahiptir; ancak bu sınıfların büyük kısmı
veri kümesinde çok az örnekle temsil edilir. Bu durum klasik bir veri madenciliği
problemidir: **çok sınıflı ama dengesiz bir veri kümesinde, sınıf sayısını azaltmanın
model başarımına etkisi nedir?**

Proje bu soruyu üç ayrı model eğiterek yanıtlar: 31 sınıf, 11 sınıf ve 6 sınıf.
Her sürüm için ayrı bir arayüz uygulaması bulunur, böylece sonuçlar aynı görüntü
üzerinde doğrudan karşılaştırılabilir.

---

## Üç sürüm, üç sınıf kümesi

| Dosya | Sınıf sayısı | Kapsanan bulgular |
|---|---|---|
| **`alti_sinif.py`** | 6 | Crown (Kaplama), Filling (Dolgu), Implant, Root Canal (Kanal Tedavisi), Root Piece (Kök Parçası), Impacted (Gömülü Diş) |
| **`on_bir_sinif.py`** | 11 | Yukarıdakilere ek olarak Caries (Çürük), Missing teeth (Eksik Diş), Periapical lesion (Lezyon), Bone Loss (Kemik Kaybı), Primary teeth (Süt Diş) |
| **`otuz_bir.py`** | 31 | Ham veri kümesinin tamamı — Malaligned, Mandibular Canal, Fracture teeth, Cyst, Root resorption, orthodontic brackets, TAD, maxillary sinus ve diğerleri |

**Neden azaltıldı?** Nadir sınıflar hem eğitimi zorlaştırır hem de yanlış pozitif
üretir. 11 sınıflı sürüm klinik olarak en anlamlı bulguları korurken veri dengesini
iyileştirir; 6 sınıflı sürüm ise yalnızca görsel olarak net ayırt edilebilen
restoratif bulgulara odaklanır.

`on_bir_sinif.py` ve `otuz_bir.py` sürümlerinde arayüzde bir **model seçici** vardır:
klasördeki tüm `*/best.pt` dosyaları otomatik bulunur, açılır listeden seçilen model
uygulamayı kapatmadan yeniden yüklenir.

---

## Arayüz

Tkinter ile yazılmış koyu temalı bir masaüstü penceresi:

| Bileşen | İşlevi |
|---|---|
| 📁 **Resim Yükle** | `.jpg`, `.jpeg`, `.png` röntgen görüntüsü seçer |
| ⚙️ **Analiz Et** | Seçili YOLO modelini `conf=0.25` eşiğiyle çalıştırır |
| 👁️ **Hepsini Göster / Kapat** | Tüm sınıf kutularını aynı anda açar veya gizler |
| **Sınıf filtreleri** | Her sınıf için ayrı bir renkli düğme; tıklayınca o sınıfın kutuları gizlenir/gösterilir |
| **Tuval** | Fare tekerleği ile yakınlaştırma (`×1.1` / `×0.9`), sol tuşla sürükleyerek kaydırma |

Tespit sonuçları `Pillow` ile görüntünün üzerine çizilir; her kutunun yanında
`sınıf adı + güven skoru` etiketi bulunur. Sınıflar sabit bir renk paletiyle ayrılır.

`resource_path()` yardımcı fonksiyonu sayesinde uygulama PyInstaller ile tek dosyalık
`.exe` olarak paketlendiğinde de model dosyasını doğru konumda bulur.

---

## Veri hazırlama ve model eğitim hattı

`Veri Madenciligi/testler/` klasöründeki altı betik, projenin **deneysel gelişim
sürecini** sırayla belgeler. Betikler Google Colab üzerinde çalışacak biçimde
yazılmıştır (`/content/...` yolları, Drive'a otomatik yedekleme).

Ortak hat şu adımlardan oluşur:

### 1. Görüntü ön işleme — CLAHE

```python
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
```

Röntgen görüntülerinde kontrast dağılımı çok dengesizdir. **CLAHE** (Contrast Limited
Adaptive Histogram Equalization) görüntüyü küçük bölgelere ayırıp her birinin
kontrastını ayrı ayrı iyileştirir; böylece karanlık bölgelerdeki çürük ve lezyon
sınırları belirginleşir.

### 2. Sınıf filtreleme ve yeniden numaralama

```python
keep_classes = {0: 'Caries', 1: 'Crown', 2: 'Filling', 3: 'Implant',
                6: 'Missing teeth', 7: 'Periapical lesion', ...}
old_to_new_id = {old: new for new, old in enumerate(keep_classes.keys())}
```

Tutulacak sınıflar seçilir, YOLO etiket dosyalarındaki eski sınıf numaraları
ardışık yeni numaralara çevrilir. Filtreleme sonrası **hiç etiketi kalmayan
görüntüler ("çöp resim") elenir** — bunlar modele yalnızca gürültü katar.

### 3. `data.yaml` üretimi

`train` / `valid` / `test` yolları, sınıf sayısı (`nc`) ve sınıf adları programatik
olarak yazılır; böylece sınıf kümesi değiştiğinde yapılandırma elle güncellenmez.

### 4. Eğitim ve yedekleme

Ultralytics `model.train()` çağrılır, sonuçlar eğitim sırasında doğrudan Google
Drive'a yazılır — Colab oturumu koptuğunda çalışma kaybolmaz.

---

## Deney günlüğü

Betikler kronolojik bir denemeler zinciridir:

| Betik | Ne denendi |
|---|---|
| **`test_1.py`** | Yalnızca veri temizliği: tüm alt klasörler dolaşılır, CLAHE uygulanır, dosyalar tek bir sayaçla yeniden adlandırılır |
| **`test_2.py`** | Aynı temizlik + 11 sınıfa filtreleme; elenen görüntü sayısı raporlanır |
| **`test_3.py`** | İlk uçtan uca eğitim — `yolo11n` (nano), 40 epoch, `imgsz=640`, `batch=32` |
| **`test_4.py`** | Yüksek çözünürlük denemesi — `yolo11m` (medium), `imgsz=1024`, `batch=24`, `patience=10` |
| **`test_5.py`** | En kapsamlı ayar — `yolo11m`, 60 epoch, `AdamW`, kosinüs öğrenme oranı, 5 epoch ısınma ve agresif veri artırma: `mosaic=1.0`, `mixup=0.2`, `copy_paste=0.3`, `degrees=10`, `fliplr=0.5`, `hsv_v=0.4` |
| **`test_6.py`** | 6 sınıfa indirgenmiş hızlı doğrulama koşusu — `yolo11m`, 25 epoch, `batch=16` |

`test_5.py` deneylerin en olgun halidir: `cache='ram'` ile veri RAM'de tutulur,
`copy_paste` ve `mixup` az örnekli sınıfların temsilini artırır, `cos_lr` ile öğrenme
oranı yumuşak biçimde düşürülür.

---

## Kurulum

### Gereksinimler

- Python 3.9+
- Tkinter (çoğu Python dağıtımıyla gelir; Arch/Debian'da ayrı paket olabilir)
- GPU zorunlu değildir — arayüz `device='cpu'` ile çıkarım yapar

```bash
pip install ultralytics opencv-python pillow pyyaml
```

`ultralytics` paketi PyTorch'u da beraberinde kurar.

Linux'ta Tkinter eksikse:

```bash
sudo pacman -S tk          # Arch / CachyOS
sudo apt install python3-tk  # Debian / Ubuntu
```

---

## Kullanım

```bash
cd "Veri Madenciligi"

python alti_sinif.py      # 6 sınıflı sürüm
python on_bir_sinif.py    # 11 sınıflı sürüm  (model seçici ile)
python otuz_bir.py        # 31 sınıflı sürüm  (model seçici ile)
```

1. **Resim Yükle** ile bir panoramik röntgen seçin
2. **Analiz Et** ile çıkarımı başlatın
3. Sol paneldeki sınıf düğmeleriyle görmek istediğiniz bulguları filtreleyin
4. Fare tekerleği ile yakınlaşıp sol tuşla sürükleyerek ilgilendiğiniz bölgeyi inceleyin

---

## Model ağırlıkları

Eğitilmiş `.pt` ağırlık dosyaları boyutları nedeniyle sürüm kontrolü dışında
tutulur; her uygulama kendi model klasörünü çalışma zamanında okur.

Ağırlıkları üretmek için `testler/` altındaki eğitim betiklerini (özellikle
`test_5.py`) kendi veri kümenizin yollarına göre uyarlayıp çalıştırın. Eğitim
sonunda Ultralytics'in ürettiği `best.pt` dosyasını ilgili uygulamanın model
klasörüne kopyalamanız yeterlidir.

`on_bir_sinif.py` ve `otuz_bir.py` çalıştıkları dizindeki model klasörlerini
kendiliğinden tarar; yeni bir klasör eklediğinizde açılır listede belirir ve
uygulamayı kapatmadan seçilebilir.

---

## Klasör yapısı

```
VeriMadenciligiProjesi/
└── Veri Madenciligi/
    ├── alti_sinif.py       # 6 sınıflı arayüz
    ├── on_bir_sinif.py     # 11 sınıflı arayüz + model seçici
    ├── otuz_bir.py         # 31 sınıflı arayüz + model seçici
    └── testler/
        ├── test_1.py       # veri temizliği + CLAHE
        ├── test_2.py       # 11 sınıfa filtreleme
        ├── test_3.py       # yolo11n, 40 epoch, 640px
        ├── test_4.py       # yolo11m, yüksek çözünürlük (1024px)
        ├── test_5.py       # yolo11m, AdamW + agresif augmentasyon
        └── test_6.py       # 6 sınıf, hızlı doğrulama koşusu
```

Model ağırlıkları çalışma zamanında `alti/` ve `on_bir/` klasörlerinden okunur.

---

## Sınırlar ve tıbbi uyarı

Bu proje bir **veri madenciliği dersi çalışmasıdır**; tanı koyan bir tıbbi cihaz
değildir.

- Model tahminleri istatistikseldir; yanlış pozitif ve yanlış negatif üretebilir
- Eğitim verisi sınırlı sayıda röntgenden oluşur, farklı cihaz ve çekim koşullarına
  genelleme garantisi yoktur
- Çıktılar **hiçbir koşulda** klinik karar için kullanılmamalıdır

Diş sağlığıyla ilgili her konuda bir hekime danışın.

---

## Lisans

Akademik ve eğitim amaçlı geliştirilmiştir.
