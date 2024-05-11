# Automatic Number Plate Recognition

Repositori ini berisi implementasi model object detection YOLO v7 Tiny dan Easy OCR untuk sistem deteksi plat nomor kendaraan otomatis.

## Technical Aspect
![Flowchart](https://github.com/daffatm/plate-detection/blob/main/asset/Scheme.jpg)

Untuk penerapannya pada *Real-world scenario* maka semua model yang digunakan adalah model dengan *low power consumption* atau *lightweight model* sehingga diharapkan sistem ini dapat berjalan pada *edge devices*.

- **Plate Detection** model di training dengan model pretrain [YOLO v7 Tiny](https://github.com/WongKinYiu/yolov7)  pada dataset pribadi. Alasan digunakannya YOLO v7 Tiny dibandingkan dengan YOLO terbaru seperti v8 atau v9 sebenarnya adalah refrensi pribadi, karna YOLO v7 memiliki repositori terpisah dari Ultralytics sehingga untuk modifikasi kode lebih mudah karna memiliki standalone code serta YOLO v7 juga masih memiliki accuracy dan speed trade off yang baik jika dibandingkan dengan YOLO terbaru.

- **Optical Character Recognition (OCR)** model yang digunakan adalah [Easy OCR](https://github.com/JaidedAI/EasyOCR) tanpa melakukan retrain karna tidak adanya data untuk training serta model yang dimiliki Easy OCR sudah cukup baik. Pemilihan model OCR didasarkan pada artikel https://blog.roboflow.com/best-ocr-models-text-recognition/ dimana didapatkan Easy OCR memiliki accuracy dan speed yang cukup baik pada deteksi plat nomor serta low cost.

- **Super Resolution** model juga digunakan pada projek ini untuk melakukan enhance pada gambar plat nomor setelah di crop agar memiliki hasil yang lebih baik untuk OCR. Untuk super resolution model digunakan pretrain dari repositori [Real TIme Super Resolution](https://github.com/braindotai/Real-Time-Super-Resolution) dengan alasan model ini ringan dan dapat berjalan dengan baik pada CPU.

Untuk detail mengenai training dan lainnya bisa di lihat pada [notebook.](https://github.com/daffatm/plate-detection/tree/main/notebook)

## Instalation

Projek ini menggunakan **python 3.8** jadi pastikan sudah menginstallnya dan ikuti langkah berikut.

**Clone Project**
```
git clone https://github.com/daffatm/plate-detection.git
```
**Go to Project Dir**
```
cd plate-detection
```
**Install Python Virtual Environtment**
```
pip install virtualenv
```
**Create Virtual Environtment**
```
python -m venv venv
```
**Activate Virtual Environtment**
```
source venv/Scripts/activate
```
**Install Requirements**
```
pip install -r requirements.txt
```

## Usage

```
use webcam: 
	python main.py --source "webcam" --threshold 0.55 --cuda

image inference: 
	python main.py --source "./img/Cars1.png" --threshold 0.55

## Automatic Number Plate Recognition ##
optional arguments:
	--source		webcam or image_path.
	--model			plate detection model path (.onnx)
	--threshold		plate detection model threshold
	--cuda			use CUDA/GPU
```

## Performance
**Plate Detection (YOLO v7 Tiny)**
Model pytorch telah dikonversi menjadi ONNX (Open Neural Network Exchange) agar dpat berjalan lebih baik pada CPU menggunakan onnxruntime dan mudah untuk dikonversi ke framework deep learning lain.

|  Model| Image Size | Precision | Recall | mAP@.5 | mAP@.5:.95 | Model Size |
|--|--|--|--|--|--|--|
|[best.onnx](https://github.com/daffatm/plate-detection/blob/main/model/best.onnx) | 640 | 0.891 | 0.891 | 0.931 | 0.555 | 23.5 MB |

*Note: tested on private dataset*

## Result
Inference Result:
![Inference Result](https://github.com/daffatm/plate-detection/blob/main/asset/Result.png)

Webcam Result:
![Realtime Result](https://github.com/daffatm/plate-detection/blob/main/asset/Realtime%20Result.png)
*Note: run on very low power cpu AMD A9-9420 Dual Core*
