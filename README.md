# pytorch_segmentation

## make sample data : generate_sample_data_class1.py
class 0 : background
class 1 : circle

## make sample data : generate_sample_data_class4.py
class 0 : background
class 1 : circle
class 2 : rectangle
class 3 : triangle

## download DRIVE dataset : get_DRIVE_dataset.py
```bash
pytorch_segment/
│
├── drive_dataset/ # 데이터 디렉토리
│   ├── train/ # 훈련 데이터 디렉토리
│   │   ├── image
│   │   └── mask
│   └── test/ # 테스트 데이터 디렉토리
│       ├── image
│       └── mask
```