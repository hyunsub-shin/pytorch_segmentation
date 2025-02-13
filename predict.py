import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from models.unet import *  # 모델 파일에서 U-Net 클래스 import
import platform  # 폰트관련 운영체제 확인

# 마스크를 컬러 이미지로 변환
def convert_to_color_mask(mask, color_map):
    """
    마스크를 클래스별 컬러로 표현합니다.

    Args:
        mask (np.ndarray): 예측된 마스크 (클래스 번호)
        color_map (dict): 클래스 번호와 RGB 컬러 값을 매핑하는 딕셔너리

    Returns:
        np.ndarray: 컬러 이미지 (RGB)
    """
    print(f'mask.shape = {mask.shape}')

    # 마스크의 각 픽셀에 대해 색상 매핑(벡터화 처리)
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_id, color in color_map.items():
        # 클래스 ID에 해당하는 마스크 영역에 색을 채움
        color_mask[mask == class_id] = color

    return color_mask

# 결과 시각화
def visualize_result(image, predicted_mask, num_classes):
    """
    이미지와 예측된 마스크를 시각화합니다.
    Args:
        image (PIL.Image): 원본 이미지
        predicted_mask (torch.Tensor): 예측된 마스크
    """
    # 클래스별 컬러 맵 정의
    COLOR_MAP, CLASS_NAMES = get_color_map(num_classes)
    
    image = np.array(image) # PIL 이미지를 NumPy 배열로 변환
    if num_classes == 1:
        mask = predicted_mask.cpu().numpy()  # 마스크를 NumPy 배열로 변환 (GPU -> CPU)
    else:
        mask = predicted_mask.cpu().numpy()  # GPU -> CPU, NumPy 배열로 변환
        color_mask = convert_to_color_mask(mask, COLOR_MAP)   # 마스크를 컬러 이미지로 변환
        # print(f'color_mask = {mask}')

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image) # 원본 이미지 표시
    axes[0].set_title('Original Image')
    axes[0].axis('off') # 축 숨기기

    if num_classes == 1:
        axes[1].imshow(mask, cmap='gray') # 예측된 마스크 표시 (Grayscale)
    else:
        axes[1].imshow(color_mask) # # 컬러 이미지 표시

    axes[1].set_title('Predicted Mask')
    axes[1].axis('off') # 축 숨기기
    
    # 범례 추가
    legend_elements = [Patch(facecolor=np.array(color)/255, 
                           label=f'{CLASS_NAMES[class_id]}')
                      for class_id, color in COLOR_MAP.items()]
    axes[1].legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout() # 레이아웃 조정
    plt.show()

def get_color_map(num_classes):
    """
    클래스 수에 따라 적절한 컬러 맵을 반환합니다.    
    Args:
        num_classes (int): 클래스의 수 (1 또는 4)
    Returns:
        dict: 클래스 번호에 대응하는 RGB 컬러 맵
    """
    if num_classes == 1:
        COLOR_MAP = {
            0: [0, 0, 0],       # 배경(검정)
            1: [255, 255, 255], # 클래스1(흰색)
        }
        CLASS_NAMES = {
            0: "배경",
            1: "OBJECT",
        }
    else:
        COLOR_MAP = {
            0: [0, 0, 0],       # 배경 (검정)
            1: [255, 0, 0],     # 클래스 1 원 (빨간색) 100 -> 1
            2: [0, 255, 0],     # 클래스 2 사각형 (초록색) 200 -> 2
            3: [0, 0, 255]      # 클래스 3 삼각형 (파란색) 255 -> 3
        }
        # 클래스 이름 매핑
        CLASS_NAMES = {
            0: "배경",
            1: "원",
            2: "사각형",
            3: "삼각형"
        }
                
    return COLOR_MAP, CLASS_NAMES

def predict_image(num_classes, img_path, model_path, img_height, img_width, DEVICE='cpu'):
    print(f'사용 중인 장치: {DEVICE}')

    # 데이터 변환 정의
    transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 모델 불러오기
    model = UNet(num_classes=num_classes, in_channels=3).to(DEVICE)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()  # 모델을 평가 모드로 설정 (Batch Normalization, Dropout 비활성화)

    # 이미지 불러오기 및 전처리
    image = Image.open(img_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)  # 배치 차원 추가

    # # for debug
    print(f'input_tenser.shape = {input_tensor.shape}')

    # 예측 수행
    with torch.no_grad():  # Gradient 계산 비활성화
        output = model(input_tensor)  # 모델에 입력 이미지 전달
        if num_classes == 1:
            output = torch.sigmoid(output) # Sigmoid 함수 적용 (출력값을 0과 1 사이로 변환)
            # predicted_mask = (output>0.22).float()
            predicted_mask = (output).float()
            predicted_mask = predicted_mask.squeeze()  # (1, H, W) -> (H, W)
        else:
            # (B, C, H, W) -> (B, H, W)
            predicted_mask = torch.argmax(output, dim=1, keepdim=True).squeeze()  # 클래스 번호 예측

    # # 예측 값 확인 (콘솔 출력)
    # print("Predicted Mask:\n", predicted_mask)

    # 예측 값 확인 (파일 저장)
    np.savetxt("predicted_mask.txt", predicted_mask.cpu().numpy(), fmt='%d') # NumPy 배열로 변환 후 저장
    # np.save("predicted_mask.npy", predicted_mask.cpu().numpy()) # NumPy 배열로 변환 후 저장

    return image, predicted_mask

if __name__ == "__main__":
    # plot용 한글 폰트 설정
    if platform.system() == 'Windows':
        font_name = 'Malgun Gothic'
    elif platform.system() == 'Darwin':
        font_name = 'AppleGothic'
    else:
        font_name = 'NanumGothic'

    plt.rc('font', family=font_name)
    plt.rcParams['axes.unicode_minus'] = False
    
    # 예측 설정
    # DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # GPU 사용 여부 확인
    DEVICE = torch.device('cpu')
    
    MODEL_PATH = 'unet_segmentation.pth'  # 학습된 모델 파일 경로 (실제 경로로 변경!)
    # IMAGE_PATH = 'sample_data/test1.png' # 예측할 이미지 파일 경로 (실제 경로로 변경!)
    IMAGE_PATH = 'drive_dataset/image_3.png' # 예측할 이미지 파일 경로 (실제 경로로 변경!)
    NUM_CLASSES = 1 # 클래스 개수 (배경 포함)

    # 이미지 크기 (학습 시 사용한 크기와 동일해야 함)
    input_height = 256   # 입력 이미지 높이
    input_width = 256    # 입력 이미지 너비

    image, predicted_mask = predict_image(NUM_CLASSES, IMAGE_PATH, MODEL_PATH, input_height, input_width, DEVICE)

    # 결과 출력
    visualize_result(image, predicted_mask, NUM_CLASSES)