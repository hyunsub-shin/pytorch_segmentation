import deeplake
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt  # Importing matplotlib for plotting

# # Deeplake TRAIN 데이터셋 로드
ds = deeplake.load("hub://activeloop/drive-train")

# 데이터셋 내 텐서 목록 출력
print(f'ds.key is : {ds}')  # 텐서 목록을 확인하여 정확한 이름을 확인합니다.

# 이미지와 마스크 텐서 선택 (예시로 rgb_images와 masks/mask 선택)
images = ds["rgb_images"]  # 이미지 데이터
masks = ds["manual_masks/mask"]  # 마스크 데이터

# 저장할 폴더 구조 설정
train_image_folder = './drive_dataset/train/image'
train_mask_folder = './drive_dataset/train/mask'

# 폴더가 없으면 생성
os.makedirs(train_image_folder, exist_ok=True)
os.makedirs(train_mask_folder, exist_ok=True)

# # 이미지와 마스크 파일을 'train/image'와 'train/mask'로 저장
for i, (image, mask) in enumerate(zip(images, masks)):
    # 파일 경로 설정
    image_file_path = os.path.join(train_image_folder, f"image_{i+1}.png")
    mask_file_path = os.path.join(train_mask_folder, f"mask_{i+1}.png")
    
    # 이미지와 마스크 저장 (numpy로 변환 후 저장)
    image_array = image.numpy()  # numpy 배열로 변환
    mask_array = mask.numpy()  # numpy 배열로 변환

    # mask_array가 3D (584, 565, 2)라면 첫 번째 채널만 추출
    if len(mask_array.shape) == 3 and mask_array.shape[2] == 2:
        mask_array = mask_array[:, :, 0]  # 첫 번째 채널만 사용 (배경 또는 일부 클래스)
        
    # # 마스크 배열이 boolean일 경우, uint8로 변환 (0, 255로 변환)
    # if mask_array.dtype == bool:  # np.bool을 bool로 변경
    #     # mask_array = (mask_array * 255).astype(np.uint8)
    #     mask_array = np.where(mask_array, 0, 1).astype(np.uint8)  # True는 255, False는 0으로 변환
    # elif mask_array.max() == 1:
    mask_array = (mask_array * 255).astype(np.uint8)
    mask_array = 255 - mask_array # 색상 반전
            
    # 이미지 배열을 PIL 이미지로 변환
    image_pil = Image.fromarray(image_array.astype(np.uint8))  # 이미지도 uint8로 변환
    mask_pil = Image.fromarray(mask_array)

    # 이미지 파일 저장
    image_pil.save(image_file_path)
    mask_pil.save(mask_file_path)
    
    # 마스크 이미지를 시각화 (plot)
    # plt.figure(figsize=(6, 6))
    # plt.imshow(mask_array, cmap='gray')  # Use 'gray' colormap to display mask
    # plt.title(f'Mask {i+1}')  # Title with mask index
    # plt.axis('off')  # Hide axes
    # plt.show()  # Display the mask

print("TRAIN 이미지와 마스크가 성공적으로 저장되었습니다.")



# # Deeplake TEST 데이터셋 로드
ds = deeplake.load("hub://activeloop/drive-test")

# 데이터셋 내 텐서 목록 출력
print(f'ds.key is : {ds}')  # 텐서 목록을 확인하여 정확한 이름을 확인합니다.

# 이미지와 마스크 텐서 선택 (예시로 rgb_images와 masks/mask 선택)
images = ds["rgb_images"]  # 이미지 데이터
masks = ds["masks"]  # 마스크 데이터

# 저장할 폴더 구조 설정
test_image_folder = './drive_dataset/test/image'
test_mask_folder = './drive_dataset/test/mask'

# 폴더가 없으면 생성
os.makedirs(test_image_folder, exist_ok=True)
os.makedirs(test_mask_folder, exist_ok=True)

# 이미지와 마스크 파일을 'test/image'와 'test/mask'로 저장
for i, (image, mask) in enumerate(zip(images, masks)):
    # 파일 경로 설정
    image_file_path = os.path.join(test_image_folder, f"image_{i+1}.png")
    mask_file_path = os.path.join(test_mask_folder, f"mask_{i+1}.png")
    
    # 이미지와 마스크 저장 (numpy로 변환 후 저장)
    image_array = image.numpy()  # numpy 배열로 변환
    mask_array = mask.numpy()  # numpy 배열로 변환
    
    # mask_array가 3D (584, 565, 2)라면 첫 번째 채널만 추출
    if len(mask_array.shape) == 3 and mask_array.shape[2] == 2:
        mask_array = mask_array[:, :, 0]  # 첫 번째 채널만 사용 (배경 또는 일부 클래스)
        
    # # 마스크 배열이 boolean일 경우, uint8로 변환 (0, 255로 변환)
    # if mask_array.dtype == bool:  # np.bool을 bool로 변경
    #     # mask_array = (mask_array * 255).astype(np.uint8)
    #     mask_array = np.where(mask_array, 0, 1).astype(np.uint8)  # True는 255, False는 0으로 변환
    # elif mask_array.max() == 1:
    mask_array = (mask_array * 255).astype(np.uint8)
    mask_array = 255 - mask_array # 색상 반전
    
    # 이미지 배열을 PIL 이미지로 변환
    image_pil = Image.fromarray(image_array.astype(np.uint8))  # 이미지도 uint8로 변환
    mask_pil = Image.fromarray(mask_array)

    # 이미지 파일 저장
    image_pil.save(image_file_path)
    mask_pil.save(mask_file_path)
    
    # 마스크 이미지를 시각화 (plot)
    # plt.figure(figsize=(6, 6))
    # plt.imshow(mask_array, cmap='gray')  # Use 'gray' colormap to display mask
    # plt.title(f'Mask {i+1}')  # Title with mask index
    # plt.axis('off')  # Hide axes
    # plt.show()  # Display the mask

print("TEST 이미지와 마스크가 성공적으로 저장되었습니다.")