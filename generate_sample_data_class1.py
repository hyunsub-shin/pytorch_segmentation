import os
from PIL import Image, ImageDraw
import random

def generate_sample_data(image_dir, mask_dir, num_samples=100, image_size=256):
    """
    샘플 이미지와 Segmentation Mask를 생성하여 지정된 디렉토리에 저장합니다.

    Args:
        image_dir (str): 이미지를 저장할 디렉토리 경로
        mask_dir (str): 마스크를 저장할 디렉토리 경로
        num_samples (int): 생성할 샘플 수 (기본값: 100)
        image_size (int): 이미지 크기 (기본값: 256)
    """

    # 디렉토리 생성
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    for i in range(num_samples):
        # 1. 이미지 생성
        image = Image.new('RGB', (image_size, image_size), color='black') # 검정색 배경 이미지 생성
        draw_image = ImageDraw.Draw(image)

        # 2. 마스크 생성
        mask = Image.new('L', (image_size, image_size), color='black') # 검정색 배경 마스크 생성 (Grayscale)
        draw_mask = ImageDraw.Draw(mask)

        # 무작위 위치와 크기를 가진 원 그리기 (이미지와 마스크에 동일하게 적용)
        num_circles = random.randint(1, 5) # 이미지에 그릴 원의 개수 (1 ~ 5개)
        circles = [] # 원의 정보를 저장할 리스트
        for _ in range(num_circles):
            x = random.randint(0, image_size)
            y = random.randint(0, image_size)
            r = random.randint(10, 50) # 원의 반지름 (10 ~ 50 픽셀)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) # 무작위 색상
            circles.append((x, y, r, color)) # 원의 정보를 리스트에 저장

        # 이미지에 원 그리기
        for x, y, r, color in circles:
            draw_image.ellipse((x - r, y - r, x + r, y + r), fill=color)

        # 마스크에 원 그리기 (흰색)
        for x, y, r, _ in circles: # 색상 정보는 사용하지 않음
            draw_mask.ellipse((x - r, y - r, x + r, y + r), fill='white')

        image_path = os.path.join(image_dir, f'image_{i}.png')
        image.save(image_path)

        mask_path = os.path.join(mask_dir, f'mask_{i}.png')
        mask.save(mask_path)

        print(f'생성 완료: {image_path}, {mask_path}')

if __name__ == '__main__':
    # 생성할 이미지와 마스크를 저장할 디렉토리 경로 설정
    IMAGE_DIR = 'sample_data/train/image'
    MASK_DIR = 'sample_data/train/mask'
    NUM_SAMPLES = 100

    # 샘플 데이터 생성
    generate_sample_data(IMAGE_DIR, MASK_DIR, num_samples=NUM_SAMPLES) # 100개의 샘플 생성
    print('샘플 데이터 생성 완료!')