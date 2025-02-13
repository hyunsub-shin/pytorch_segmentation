import os
from PIL import Image, ImageDraw
import random

def generate_sample_data(image_dir, mask_dir, num_samples=10, image_size=256):
    """
    샘플 이미지와 Segmentation Mask를 생성하여 지정된 디렉토리에 저장합니다.
    4가지 클래스 (배경, 원, 사각형, 삼각형)를 생성합니다.

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
        image = Image.new('RGB', (image_size, image_size), color='black')  # 검정색 배경
        draw_image = ImageDraw.Draw(image)

        # 2. 마스크 생성 (초기화)
        mask = Image.new('L', (image_size, image_size), color=0)  # 검정색 배경 (0으로 초기화)
        draw_mask = ImageDraw.Draw(mask)

        # 3. 객체 정보 리스트 (모양, 위치, 크기, 색상)
        objects = []

        # 4. 객체 생성 (최대 3개)
        num_objects = random.randint(1, 3)
        for _ in range(num_objects):
            shape = random.choice(['circle', 'rectangle', 'triangle'])  # 모양 선택
            x = random.randint(50, image_size - 50) # x 좌표
            y = random.randint(50, image_size - 50) # y 좌표
            size = random.randint(20, 70) # 크기
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) # 색상
            objects.append((shape, x, y, size, color))

        # 5. 이미지에 객체 그리기 및 마스크 생성
        for shape, x, y, size, color in objects:
            if shape == 'circle':
                draw_image.ellipse((x - size, y - size, x + size, y + size), fill=color)
                # 마스크에 클래스 ID를 채우기 (원: 1)->100
                draw_mask.ellipse((x - size, y - size, x + size, y + size), fill=100)
            elif shape == 'rectangle':
                draw_image.rectangle((x - size, y - size, x + size, y + size), fill=color)
                # 마스크에 클래스 ID를 채우기 (사각형: 2)->200
                draw_mask.rectangle((x - size, y - size, x + size, y + size), fill=200)
            elif shape == 'triangle':
                points = [(x, y - size), (x - size, y + size), (x + size, y + size)]
                draw_image.polygon(points, fill=color)
                # 마스크에 클래스 ID를 채우기 (삼각형: 3)->255
                draw_mask.polygon(points, fill=255)

        image_path = os.path.join(image_dir, f'image_{i}.png')
        image.save(image_path)

        mask_path = os.path.join(mask_dir, f'mask_{i}.png')
        mask.save(mask_path)

        print(f'생성 완료: {image_path}, {mask_path}')

if __name__ == '__main__':
    # 생성할 이미지와 마스크를 저장할 디렉토리 경로 설정
    IMAGE_DIR = 'sample_data/train/image'
    MASK_DIR = 'sample_data/train/mask'
    NUM_SAMPLES = 50

    # 샘플 데이터 생성
    generate_sample_data(IMAGE_DIR, MASK_DIR, num_samples=NUM_SAMPLES)
    print('샘플 데이터 생성 완료!')