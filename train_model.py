import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from models.unet import *
import numpy as np
from tqdm import tqdm
import os
from PIL import Image
import platform  # 폰트관련 운영체제 확인
import matplotlib.pyplot as plt
from torch.multiprocessing import freeze_support
from torchvision import transforms

class SegmentationDataset(Dataset):
    def __init__(self, data_dir, dataset_type, image_transform=None, mask_transform=None, num_classes=1):       
        self.image_dir = os.path.join(data_dir, dataset_type, 'image')
        self.mask_dir = os.path.join(data_dir, dataset_type, 'mask')
        
        # 디렉토리 존재 여부 확인
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"이미지 디렉토리가 없습니다: {self.image_dir}")
        if not os.path.exists(self.mask_dir):
            raise FileNotFoundError(f"마스크 디렉토리가 없습니다: {self.mask_dir}")
            
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        
        # 파일 이름 패턴 매칭 수정
        self.images = sorted([f for f in os.listdir(self.image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.masks = sorted([f for f in os.listdir(self.mask_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        # 이미지와 마스크 파일 개수 확인
        if len(self.images) != len(self.masks):
            raise ValueError(f"이미지({len(self.images)})와 마스크({len(self.masks)})의 개수가 일치하지 않습니다.")
            
        self.num_classes = num_classes
        
        print(f"데이터셋 크기: {len(self.images)} 이미지")

        # # for debug
        # # 파일 매칭 확인 로그
        # for img, mask in zip(self.images[:5], self.masks[:5]):
        #     print(f"이미지-마스크 쌍: {img} - {mask}")
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 특정 인덱스의 이미지-마스크 쌍 가져오기
        img_name = self.images[idx]
        mask_name = self.masks[idx]
        
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        try:
            # 이미지-마스크 쌍 로드
            image = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')
           
            if self.image_transform:
                # 현재 이미지-마스크 쌍에 대한 시드 설정
                # 데이터 증강시 image와 mask가 동일하게 적용하기 위해 seed 필요
                seed = np.random.randint(2147483647)

                # 이미지 transform 시드 적용
                torch.manual_seed(seed)                
                image = self.image_transform(image)

                # 마스크에도 동일한 transform 적용을 위해 같은 시드 재설정
                torch.manual_seed(seed)
                mask = self.mask_transform(mask)
            
            # CrossEntropyLoss는 클래스 인덱스가 0부터 num_classes - 1까지의 값을 가져야 한다
            # mask data에 맞게 수정 필요
            # 마스크 값 변환 (예: 200을 2로 변환)
            mask_array = np.array(mask)
            if self.num_classes > 1:
                mask_array[mask_array == 100] = 1  # 100을 1로 변환
                mask_array[mask_array == 200] = 2  # 200을 2로 변환
                mask_array[mask_array == 255] = 3  # 255을 3로 변환
            # else:
            #     mask_array[mask_array == 255] = 1  # 255을 1로 변환 ==> drive mask 이진화 마스크가 아님
                       
            mask1 = Image.fromarray(mask_array)

            # # for debug
            # plt.subplot(121)
            # plt.imshow(mask, cmap='gray')  # 마스크를 그레이스케일로 시각화
            # plt.subplot(122)
            # plt.imshow(mask1, cmap='gray')  # 마스크를 그레이스케일로 시각화 
            # plt.title("Mask Visualization")
            # plt.colorbar()  # 색상 바 추가
            # plt.show()

            # transform 에서 ToTensor 삭제 : Tensor로 변환 필요
            mask = torch.from_numpy(np.array(mask1))   # Image => NumPy 배열 => Tensor로 변환
            
            if self.num_classes == 1:
                mask = mask / 255.0   # class 범위를 0 ~ 1 값으로 변환
                mask = mask.float()
            else:
                # mask = mask.squeeze(0).long() 
                mask = mask.long()  # transform 에서 ToTensor 삭제 : 차원 삭제(squeeze) 미적용

            # 항상 쌍으로 반환
            return image, mask

        except Exception as e:
            print(f"이미지 로딩 에러 - {img_path} 또는 {mask_path}")
            print(f"에러 메시지: {str(e)}")
            raise e

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        # 평탄화
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
    
    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.alpha * bce_loss + (1 - self.alpha) * dice_loss

def verify_mask_files(dataset):
    print("\n마스크 파일 확인:")
    for i in range(min(5, len(dataset.masks))):  # 처음 5개 샘플만 확인
        mask_path = os.path.join(dataset.mask_dir, dataset.masks[i])
        # PIL로 읽기
        mask_pil = Image.open(mask_path)
        mask_array = np.array(mask_pil)
        print(f"\nMask file {i}: {dataset.masks[i]}")
        print(f"Original mask unique values: {np.unique(mask_array)}")
        print(f"Mask shape: {mask_array.shape}")
        print(f"Mask dtype: {mask_array.dtype}")
        
        # 이미지로 저장하여 확인
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.imshow(mask_array)
        plt.title('Original Mask')
        plt.colorbar()
        
        # 히스토그램 확인
        plt.subplot(122)
        plt.hist(mask_array.ravel(), bins=256)
        plt.title('Mask Histogram')
        plt.savefig(f'mask_check_{i}.png')
        plt.close()

def check_mask_values(dataset):
    print("\n마스크 값 확인:")
    for i in range(min(5, len(dataset))):  # 처음 5개 샘플만 확인
        _, mask = dataset[i]
        unique_values = torch.unique(mask)
        print(f"Sample {i} unique values: {unique_values}")
        print(f"Mask shape: {mask.shape}")
        print(f"Mask dtype: {mask.dtype}")
        
        # 각 클래스의 픽셀 수 계산
        for class_id in [0, 1, 2, 3]:  # 매핑된 클래스 값 사용
            pixel_count = (mask == class_id).sum().item()
            print(f"Class {class_id} pixels: {pixel_count}")
            
        print("-" * 50)

def check_class_distribution(dataset):
    print("\n클래스 분포:")
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # 0: 배경, 1: 원, 2: 사각형, 3: 삼각형
    total_pixels = 0

    for i in range(len(dataset)):
        _, mask = dataset[i]
        unique_values = torch.unique(mask)
        
        for val in unique_values:
            if val.item() in class_counts:
                class_counts[val.item()] += (mask == val).sum().item()
                total_pixels += (mask == val).sum().item()

    for class_id, count in class_counts.items():
        percentage = (count / total_pixels) * 100 if total_pixels > 0 else 0
        print(f"클래스 {class_id}: {percentage:.2f}% ({count} pixels)")
                    
def get_transforms(input_height, input_width, is_train=True):
    """
    데이터 변환(transform) 함수를 반환합니다.
    
    Args:
        input_height (int): 입력 이미지 높이
        input_width (int): 입력 이미지 너비
        is_train (bool): 학습용 transform 여부
    
    Returns:
        image_transform, mask_transform (tuple): 이미지와 마스크에 대한 transform 함수들
    """
    # 이미지 transform 설정
    if is_train: # 트레인 transform
        # 이미지 transform 설정: 이미지 증강 적용
        image_transform = transforms.Compose([
            transforms.Resize((input_height, input_width)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),            
            # transforms.Normalize(mean=[0.7, 0.5, 0.3], std=[0.2, 0.2, 0.2])  # 안저 이미지에 특화된 정규화 값 사용
        ])
        # 마스크 transform 설정
        mask_transform = transforms.Compose([
            transforms.Resize((input_height, input_width)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            # transforms.ToTensor(),  # 적용시 class값 이상 # 마스크는 정규화 하지 않음        
        ])
        
    else: # 테스트 transform
        # 이미지 transform 설정: 이미지 증강 미적용
        image_transform = transforms.Compose([
            transforms.Resize((input_height, input_width)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.7, 0.5, 0.3], std=[0.2, 0.2, 0.2])  # 안저 이미지에 특화된 정규화 값 사용
        ])
        # 마스크 transform 설정
        mask_transform = transforms.Compose([
            transforms.Resize((input_height, input_width)),
            # transforms.ToTensor(),  # 적용시 class값 이상 # 마스크는 정규화 하지 않음        
        ])   
    
    return image_transform, mask_transform

def create_dataloaders(data_dir, input_height, input_width, batch_size, num_classes, is_train=True):
    """
    데이터로더를 생성하는 함수입니다.
    
    Args:
        image_dir (str): 이미지 디렉토리 경로
        mask_dir (str): 마스크 디렉토리 경로
        input_height (int): 입력 이미지 높이
        input_width (int): 입력 이미지 너비
        batch_size (int): 배치 크기
        num_classes (int): 클래스 개수
    
    Returns:
        train_loader, test_loader (tuple): 학습용과 테스트용 데이터로더
    """
    # Transform 함수 가져오기
    image_transform, mask_transform = get_transforms(input_height, input_width, is_train)
    
    if is_train:
        # train 데이터셋 생성
        dataset = SegmentationDataset(
            data_dir=data_dir,
            dataset_type="train",
            image_transform=image_transform,
            mask_transform=mask_transform,
            num_classes=num_classes
        )
        # train 데이터로더 생성
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,  # CPU 코어 개수에 맞춰 조정 가능
            pin_memory=True  # GPU 사용 시 학습 속도 향상
        )
    else:
        # test 데이터셋 생성
        dataset = SegmentationDataset(
            data_dir=data_dir,
            dataset_type="test",
            image_transform=image_transform,
            mask_transform=mask_transform,
            num_classes=num_classes
        )
        # test 데이터로더 생성        
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,  # CPU 코어 개수에 맞춰 조정 가능
            pin_memory=True  # GPU 사용 시 학습 속도 향상
        )
    
    return data_loader

def visualize_batch(images, masks, epoch, batch_idx):
    """배치의 이미지와 마스크를 시각화하는 함수"""
    # CPU로 데이터 이동 및 numpy 변환
    images = images.cpu().numpy()
    masks = masks.cpu().numpy()
    
    # 배치에서 4개의 이미지만 선택 (또는 배치 크기가 4보다 작은 경우 전체)
    n_samples = min(4, images.shape[0])
    
    plt.figure(figsize=(15, 5))
    for idx in range(n_samples):
        # 원본 이미지
        plt.subplot(2, n_samples, idx + 1)
        # 채널 순서 변경 (C,H,W) -> (H,W,C)
        img = images[idx].transpose(1, 2, 0)
        # 정규화 해제
        img = img * np.array([0.2, 0.2, 0.2]) + np.array([0.7, 0.5, 0.3])
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.title(f'Image {idx+1}')
        plt.axis('off')
        
        # 마스크
        plt.subplot(2, n_samples, idx + 1 + n_samples)
        plt.imshow(masks[idx].squeeze(), cmap='gray')
        plt.title(f'Mask {idx+1}')
        plt.axis('off')
    
    plt.suptitle(f'Epoch {epoch+1}, Batch {batch_idx}')
    plt.tight_layout()
    
    # 그래프를 파일로 저장
    plt.savefig(f'batch_visualization_epoch_{epoch+1}_batch_{batch_idx}.png')
    plt.close()

def plot_training_loss(train_losses):
    """학습 히스토리를 시각화하는 함수"""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.title('학습 손실 변화')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_result.png')
    plt.close()

def train_model():
    ###################################################
    # # 하이퍼 파라메터 설정
    num_epochs = 10
    batch_size = 4
    learning_rate = 1e-4
    
    # 데이터 경로 설정
    DATA_DIR = './drive_dataset'
    num_classes = 1
    # DATA_DIR = './sample_data'
    # num_classes = 4
    
    # 입력 크기 설정
    input_height = 256
    input_width = 256
    ###################################################
    
    # 장치 설정
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    
    if device.type == 'cuda':        
        print(f'현재 사용 중인 GPU: {torch.cuda.get_device_name(0)}')
        
        # CUDA 설정
        torch.backends.cudnn.benchmark = True # 속도 향상을 위한 설정
        torch.backends.cudnn.deterministic = True # 재현 가능성 확보
        torch.backends.cuda.matmul.allow_tf32 = True  # TensorFloat-32 사용        
        torch.cuda.empty_cache()
        # 메모리 할당 모드 설정
        torch.cuda.set_per_process_memory_fraction(0.8)  # GPU 메모리의 80% 사용

    print("\n=== 시스템 정보 ===")
    print(f"CUDA 사용 가능 여부: {torch.cuda.is_available()}")
    print(f"현재 PyTorch의 CUDA 버전: {torch.version.cuda}")
    print(f"PyTorch 버전: {torch.__version__}") # 예: '2.0.0+cu121'는 CUDA 12.1 버전을 지원
    print(f'사용 중인 장치: {device}')
    print('-' * 50)
    
    # 모델 초기화
    model = UNet(num_classes=num_classes, in_channels=3).to(device)
    
    # 손실 함수 설정
    if num_classes == 1:
        # 이진 분류일 경우
        criterion = nn.BCEWithLogitsLoss()
        # # for test
        # criterion = CombinedLoss(alpha=0.5)
    else:
        # 다중 클래스일 경우
        criterion = nn.CrossEntropyLoss()
    
    # 옵티마이저 설정
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 데이터로더 생성
    train_loader = create_dataloaders(
        data_dir=DATA_DIR,
        input_height=input_height,
        input_width=input_width,
        batch_size=batch_size,
        num_classes=num_classes,
        is_train=True
    )
    test_loader = create_dataloaders(
        data_dir=DATA_DIR,
        input_height=input_height,
        input_width=input_width,
        batch_size=batch_size,
        num_classes=num_classes,
        is_train=False
    )
    
    print("\n=== 학습 시작 ===")
    train_losses = []    
    # 학습 루프
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (images, masks) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch')):
            images = images.to(device)
            masks = masks.to(device)
                       
            # # for debug
            # # 첫 번째 배치와 마지막 배치의 이미지와 마스크 시각화
            # if batch_idx in [0, len(train_loader)-1]:
            #     visualize_batch(images, masks, epoch, batch_idx)
            
            # 순전파
            outputs = model(images)
            
            # 손실 계산
            if num_classes == 1:
                # # mask transform에서 ToTensor삭제로 채널 차원 추가
                # # [4 ,256, 256] => [4, 1, 256, 256]
                # 이진 분류일 경우 마스크를 float로 변환
                masks = masks.unsqueeze(1).float()
                loss = criterion(outputs, masks)
            else:
                # # 다중 클래스일 경우 masks를 3D 텐서로 변환
                # loss = criterion(outputs, masks.squeeze(1).long())
                loss = criterion(outputs, masks.long()) # mask transform에서 ToTensor삭제로 차원 삭제(squeeze) 미적용
            
            # 역전파 및 옵티마이저 스텝
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # 메모리 정리 및 사용량 출력 (10 배치마다)
            if batch_idx % 10 == 0:
                del outputs, loss   # 메모리 정리: Tensor 객체 삭제                
                if device.type == 'cuda':
                    torch.cuda.empty_cache()    # 캐시된 메모리 정리
        
        train_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch+1} average loss: {train_loss:.4f}')
        train_losses.append(train_loss) # 에포크당 평균 손실 추가
        
        # 모델 저장 - checkpoint
        # if (epoch + 1) % 10 == 0:
        #     torch.save(model.state_dict(), f'checkpoints/unet_epoch_{epoch+1}.pth')
            
    # 모델 저장
    torch.save(model.state_dict(), 'unet_segmentation.pth')
    print('Finished Training & Save model')

    # 학습이 끝난 후 손실 그래프 그리기
    plot_training_loss(train_losses)

if __name__ == '__main__':
    freeze_support()  # Windows에서 필요
    
    # plot용 한글 폰트 설정
    if platform.system() == 'Windows':
        font_name = 'Malgun Gothic'
    elif platform.system() == 'Darwin':
        font_name = 'AppleGothic'
    else:
        font_name = 'NanumGothic'

    plt.rc('font', family=font_name)
    plt.rcParams['axes.unicode_minus'] = False
    
    train_model()
    
    # ##################################################
    # # for debug
    # # 학습 시작 전에 검증 실행
    # train_loader = create_dataloaders(
    #     data_dir='drive_dataset',
    #     input_height=256,
    #     input_width=256,
    #     batch_size=4,
    #     num_classes=1,
    #     is_train=True
    # )
    # train_dataset = train_loader.dataset
    
    # # verify_mask_files(train_dataset)
    # check_mask_values(train_dataset)
    # check_class_distribution(train_dataset)
    # ##################################################
        
    
