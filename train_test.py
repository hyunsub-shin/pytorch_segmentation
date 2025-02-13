import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from models.unet import UNet
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
            
            # #####################################################################
            # # sample data class4 generate test    
            # if self.num_classes > 1:     
            #     # 클래스 값을 0,1,2,3으로 매핑
            #     mapped_mask = np.zeros_like(mask, dtype=np.uint8)
            #     for orig_val, mapped_val in {0: 0, 100: 1, 200: 2, 255: 3}.items():
            #         mapped_mask[mask == orig_val] = mapped_val
                
            #     # 매핑된 NumPy 배열 마스크를 PIL 이미지로 변환
            #     mask = Image.fromarray(mapped_mask.astype(np.uint8))
            # #####################################################################   
                     
            if self.image_transform:
                # 현재 이미지-마스크 쌍에 대한 시드 설정
                # 데이터 증강시 image와 mask가 동일하게 적용하기 위해 seed 필요
                seed = np.random.randint(2147483647)

                # 이미지 transform 시드 적용
                torch.manual_seed(seed)                
                image = self.image_transform(image)

                # 마스크에도 동일한 transform 적용을 위해 같은 시드 재설정
                torch.manual_seed(seed)
                mask1 = self.mask_transform(mask)
                
                mask_array = np.array(mask1)
                mask_array[mask_array == 100] = 1  # 100을 1로 변환
                mask_array[mask_array == 200] = 2  # 200을 2로 변환
                mask_array[mask_array == 255] = 3  # 255을 3로 변환
                mask1 = Image.fromarray(mask_array)
                ###################################################################
                # for debug
                plt.subplot(121)
                plt.imshow(mask, cmap='gray')  # 마스크를 그레이스케일로 시각화
                plt.subplot(122)
                plt.imshow(mask1, cmap='gray')  # 마스크를 그레이스케일로 시각화 
                plt.title("Mask Visualization")
                plt.colorbar()  # 색상 바 추가
                plt.show()
                ###################################################################
                
            # 매핑된 NumPy 배열 마스크를 LongTensor로 변환
            mask1 = np.array(mask1)
            mask1 = torch.from_numpy(mask1).long()  # NumPy 배열을 LongTensor로 변환
            
            if self.num_classes == 1:
                mask = mask1 / 255.0
                mask = mask.float()
            else:
                # mask = mask.squeeze(0).long()
                mask = mask1.long()
                
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
    
class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        # 클래스 1,2,3에 더 높은 가중치 부여
        self.weight = torch.FloatTensor([1.0, 5.0, 5.0, 5.0]).to(device)
        self.criterion = nn.CrossEntropyLoss(weight=self.weight)
    
    def forward(self, pred, target):
        return self.criterion(pred, target)

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
            # transforms.ToTensor(),  # 마스크는 정규화 하지 않음            
        ])
    else: # 테스트 transform
        # 이미지 transform 설정: 이미지 증강 미적용
        image_transform = transforms.Compose([
            transforms.Resize((input_height, input_width)),
            # transforms.ToTensor(),
            # transforms.Normalize(mean=[0.7, 0.5, 0.3], std=[0.2, 0.2, 0.2])  # 안저 이미지에 특화된 정규화 값 사용
        ])
        # 마스크 transform 설정
        mask_transform = transforms.Compose([
            transforms.Resize((input_height, input_width)),
            # transforms.ToTensor(),  # 마스크는 정규화 하지 않음            
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
            num_workers=4,  # CPU 코어 개수에 맞춰 조정 가능
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

def plot_training_history(train_losses):
    """학습 히스토리를 시각화하는 함수"""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.title('학습 손실 변화')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_history.png')
    plt.close()

def check_class_distribution(dataset):
    class_counts = torch.zeros(4)  # 배경 포함 4개 클래스
    for _, mask in dataset:
        for i in range(4):
            class_counts[i] += (mask == i).sum().item()
    
    total_pixels = class_counts.sum()
    print("\n클래스 분포:")
    for i in range(4):
        percentage = (class_counts[i] / total_pixels) * 100
        print(f"클래스 {i}: {percentage:.2f}% ({class_counts[i]})")

def calculate_class_weights(dataset):
    class_counts = torch.zeros(4)
    for _, mask in dataset:
        for i in range(4):
            class_counts[i] += (mask == i).sum().item()
    
    # 클래스 가중치 계산
    total_pixels = class_counts.sum()
    class_weights = total_pixels / (class_counts * 4)  # 역수 가중치
    return class_weights.to(device)

def visualize_predictions(model, val_loader, num_samples=4):
    model.eval()
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    
    with torch.no_grad():
        for i, (images, masks) in enumerate(val_loader):
            if i >= num_samples:
                break
                
            images = images.to(device)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            
            # CPU로 이동 및 numpy 변환
            image = images[0].cpu().numpy().transpose(1,2,0)
            mask = masks[0].cpu().numpy()
            pred = predictions[0].cpu().numpy()
            
            # 정규화 해제
            image = image * np.array([0.2, 0.2, 0.2]) + np.array([0.7, 0.5, 0.3])
            image = np.clip(image, 0, 1)
            
            # 시각화
            axes[i,0].imshow(image)
            axes[i,0].set_title('Input Image')
            axes[i,1].imshow(mask, cmap='tab10')
            axes[i,1].set_title('True Mask')
            axes[i,2].imshow(pred, cmap='tab10')
            axes[i,2].set_title('Prediction')
            
    plt.tight_layout()
    plt.show()

def verify_mask_files(dataset):
    print("\n마스크 파일 확인:")
    for i in range(min(5, len(dataset.masks))):
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
        
        # print(f'mask type ::::: {mask.dtype}')
        # plt.imshow(mask.numpy(), cmap='gray')  # 마스크를 그레이스케일로 시각화
        # plt.title("Mask Visualization")
        # plt.colorbar()  # 색상 바 추가
        # plt.show()

        unique_values = torch.unique(mask)
        print(f"Sample {i} unique values: {unique_values}")
        
        # 각 클래스의 픽셀 수 계산
        for class_id in [0, 100, 200, 255]:  # 매핑된 클래스 값 사용
            pixel_count = (mask == class_id).sum().item()
            print(f"Class {class_id} pixels: {pixel_count}")
        
        print("-" * 50)

def check_class_distribution(dataset):
    print("\n클래스 분포:")
    class_counts = {0: 0, 100: 0, 200: 0, 255: 0}  # 0: 배경, 1: 원, 2: 사각형, 3: 삼각형
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
        
def check_model_output(model, train_loader):
    print("\n모델 출력 확인:")
    model.eval()
    with torch.no_grad():
        images, masks = next(iter(train_loader))
        images = images.to(device)
        outputs = model(images)
        
        print(f"Output shape: {outputs.shape}")
        print(f"Output min: {outputs.min().item():.4f}")
        print(f"Output max: {outputs.max().item():.4f}")
        
        predictions = torch.argmax(outputs, dim=1)
        unique_preds = torch.unique(predictions)
        print(f"Unique predictions: {unique_preds}")
        
        # 각 클래스의 예측 수 계산
        for c in range(4):
            pred_count = (predictions == c).sum().item()
            print(f"Class {c} predictions: {pred_count}")

def verify_dataset(dataset):
    print("\n데이터셋 검증:")
    class_counts = torch.zeros(4)
    total_samples = 0
    
    for _, mask in dataset:
        total_samples += 1
        for c in range(4):
            if (mask == c).any():
                class_counts[c] += 1
    
    print(f"총 샘플 수: {total_samples}")
    for c in range(4):
        print(f"클래스 {c}를 포함하는 샘플 수: {class_counts[c]} "
              f"({(class_counts[c]/total_samples)*100:.2f}%)")

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (images, masks) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch')):
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # 각 클래스별 손실 계산
            if batch_idx % 10 == 0:  # 10배치마다 출력
                predictions = torch.argmax(outputs, dim=1)
                bce_criterion = nn.BCEWithLogitsLoss()  # 클래스별 손실 계산용
                for c in range(4):
                    class_mask = (masks == c)
                    if class_mask.any():
                        class_loss = bce_criterion(
                            outputs[:, c, :, :],  # c번째 클래스의 출력
                            class_mask.float()    # c번째 클래스의 마스크
                        )
                        print(f"Epoch {epoch}, Batch {batch_idx}, "
                              f"Class {c} loss: {class_loss.item():.4f}")
            
            loss.backward()
            optimizer.step()
            
        # 검증 단계
        model.eval()
        with torch.no_grad():
            val_predictions = []
            val_masks = []
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                predictions = torch.argmax(outputs, dim=1)
                
                val_predictions.extend(predictions.cpu().numpy())
                val_masks.extend(masks.cpu().numpy())
            
            # 클래스별 IoU 계산
            val_predictions = np.array(val_predictions)
            val_masks = np.array(val_masks)
            for c in range(4):
                intersection = np.logical_and(val_predictions == c, 
                                           val_masks == c).sum()
                union = np.logical_or(val_predictions == c, 
                                    val_masks == c).sum()
                iou = intersection / (union + 1e-10)
                print(f"Class {c} IoU: {iou:.4f}")

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
    
    #################################
    # class 수 설정
    NUM_CLASSES = 4

    # 데이터 경로 설정
    DATA_DIR = './sample_data'
    #################################
    
    # 장치 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    print("\n=== 시스템 정보 ===")
    print(f"CUDA 사용 가능 여부: {torch.cuda.is_available()}")
    print(f"현재 PyTorch의 CUDA 버전: {torch.version.cuda}")
    print(f"PyTorch 버전: {torch.__version__}") # 예: '2.0.0+cu121'는 CUDA 12.1 버전을 지원
    print(f'사용 중인 장치: {device}')
    print('-' * 50)
    
    # 모델 초기화
    model = UNet(num_classes=NUM_CLASSES, in_channels=3).to(device)
    
    # 손실 함수 설정
    if NUM_CLASSES == 1:
        # 이진 분류일 경우
        criterion = nn.BCEWithLogitsLoss()
        # for test
        # criterion = CombinedLoss(alpha=0.5)
    else:
        # 다중 클래스일 경우
        criterion = WeightedCrossEntropyLoss()
    
    # 옵티마이저 설정
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # 데이터로더 생성
    train_loader = create_dataloaders(
        data_dir=DATA_DIR,
        input_height=256,
        input_width=256,
        batch_size=4,
        num_classes=NUM_CLASSES,
        is_train=True
    )
    val_loader = create_dataloaders(
        data_dir=DATA_DIR,
        input_height=256,
        input_width=256,
        batch_size=4,
        num_classes=NUM_CLASSES,
        is_train=False
    )
    
    # 데이터셋 가져오기
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset

    # 학습 시작 전에 검증 실행
    # verify_mask_files(train_dataset)
    check_mask_values(train_dataset)
    check_class_distribution(train_dataset)
    # check_model_output(model, train_loader)

    print("\n=== 학습 시작 ===")
    
    train_losses = []    
    # 학습 루프
    # train_model(model, train_loader, val_loader, criterion, optimizer, 20)


