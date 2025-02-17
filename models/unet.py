import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            # out = (input - kernel + 2*padding)/stride + 1
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class TripleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.triple_conv = nn.Sequential(
            # out = (input - kernel + 2*padding)/stride + 1
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.triple_conv(x)

class UNet(nn.Module):
    def __init__(self, num_classes=1, in_channels=3):
        super().__init__()
        
        # Encoder
        self.enc1 = TripleConv(in_channels, 64)
        self.enc2 = TripleConv(64, 128)
        self.enc3 = TripleConv(128, 256)
        self.enc4 = TripleConv(256, 512)
        
        # buttleneck 대체
        self.enc5 = DoubleConv(512, 1024)
        
        # Decoder + Skip Connection
        self.dec4 = TripleConv(1024, 512)
        self.dec3 = TripleConv(512, 256)
        self.dec2 = TripleConv(256, 128)
        self.dec1 = TripleConv(128, 64)
        
        # Max pooling
        self.pool = nn.MaxPool2d(2)
        
        # Upsampling
        self.up = nn.ModuleList([
            # Output = Stride × (Input − 1) + Kernel Size − (2 × Padding)
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
        ])
        
        # Final convolution
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        enc5 = self.enc5(self.pool(enc4))
        
        # Decoder
        dec4 = self.dec4(torch.cat([self.up[0](enc5), enc4], dim=1))
        dec3 = self.dec3(torch.cat([self.up[1](dec4), enc3], dim=1))
        dec2 = self.dec2(torch.cat([self.up[2](dec3), enc2], dim=1))
        dec1 = self.dec1(torch.cat([self.up[3](dec2), enc1], dim=1))
        
        return self.final_conv(dec1) 

# Generate google AI
class UNet_googleAI(nn.Module):
    def __init__(self, num_classes=1, in_channels=3):
        super().__init__()

        # Convolution Block: Convolution -> Batch Normalization -> ReLU
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        # Bottleneck Block: 여러 개의 Convolution Layer, Batch Normalization, ReLU (함수로 구현)
        def bottleneck_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),  # 추가적인 Conv Layer
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        # Encoder (Downsampling)
        self.enc1 = conv_block(in_channels, 64)  # 입력 채널: 3 (RGB), 출력 채널: 64
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)

        # Bottleneck
        self.bottleneck = bottleneck_block(512, 1024)  # Bottleneck 함수 사용

        # pool
        self.pool = nn.MaxPool2d(kernel_size=2)
        
        # Decoder (Upsampling)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2) # Up-Convolution
        self.dec4 = conv_block(1024, 512) # Skip connection으로 Concatenate된 Feature Map을 처리
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = conv_block(128, 64)

        # Output Layer
        self.outconv = nn.Conv2d(64, num_classes, kernel_size=1) # 출력 채널: 1 (Segmentation Mask)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1)) # Max Pooling으로 Downsampling
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))  # Bottleneck 함수 사용

        # Decoder
        dec4 = self.dec4(torch.cat((self.upconv4(bottleneck), enc4), dim=1)) # Skip connection (Concatenate)
        dec3 = self.dec3(torch.cat((self.upconv3(dec4), enc3), dim=1))
        dec2 = self.dec2(torch.cat((self.upconv2(dec3), enc2), dim=1))
        dec1 = self.dec1(torch.cat((self.upconv1(dec2), enc1), dim=1))

        # Output
        output = self.outconv(dec1)
        return output