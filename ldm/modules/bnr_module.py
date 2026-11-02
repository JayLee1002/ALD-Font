import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class LightVGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_FEATURES).features
        self.slice = nn.Sequential(*vgg[:9]).eval()  # up to relu2_2
        for p in self.slice.parameters():
            p.requires_grad = False

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def preprocess(self, x):
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        return x

    def forward(self, x, y):
        x = self.preprocess(x)
        y = self.preprocess(y)
        x_feat = self.slice(x)
        y_feat = self.slice(y)
        return F.l1_loss(x_feat, y_feat)

def sobel_edges(img):
    Gx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=img.device).view(1, 1, 3, 3)
    Gy = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=img.device).view(1, 1, 3, 3)
    
    if img.shape[1] == 3:  # RGB
        img = img.mean(dim=1, keepdim=True)  # 灰度化
    
    edge_x = F.conv2d(img, Gx, padding=1)
    edge_y = F.conv2d(img, Gy, padding=1)
    edge = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)
    return edge

def sobel_loss(output, target):
    return F.l1_loss(sobel_edges(output), sobel_edges(target))

class DoubleConv(nn.Module):
    """Conv2d -> BN -> ReLU (x2)"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class BNR(nn.Module):
    def __init__(self, in_channels=6, out_channels=3, base_c=64, ckpt_path=None):
        super().__init__()
        # Down
        self.enc1 = DoubleConv(in_channels, base_c)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base_c, base_c * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base_c * 2, base_c * 4)
        self.pool3 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(base_c * 4, base_c * 8)
        
        # Up
        self.up3 = nn.ConvTranspose2d(base_c * 8, base_c * 4, 2, stride=2)
        self.dec3 = DoubleConv(base_c * 8, base_c * 4)
        self.up2 = nn.ConvTranspose2d(base_c * 4, base_c * 2, 2, stride=2)
        self.dec2 = DoubleConv(base_c * 4, base_c * 2)
        self.up1 = nn.ConvTranspose2d(base_c * 2, base_c, 2, stride=2)
        self.dec1 = DoubleConv(base_c * 2, base_c)
        
        self.final = nn.Conv2d(base_c, out_channels, 1)
        
        if ckpt_path:
            self.resume(ckpt_path)
            
        self.hyper_parameters = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "base_c": base_c,
        }
        
        object.__setattr__(self, "light_vgg_loss", LightVGGPerceptualLoss())

    def forward(self, model_out, source_img):
        # Encoder
        x = self.process(model_out, source_img)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        # Bottleneck
        b = self.bottleneck(self.pool3(e3))

        # Decoder
        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.post_process(torch.sigmoid(self.final(d1)))  # assuming image output in [0,1]
    
    def resume(self, ckpt_path):
        state_dict = torch.load(ckpt_path, "cpu")["state_dict"]
        self.load_state_dict(state_dict)
        print(f"UNetPostProcess model loaded successfully from {ckpt_path}")
        
    def process(self, model_out, source_img):
        if self.hyper_parameters['in_channels'] == 6:
            return torch.cat([model_out, source_img], dim=1)
        elif self.hyper_parameters['in_channels'] == 4:
            model_out_bin = self.to_binary(model_out)
            source_img_one = source_img[:, :1]
            return torch.cat([model_out_bin, source_img_one], dim=1)
        
    def post_process(self, out):
        if self.hyper_parameters['out_channels'] == 1:
            out = torch.cat([out, out, out], dim=1)
        return out
        
    @staticmethod
    def to_binary(tensor):
        return (tensor >= 0.5).float()
    
    @staticmethod
    def from_config(json_path):
        import json
        with open(json_path, 'r') as f:
            config = json.load(f)
        model = BNR(**config['params'])
        return model
    
    def get_loss(self, model_out, source_img, target_raw):
        output = self(model_out, source_img)
        device = output.device
        self.light_vgg_loss.to(device)
        loss_l1 = torch.nn.functional.l1_loss(output, target_raw)
        loss_edge = sobel_loss(output, target_raw)
        vgg_loss = self.light_vgg_loss(output, target_raw)
        loss = loss_l1 + loss_edge + 0.05 * vgg_loss
        loss_info = {
                        "post_loss": round(loss.item(), 3),
                        "L_l1": round(loss_l1.item(), 3),
                        "L_edge": round(loss_edge.item(), 3),
                        "vgg_loss": round(vgg_loss.item(), 3)
                    }
        return loss, loss_info
