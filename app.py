import io
import base64
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights

app = Flask(__name__)
CORS(app)

# --- ARCHITECTURE ---

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=32):
        super().__init__()
        self.down1 = DoubleConv(in_ch, base); self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(base, base*2); self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(base*2, base*4); self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(base*4, base*8); self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base*8, base*16)

        self.up4 = nn.ConvTranspose2d(base*16, base*8, 2, 2); self.dec4 = DoubleConv(base*16, base*8)
        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, 2); self.dec3 = DoubleConv(base*8, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, 2); self.dec2 = DoubleConv(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, 2); self.dec1 = DoubleConv(base*2, base)
        self.outc = nn.Conv2d(base, out_ch, 1)

    def encode(self, x):
        c1 = self.down1(x); p1 = self.pool1(c1)
        c2 = self.down2(p1); p2 = self.pool2(c2)
        c3 = self.down3(p2); p3 = self.pool3(c3)
        c4 = self.down4(p3); p4 = self.pool4(c4)
        bn = self.bottleneck(p4)
        return bn, (c1, c2, c3, c4)

class ViTEncoder(nn.Module):
    def __init__(self, vit):
        super().__init__()
        self.vit = vit
    def forward(self, x):
        x = self.vit._process_input(x)
        n = x.shape[0]
        cls_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.vit.encoder(x)
        return x

class ViTtoUNetFusion(nn.Module):
    def __init__(self, vit_dim, unet_dim):
        super().__init__()
        self.proj = nn.Linear(vit_dim, unet_dim)
    def forward(self, vit_patches, H, W):
        B, N, D = vit_patches.shape
        x = self.proj(vit_patches)
        x = x.transpose(1, 2).reshape(B, -1, 14, 14)
        return torch.nn.functional.interpolate(x, size=(H, W), mode="bilinear")

class ViTClassifierHead(nn.Module):
    def __init__(self, vit_dim=768, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vit_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    def forward(self, cls_token): return self.net(cls_token)

class FeatureFusionHybrid(nn.Module):
    def __init__(self, unet, vit_encoder, vit_to_unet):
        super().__init__()
        self.unet = unet
        self.vit = vit_encoder
        self.vit_to_unet = vit_to_unet
        self.cls_head = ViTClassifierHead()

    def forward(self, x, x_vit):
        unet_bn, (c1, c2, c3, c4) = self.unet.encode(x)

        vit_tokens = self.vit(x_vit)
        cls_token, vit_patches = vit_tokens[:, 0], vit_tokens[:, 1:]

        cls_logits = self.cls_head(cls_token)

        vit_bn = self.vit_to_unet(
            vit_patches,
            unet_bn.shape[2],
            unet_bn.shape[3]
        )

        fused = unet_bn + vit_bn

        u4 = self.unet.up4(fused); d4 = self.unet.dec4(torch.cat([u4, c4], dim=1))
        u3 = self.unet.up3(d4); d3 = self.unet.dec3(torch.cat([u3, c3], dim=1))
        u2 = self.unet.up2(d3); d2 = self.unet.dec2(torch.cat([u2, c2], dim=1))
        u1 = self.unet.up1(d2); d1 = self.unet.dec1(torch.cat([u1, c1], dim=1))

        seg_out = self.unet.outc(d1)

        return seg_out, cls_logits

# --- INITIALIZATION ---

device = torch.device("cpu")

vit_model = vit_b_16(weights=None)
vit_model.load_state_dict(torch.load("vit_b_16-c867db91.pth", map_location=device))

unet = UNet(base=32)
vit_encoder = ViTEncoder(vit_model)
vit_to_unet = ViTtoUNetFusion(768, 512)

model = FeatureFusionHybrid(unet, vit_encoder, vit_to_unet)
model.load_state_dict(torch.load("fusion_unet_vit.pt", map_location=device))
model.eval()

# --- TRANSFORMS ---

unet_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

vit_transform = ViT_B_16_Weights.DEFAULT.transforms()

# --- API ---

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('image')
    if not file:
        return jsonify({"error": "No file"}), 400

    img = Image.open(io.BytesIO(file.read())).convert("L")

    xb = unet_transform(img).unsqueeze(0).to(device)
    x_vit = vit_transform(img.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        seg_logits, cls_logits = model(xb, x_vit)

        # Classification
        probs = torch.softmax(cls_logits, dim=1)
        conf, pred = torch.max(probs, 1)

        # Segmentation
        seg_mask = torch.sigmoid(seg_logits)
        seg_mask = (seg_mask > 0.5).float()

    # Convert mask → image
    mask = seg_mask.squeeze().cpu().numpy() * 255
    mask_img = Image.fromarray(mask.astype('uint8'))

    # Encode mask
    buffer = io.BytesIO()
    mask_img.save(buffer, format="PNG")
    mask_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    #  Overlay (for better visualization)
    orig = img.resize((256, 256)).convert("RGB")
    orig_np = np.array(orig)
    mask_np = mask.astype(np.uint8)

    orig_np[mask_np > 0] = [255, 0, 0]  # red tumor highlight
    overlay_img = Image.fromarray(orig_np)

    buffer2 = io.BytesIO()
    overlay_img.save(buffer2, format="PNG")
    overlay_base64 = base64.b64encode(buffer2.getvalue()).decode("utf-8")

    classes = ["Pituitary Tumor", "No Tumor"]

    return jsonify({
        "prediction": classes[pred.item()],
        "confidence": f"{conf.item() * 100:.2f}%",
        "segmentation_mask": mask_base64,
        "overlay_image": overlay_base64   # 🔥 BEST OUTPUT
    })

# --- RUN ---

if __name__ == '__main__':
    app.run(port=5000)
