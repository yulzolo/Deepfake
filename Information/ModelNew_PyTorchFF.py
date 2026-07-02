import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import dlib

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✓ Using device: {device}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.backends.cudnn.benchmark        = True


class IdentityEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net  = None
        self.dim  = 512
        self.mode = 'none'
        self._load()

    def _load(self):
        import glob as _glob

        onnx_patterns = [
            os.path.join(os.path.expanduser('~'), '.insightface', 'models', 'buffalo_l', 'w600k_r50.onnx'),
            os.path.join(os.path.expanduser('~'), '.insightface', 'models', 'buffalo_s', 'w600k_r50.onnx'),
        ]
        found_onnx = []
        for pat in onnx_patterns:
            found_onnx += _glob.glob(pat)

        if found_onnx:
            try:
                import onnxruntime as ort

                providers = (
                    ['CUDAExecutionProvider', 'CPUExecutionProvider']
                    if torch.cuda.is_available()
                    else ['CPUExecutionProvider']
                )
                sess        = ort.InferenceSession(found_onnx[0], providers=providers)
                input_name  = sess.get_inputs()[0].name
                output_name = sess.get_outputs()[0].name

                class ArcFaceONNX(nn.Module):
                    def __init__(self, session, inp, out):
                        super().__init__()
                        self._sess = session
                        self._inp  = inp
                        self._out  = out

                    def forward(self, x):
                        x112 = F.interpolate(x, size=(112, 112), mode='bilinear', align_corners=False)
                        x_np = x112.detach().cpu().numpy().astype(np.float32)
                        emb  = self._sess.run([self._out], {self._inp: x_np})[0]
                        return F.normalize(torch.from_numpy(emb).to(x.device), dim=1)

                    def parameters(self, recurse=True):
                        return iter([])

                self.net  = ArcFaceONNX(sess, input_name, output_name)
                self.dim  = 512
                self.mode = 'arcface'
                print(f"✓ IdentityEncoder: ArcFace ONNX ({os.path.basename(found_onnx[0])})")
                print(f"  Provider: {sess.get_providers()[0]}")
                return
            except ImportError:
                print("⚠ onnxruntime не найден: pip install onnxruntime-gpu")
            except Exception as e:
                print(f"⚠ ArcFace ONNX ошибка: {e}")

        try:
            from insightface.recognition.arcface_torch.backbones import get_model
            pth_patterns = [
                os.path.join(os.path.expanduser('~'), '.insightface', 'models', 'buffalo_l', 'w600k_r50.pth'),
                os.path.join(os.path.expanduser('~'), '.insightface', 'models', 'buffalo_s', 'w600k_r50.pth'),
            ]
            found_pth = []
            for pat in pth_patterns:
                found_pth += _glob.glob(pat)
            if found_pth:
                net   = get_model('r50', fp16=False)
                state = torch.load(found_pth[0], map_location='cpu', weights_only=True)
                net.load_state_dict(state)
                self.net = net.eval().to(device)
                for p in self.net.parameters():
                    p.requires_grad = False
                self.dim  = 512
                self.mode = 'arcface'
                print("✓ IdentityEncoder: ArcFace iresnet50 PyTorch (.pth)")
                return
        except Exception as e:
            print(f"⚠ ArcFace PyTorch ошибка: {e}")

        if not found_onnx:
            print("⚠ InsightFace веса не найдены в ~/.insightface/models/buffalo_l/")
            print("  Запусти ячейку 0 для скачивания buffalo_l.")
        print("  Используется VGG16 fallback — Identity Loss будет хуже!")
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.net = nn.Sequential(*list(vgg.features), nn.AdaptiveAvgPool2d(4)).eval().to(device)
        for p in self.net.parameters():
            p.requires_grad = False
        self.proj = nn.Linear(512 * 16, 512).to(device)
        self.dim  = 512
        self.mode = 'vgg16'
        print("✓ IdentityEncoder: VGG16 fallback")

    def forward(self, x):
        if self.mode == 'arcface':
            x112 = F.interpolate(x, size=(112, 112), mode='bilinear', align_corners=False)
            return F.normalize(self.net(x112), dim=1)
        else:
            mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
            std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
            x_n  = (x * 0.5 + 0.5 - mean) / std
            feat = self.net(x_n).flatten(1)
            return F.normalize(self.proj(feat), dim=1)


class AdaIN(nn.Module):
    def __init__(self, channels, style_dim=512):
        super().__init__()
        self.norm  = nn.InstanceNorm2d(channels, affine=False)
        self.style = nn.Linear(style_dim, channels * 2)
        nn.init.ones_(self.style.weight[:channels])
        nn.init.zeros_(self.style.bias)

    def forward(self, content, style_vec):
        gamma_beta = self.style(style_vec)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta  = beta.unsqueeze(-1).unsqueeze(-1)
        return self.norm(content) * (1 + gamma) + beta


class ResBlockAdaIN(nn.Module):
    def __init__(self, channels, style_dim=512):
        super().__init__()
        self.conv1  = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.conv2  = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.adain1 = AdaIN(channels, style_dim)
        self.adain2 = AdaIN(channels, style_dim)

    def forward(self, x, style):
        h = F.relu(self.adain1(self.conv1(x), style), inplace=True)
        h = self.adain2(self.conv2(h), style)
        return x + h


class FaceSwapGenerator(nn.Module):
    def __init__(self, style_dim=512):
        super().__init__()
        self.style_dim = style_dim

        def enc(ic, oc, norm=True):
            layers = [nn.Conv2d(ic, oc, 4, 2, 1, bias=False)]
            if norm:
                layers.append(nn.InstanceNorm2d(oc))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.ep0 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(32), nn.LeakyReLU(0.2, inplace=True))
        self.ep1 = enc(32,  64,  norm=False)
        self.ep2 = enc(64,  128)
        self.ep3 = enc(128, 256)
        self.ep4 = enc(256, 512)
        self.ep5 = enc(512, 512)

        self.eid = nn.Sequential(
            nn.Conv2d(3,   32,  4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32,  64,  4, 2, 1, bias=False),
            nn.InstanceNorm2d(64), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64,  128, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(512), nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.eid_proj = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(inplace=True),
            nn.Linear(512, style_dim))

        self.res_blocks = nn.ModuleList([
            ResBlockAdaIN(512, style_dim) for _ in range(6)
        ])

        def dec(ic, oc, drop=False):
            layers = [nn.ConvTranspose2d(ic, oc, 4, 2, 1, bias=False),
                      nn.InstanceNorm2d(oc), nn.ReLU(inplace=True)]
            if drop:
                layers.append(nn.Dropout(0.3))
            return nn.Sequential(*layers)

        self.d5 = dec(1024, 512, drop=True)
        self.d4 = dec(1024, 256, drop=True)
        self.d3 = dec(512,  128)
        self.d2 = dec(256,  64)
        self.d1 = dec(128,  64)
        self.d0 = nn.Sequential(
            nn.Conv2d(96, 32, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(32), nn.ReLU(inplace=True))
        self.out = nn.Sequential(nn.Conv2d(32, 3, 7, 1, 3), nn.Tanh())

    def encode_pose(self, b):
        ep0 = self.ep0(b)
        ep1 = self.ep1(ep0)
        ep2 = self.ep2(ep1)
        ep3 = self.ep3(ep2)
        ep4 = self.ep4(ep3)
        ep5 = self.ep5(ep4)
        return ep0, ep1, ep2, ep3, ep4, ep5

    def decode(self, ep0, ep1, ep2, ep3, ep4, ep5, style):
        x = ep5
        for res in self.res_blocks:
            x = res(x, style)
        d = self.d5(torch.cat([x,  ep5], 1))
        d = self.d4(torch.cat([d,  ep4], 1))
        d = self.d3(torch.cat([d,  ep3], 1))
        d = self.d2(torch.cat([d,  ep2], 1))
        d = self.d1(torch.cat([d,  ep1], 1))
        d = self.d0(torch.cat([d,  ep0], 1))
        return self.out(d)

    def forward(self, b, a):
        ep0, ep1, ep2, ep3, ep4, ep5 = self.encode_pose(b)
        style = self.eid_proj(self.eid(a))
        return self.decode(ep0, ep1, ep2, ep3, ep4, ep5, style)


class FaceSwapDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        sn = nn.utils.spectral_norm

        def make_patch(in_ch=6):
            return nn.Sequential(
                sn(nn.Conv2d(in_ch, 64,  4, 2, 1)),
                nn.LeakyReLU(0.2, inplace=True),
                sn(nn.Conv2d(64,  128, 4, 2, 1)),
                nn.InstanceNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
                sn(nn.Conv2d(128, 256, 4, 2, 1)),
                nn.InstanceNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
                sn(nn.Conv2d(256, 512, 4, 1, 1)),
                nn.InstanceNorm2d(512), nn.LeakyReLU(0.2, inplace=True),
                sn(nn.Conv2d(512,   1, 4, 1, 1)),
            )

        self.net1 = make_patch()
        self.net2 = make_patch()

    def forward(self, b, face):
        x  = torch.cat([b, face], dim=1)
        x2 = F.avg_pool2d(x, 3, stride=2, padding=1)
        return self.net1(x), self.net2(x2)


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.slice1 = nn.Sequential(*list(vgg)[:4]).eval()
        self.slice2 = nn.Sequential(*list(vgg)[4:9]).eval()
        self.slice3 = nn.Sequential(*list(vgg)[9:18]).eval()
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x, y):
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        xn = (x * 0.5 + 0.5 - mean) / std
        yn = (y * 0.5 + 0.5 - mean) / std
        loss = 0.0
        for sl in [self.slice1, self.slice2, self.slice3]:
            xn = sl(xn)
            yn = sl(yn)
            loss = loss + F.l1_loss(xn, yn)
        return loss


class FaceDetector:
    def __init__(self):
        self.detector  = dlib.get_frontal_face_detector()
        self.predictor = None
        for path in [
            'shape_predictor_68_face_landmarks.dat',
            os.path.join('.', 'shape_predictor_68_face_landmarks.dat'),
        ]:
            if os.path.exists(path):
                try:
                    self.predictor = dlib.shape_predictor(path)
                    print(f"✓ Predictor loaded: {path}")
                    return
                except Exception as e:
                    print(f"✗ {e}")
        print("⚠ shape_predictor_68_face_landmarks.dat not found")

    def detect_face(self, image):
        gray  = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = self.detector(gray, 1)
        return faces[0] if faces else dlib.rectangle(0, 0, image.shape[1], image.shape[0])

    def get_landmarks(self, image, face_rect):
        if self.predictor is None or face_rect.is_empty():
            return None
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        try:
            shape = self.predictor(gray, face_rect)
            return np.array([[p.x, p.y] for p in shape.parts()])
        except Exception:
            return None

    def align_face(self, image, landmarks, size=256):
        h, w = image.shape[:2]
        if landmarks is not None and len(landmarks) == 68:
            le    = landmarks[36:42].mean(0).astype(np.float32)
            re    = landmarks[42:48].mean(0).astype(np.float32)
            dy, dx = re[1] - le[1], re[0] - le[0]
            angle  = np.degrees(np.arctan2(dy, dx))
            ec     = ((le[0] + re[0]) / 2, (le[1] + re[1]) / 2)
            M      = np.vstack([cv2.getRotationMatrix2D(ec, angle, 1.0), [0, 0, 1]])
            rotated = cv2.warpAffine(image, M[:2], (w, h), flags=cv2.INTER_CUBIC)
            dist   = np.sqrt(dx ** 2 + dy ** 2)
            scale  = 0.65 * size / (dist + 1e-6)
            ecr    = (M[0, 0] * ec[0] + M[0, 1] * ec[1] + M[0, 2],
                      M[1, 0] * ec[0] + M[1, 1] * ec[1] + M[1, 2])
            cx, cy = int(ecr[0] - size * 0.5 * scale), int(ecr[1] - size * 0.5 * scale)
            crop   = rotated[max(0, cy):cy + int(size * scale),
                             max(0, cx):cx + int(size * scale)]
            if crop.shape[0] < 4 or crop.shape[1] < 4:
                crop = cv2.resize(rotated, (size, size))
            return cv2.resize(crop, (size, size)), ec, angle, scale
        else:
            sh   = max(0, h - size) // 2
            sw   = max(0, w - size) // 2
            crop = image[sh:sh + size, sw:sw + size]
            return (cv2.resize(crop if crop.shape[0] == size else image, (size, size)),
                    None, 0, 1.0)

    def face_quality_score(self, image, landmarks):
        if landmarks is None:
            return False
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return gray.mean() >= 10


class FaceDatasetOptimized(Dataset):
    def __init__(self, faces_B, faces_A, masks, samples, augment=True):
        self.faces_B = faces_B
        self.faces_A = faces_A
        self.masks   = masks
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ai, bi = self.samples[idx]
        B    = torch.from_numpy(self.faces_B[bi].transpose(2, 0, 1)).float()
        A    = torch.from_numpy(self.faces_A[ai].transpose(2, 0, 1)).float()
        mask = torch.from_numpy(self.masks[ai]).float().unsqueeze(0)

        if self.augment:
            if torch.rand(1) > 0.5:
                B    = torch.flip(B, [-1])
                mask = torch.flip(mask, [-1])

            if torch.rand(1) > 0.5:
                B = (B * (1.0 + (torch.rand(1) - 0.5) * 0.3)).clamp(-1, 1)

            if torch.rand(1) > 0.5:
                B = (B + (torch.rand(3) - 0.5).view(3, 1, 1) * 0.15).clamp(-1, 1)

            if torch.rand(1) > 0.5:
                angle = (torch.rand(1).item() - 0.5) * 20.0
                rad   = angle * np.pi / 180
                theta = torch.tensor([
                    [np.cos(rad), -np.sin(rad), 0],
                    [np.sin(rad),  np.cos(rad), 0],
                ], dtype=torch.float32).unsqueeze(0)
                grid = F.affine_grid(theta, B.unsqueeze(0).shape, align_corners=False)
                B    = F.grid_sample(B.unsqueeze(0),    grid, align_corners=False).squeeze(0)
                mask = F.grid_sample(mask.unsqueeze(0), grid, align_corners=False).squeeze(0)

            if torch.rand(1) > 0.8:
                k = torch.ones(1, 1, 3, 3) / 9.0
                B = torch.cat([
                    F.conv2d(B[c:c+1].unsqueeze(0), k, padding=1).squeeze(0)
                    for c in range(3)
                ], dim=0).clamp(-1, 1)

        return B, A, mask


class FaceSwapModel:
    def __init__(self, img_size=256,
                 id_lambda=8.0,
                 perceptual_lambda=1.0,
                 self_recon_lambda=5.0,
                 l1_lambda=0.5,
                 pose_lambda=1.0,
                 **kwargs):
        self.img_size          = img_size
        self.id_lambda         = id_lambda
        self.perceptual_lambda = perceptual_lambda
        self.self_recon_lambda = self_recon_lambda
        self.l1_lambda         = l1_lambda
        self.pose_lambda       = pose_lambda
        self.adv_lambda        = 0.0
        self.device            = device
        self.face_detector     = FaceDetector()
        self.history           = {'g_loss': [], 'd_loss': [], 'id': [], 'sr': [], 'l1': [], 'pose': []}
        self.faces_A           = []
        self.faces_B           = []
        self.masks             = []
        self.paired_dataset    = {}
        self.generator         = None
        self.discriminator     = None
        self.id_encoder        = None
        self.best_score        = float('inf')
        self._last_batch       = None
        print(f"✓ FaceSwapGAN v13 (Dual Encoder + AdaIN + Multi-scale D)")
        print(f"  id_λ={id_lambda}, perc_λ={perceptual_lambda}, "
              f"sr_λ={self_recon_lambda}, l1_λ={l1_lambda}, pose_λ={pose_lambda}")

    def build_gan(self, compile_model=False):
        self.generator     = FaceSwapGenerator(style_dim=512).to(self.device)
        self.discriminator = FaceSwapDiscriminator().to(self.device)
        self.id_encoder    = IdentityEncoder().to(self.device)

        if compile_model and hasattr(torch, 'compile'):
            self.generator = torch.compile(self.generator, mode='reduce-overhead')

        self.gen_optimizer  = Adam(self.generator.parameters(),     lr=1e-4, betas=(0.5, 0.999))
        self.disc_optimizer = Adam(self.discriminator.parameters(), lr=4e-4, betas=(0.5, 0.999))
        self.g_scaler  = GradScaler()
        self.d_scaler  = GradScaler()
        self.perc_loss = PerceptualLoss().to(self.device)

        params_g = sum(p.numel() for p in self.generator.parameters()) / 1e6
        params_d = sum(p.numel() for p in self.discriminator.parameters()) / 1e6
        print(f"✓ Generator:      {params_g:.1f}M params")
        print(f"✓ Discriminator:  {params_d:.1f}M params")
        print(f"✓ Identity mode:  {self.id_encoder.mode}")

    def load_paired_data(self, photo_paths, video_paths, max_frames_per_video=100):
        self.faces_A, self.faces_B, self.masks = [], [], []
        self.paired_dataset = {}
        skipped    = []
        total_ok   = 0
        total_filt = 0

        for i in range(min(len(photo_paths), len(video_paths))):
            ppath, vpath = photo_paths[i], video_paths[i]
            pair_id      = f"p{i+1}v{i+1}"
            img_bgr = cv2.imread(ppath)
            if img_bgr is None:
                skipped.append(pair_id)
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            rect    = self.face_detector.detect_face(img_rgb)
            lm      = self.face_detector.get_landmarks(img_rgb, rect)
            face_A, _, _, _ = self.face_detector.align_face(img_rgb, lm, self.img_size)
            mask_A = self._make_face_mask(lm)
            self.faces_A.append(face_A.astype(np.float32) / 127.5 - 1.0)
            self.masks.append(mask_A)
            self.paired_dataset[pair_id] = {'A': len(self.faces_A) - 1, 'B': {}}

            vfaces, nf = self._extract_faces_from_video(vpath, max_frames_per_video)
            total_filt += nf
            if vfaces:
                total_ok += len(vfaces)
                start     = len(self.faces_B)
                self.faces_B.extend(vfaces)
                self.paired_dataset[pair_id]['B'] = {'start': start, 'count': len(vfaces)}
                print(f"✓ {pair_id}: {os.path.basename(ppath)} | +{len(vfaces)} frames (отброшено: {nf})")
            else:
                print(f"⚠ {pair_id}: нет кадров после фильтрации")
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(photo_paths)} пар обработано...")

        self.faces_A = np.array(self.faces_A)
        self.faces_B = np.array(self.faces_B)
        self.masks   = np.array(self.masks)
        total = total_ok + total_filt
        print(f"\n=== ИТОГ ===")
        print(f"✓ Пар: {len(self.paired_dataset)}, A: {len(self.faces_A)}, B: {len(self.faces_B)}")
        print(f"  Принято: {total_ok}, Отброшено: {total_filt} ({100*total_filt/(total+1e-6):.1f}%)")
        if skipped:
            print(f"⚠ Пропущено пар: {skipped}")

    def _make_face_mask(self, landmarks):
        s    = self.img_size
        mask = np.zeros((s, s), dtype=np.uint8)
        if landmarks is not None:
            lift = int((landmarks[0:17][:, 1].max() - landmarks[17:27][:, 1].min()) * 0.5)
            fh   = [[int(pt[0]), int(pt[1]) - lift] for pt in landmarks[17:27]]
            pts  = [[int(p[0]), int(p[1])] for p in landmarks[0:17]] + fh[::-1]
            cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)
        else:
            cv2.ellipse(mask, (s // 2, s // 2), (s // 3, int(s * 0.55)), 0, 0, 360, 255, -1)
        return cv2.GaussianBlur(mask, (51, 51), 0)

    def _extract_faces_from_video(self, video_path, max_frames=100):
        cap     = cv2.VideoCapture(video_path)
        total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 10000
        indices = set(np.linspace(0, total - 1, min(max_frames * 3, total), dtype=int).tolist())
        frames, n_filtered, fc = [], 0, 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if fc in indices and len(frames) < max_frames:
                rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rect = self.face_detector.detect_face(rgb)
                lm   = self.face_detector.get_landmarks(rgb, rect)
                if self.face_detector.face_quality_score(rgb, lm):
                    aligned, _, _, _ = self.face_detector.align_face(rgb, lm, self.img_size)
                    frames.append(aligned.astype(np.float32) / 127.5 - 1.0)
                else:
                    n_filtered += 1
            fc += 1
        cap.release()
        return frames, n_filtered

    def create_dataloader(self, batch_size=8, shuffle=True, augment=True):
        samples = []
        for pid, info in self.paired_dataset.items():
            if not info['B']:
                continue
            ai  = info['A']
            bs  = info['B']['start']
            cnt = info['B']['count']
            for bi in range(bs, bs + cnt):
                samples.append((ai, bi))
        print(f"Dataset samples: {len(samples)}")
        ds = FaceDatasetOptimized(
            faces_B=self.faces_B, faces_A=self.faces_A,
            masks=self.masks, samples=samples, augment=augment)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=0, pin_memory=False,
                          drop_last=True, persistent_workers=False)

    def train(self, epochs=30, batch_size=8, save_interval=5, adv_lambda=None, **kwargs):
        if adv_lambda is not None:
            self.adv_lambda = adv_lambda
        use_adv = self.adv_lambda > 0

        print(f"⚙ v13: epochs={epochs}, batch={batch_size}, "
              f"adv_λ={self.adv_lambda}, id_λ={self.id_lambda}, "
              f"sr_λ={self.self_recon_lambda}, l1_λ={self.l1_lambda}, "
              f"pose_λ={self.pose_lambda}, GAN={'ON' if use_adv else 'OFF'}")

        dataloader = self.create_dataloader(batch_size, augment=True)
        g_sched = CosineAnnealingLR(self.gen_optimizer,  T_max=max(epochs, 1), eta_min=1e-5)
        d_sched = CosineAnnealingLR(self.disc_optimizer, T_max=max(epochs, 1), eta_min=5e-5)

        for epoch in range(1, epochs + 1):
            g_losses, d_losses, id_losses = [], [], []
            sr_losses, l1_losses, pose_losses = [], [], []

            pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}", leave=False)

            for batch_B, batch_A, batch_masks in pbar:
                batch_B     = batch_B.to(self.device)
                batch_A     = batch_A.to(self.device)
                batch_masks = batch_masks.to(self.device)

                if use_adv:
                    self.disc_optimizer.zero_grad()
                    with autocast('cuda'):
                        with torch.no_grad():
                            fake_d = self.generator(batch_B, batch_A)
                        d_real1, d_real2 = self.discriminator(batch_B, batch_A)
                        d_fake1, d_fake2 = self.discriminator(batch_B, fake_d)
                        d_loss = (
                            (F.relu(0.9 - d_real1).mean() + F.relu(1.0 + d_fake1).mean()) * 0.5 +
                            (F.relu(0.9 - d_real2).mean() + F.relu(1.0 + d_fake2).mean()) * 0.5
                        ) * 0.5
                    self.d_scaler.scale(d_loss).backward()
                    self.d_scaler.unscale_(self.disc_optimizer)
                    nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
                    self.d_scaler.step(self.disc_optimizer)
                    self.d_scaler.update()
                    d_losses.append(d_loss.item())
                else:
                    d_losses.append(0.0)

                self.gen_optimizer.zero_grad()
                with autocast('cuda'):
                    ep0_b, ep1_b, ep2_b, ep3_b, ep4_b, ep5_b = self.generator.encode_pose(batch_B)
                    style_a    = self.generator.eid_proj(self.generator.eid(batch_A))
                    fake_cross = self.generator.decode(ep0_b, ep1_b, ep2_b, ep3_b, ep4_b, ep5_b, style_a)

                    id_loss = self.perc_loss(fake_cross, batch_A)
                    with torch.no_grad():
                        emb_fake_arc = self.id_encoder(fake_cross)
                        emb_real_arc = self.id_encoder(batch_A)
                    arc_sim = F.cosine_similarity(emb_fake_arc, emb_real_arc).mean().item()

                    perc_loss = self.perc_loss(fake_cross, batch_B)

                    mask_norm = batch_masks.mean() + 1e-8
                    l1_loss   = (batch_masks * (fake_cross - batch_B).abs()).mean() / mask_norm

                    style_a2  = self.generator.eid_proj(self.generator.eid(batch_A))
                    ep0_a, ep1_a, ep2_a, ep3_a, ep4_a, ep5_a = self.generator.encode_pose(batch_A)
                    fake_self = self.generator.decode(ep0_a, ep1_a, ep2_a, ep3_a, ep4_a, ep5_a, style_a2)
                    sr_loss   = F.l1_loss(fake_self, batch_A)

                    ep0_f, ep1_f, ep2_f, ep3_f, ep4_f, ep5_f = self.generator.encode_pose(fake_cross)
                    pose_loss = F.l1_loss(ep5_f, ep5_b.detach())

                    perc_fake_b      = self.perc_loss(fake_cross, batch_B)
                    anticollapse_loss = F.relu(0.05 - perc_fake_b)

                    if use_adv:
                        adv1, adv2 = self.discriminator(batch_B, fake_cross)
                        g_adv = (-adv1.mean() - adv2.mean()) * 0.5 * self.adv_lambda
                    else:
                        g_adv = torch.tensor(0.0, device=self.device)

                    g_total = (
                        id_loss           * self.id_lambda +
                        perc_loss         * self.perceptual_lambda +
                        l1_loss           * self.l1_lambda +
                        sr_loss           * self.self_recon_lambda +
                        pose_loss         * self.pose_lambda +
                        anticollapse_loss * 5.0 +
                        g_adv
                    )

                if torch.isnan(g_total) or torch.isinf(g_total):
                    self.gen_optimizer.zero_grad()
                    continue

                self.g_scaler.scale(g_total).backward()
                self.g_scaler.unscale_(self.gen_optimizer)
                nn.utils.clip_grad_norm_(self.generator.parameters(), 0.5)
                self.g_scaler.step(self.gen_optimizer)
                self.g_scaler.update()

                g_losses.append(g_total.item())
                id_losses.append(id_loss.item())
                sr_losses.append(sr_loss.item())
                l1_losses.append(l1_loss.item())
                pose_losses.append(pose_loss.item())

                pbar.set_postfix(
                    D=f"{np.mean(d_losses):.3f}",
                    ID=f"{np.mean(id_losses):.3f}",
                    ArcSim=f"{arc_sim:.3f}",
                    SR=f"{np.mean(sr_losses):.4f}",
                    L1=f"{np.mean(l1_losses):.4f}")

                self._last_batch = (batch_B[:4].detach().clone(), batch_A[:4].detach().clone())

            g_sched.step()
            if use_adv:
                d_sched.step()

            avg_g    = np.mean(g_losses)
            avg_d    = np.mean(d_losses)
            avg_id   = np.mean(id_losses)
            avg_sr   = np.mean(sr_losses)
            avg_l1   = np.mean(l1_losses)
            avg_pose = np.mean(pose_losses)

            self.history['g_loss'].append(avg_g)
            self.history['d_loss'].append(avg_d)
            self.history['id'].append(avg_id)
            self.history['sr'].append(avg_sr)
            self.history['l1'].append(avg_l1)
            self.history['pose'].append(avg_pose)

            print(f"✅ Epoch {epoch:3d} | D:{avg_d:.3f} | ID:{avg_id:.3f} | "
                  f"SR:{avg_sr:.4f} | L1:{avg_l1:.4f} | Pose:{avg_pose:.4f}")

            if epoch == 5 and avg_sr > 0.08:
                print(f"  ⚠ SR={avg_sr:.4f} высокий после 5 эпох")
            if epoch >= 10 and avg_sr < 0.005:
                print(f"  ✓ SR={avg_sr:.4f} — отличная self-reconstruction")

            score = avg_id + avg_sr * 0.5
            if score < self.best_score:
                self.best_score = score
                os.makedirs(os.path.join('output', 'trained_models'), exist_ok=True)
                torch.save(
                    self.generator.state_dict(),
                    os.path.join('output', 'trained_models', 'generator_best.pth'))
                print(f"  ⭐ Best score={self.best_score:.4f} (id={avg_id:.4f}, sr={avg_sr:.4f})")

            if epoch % save_interval == 0:
                self.save_progress(epoch)
                if self._last_batch is not None:
                    self.generate_progress_grid_from_batch(epoch, *self._last_batch)

        print("🎉 Training completed!")
        return self.history

    def swap_face_in_video_PATCHED(self, source_photo_path, target_video_path, output_path):
        print(f"\n✨ Face Swap v13")
        self.generator.eval()

        src_bgr  = cv2.imread(source_photo_path)
        if src_bgr is None:
            raise FileNotFoundError(f"Не найдено: {source_photo_path}")
        src_rgb  = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB)
        src_rect = self.face_detector.detect_face(src_rgb)
        src_lm   = self.face_detector.get_landmarks(src_rgb, src_rect)
        face_A, _, _, _ = self.face_detector.align_face(src_rgb, src_lm, self.img_size)
        face_A_t = (torch.from_numpy(face_A.astype(np.float32) / 127.5 - 1.0)
                    .permute(2, 0, 1).unsqueeze(0).to(self.device))
        print(f"✓ Source face aligned")

        cap    = cv2.VideoCapture(target_video_path)
        fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        print(f"Processing {total} frames...")

        with torch.no_grad():
            idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rect = self.face_detector.detect_face(rgb)
                lm   = self.face_detector.get_landmarks(rgb, rect)

                if lm is not None:
                    x1, y1 = rect.left(),  rect.top()
                    x2, y2 = rect.right(), rect.bottom()
                    pad  = int(max(x2 - x1, y2 - y1) * 0.2)
                    y1s  = max(0,      y1 - pad)
                    y2s  = min(height, y2 + pad)
                    x1s  = max(0,      x1 - pad)
                    x2s  = min(width,  x2 + pad)
                    rh, rw = y2s - y1s, x2s - x1s
                    roi    = rgb[y1s:y2s, x1s:x2s]

                    if roi.shape[0] > 4 and roi.shape[1] > 4:
                        face_B   = cv2.resize(roi, (self.img_size, self.img_size))
                        face_B_t = (torch.from_numpy(face_B.astype(np.float32) / 127.5 - 1.0)
                                    .permute(2, 0, 1).unsqueeze(0).to(self.device))

                        fake     = self.generator(face_B_t, face_A_t)
                        fake_rgb = ((fake[0].permute(1, 2, 0).cpu().numpy() + 1) * 127.5
                                    ).clip(0, 255).astype(np.uint8)

                        flm = lm.copy().astype(np.float32)
                        flm[:, 0] = np.clip(flm[:, 0] - x1s, 0, rw - 1)
                        flm[:, 1] = np.clip(flm[:, 1] - y1s, 0, rh - 1)

                        mask_roi = np.zeros((rh, rw), dtype=np.uint8)
                        hull     = cv2.convexHull(flm.astype(np.int32))
                        cv2.fillConvexPoly(mask_roi, hull, 255)
                        mask_roi = cv2.erode(mask_roi, np.ones((5, 5), np.uint8), 1)

                        resized_fake = cv2.cvtColor(cv2.resize(fake_rgb, (rw, rh)), cv2.COLOR_RGB2BGR)

                        m_bool = mask_roi > 128
                        if m_bool.sum() > 100:
                            roi_orig_f = frame[y1s:y2s, x1s:x2s].astype(np.float32)
                            fake_f     = resized_fake.astype(np.float32)
                            for c in range(3):
                                src_mean = roi_orig_f[:, :, c][m_bool].mean()
                                src_std  = roi_orig_f[:, :, c][m_bool].std() + 1e-6
                                dst_mean = fake_f[:, :, c][m_bool].mean()
                                dst_std  = fake_f[:, :, c][m_bool].std() + 1e-6
                                fake_f[:, :, c] = (
                                    (fake_f[:, :, c] - dst_mean) / dst_std * src_std + src_mean)
                            resized_fake = fake_f.clip(0, 255).astype(np.uint8)

                        moments = cv2.moments(mask_roi)
                        if moments['m00'] > 0:
                            cx = int(moments['m10'] / moments['m00'])
                            cy = int(moments['m01'] / moments['m00'])
                            center = (cx, cy)
                        else:
                            center = (rw // 2, rh // 2)

                        roi_bgr = frame[y1s:y2s, x1s:x2s].copy()
                        try:
                            blended_roi = cv2.seamlessClone(
                                resized_fake, roi_bgr, mask_roi, center, cv2.MIXED_CLONE)
                            frame[y1s:y2s, x1s:x2s] = blended_roi
                        except cv2.error:
                            mask_blur = cv2.GaussianBlur(mask_roi, (21, 21), 0)
                            m3 = np.stack([mask_blur / 255.0] * 3, -1)
                            blended = (roi_bgr.astype(np.float32) * (1 - m3)
                                       + resized_fake.astype(np.float32) * m3)
                            frame[y1s:y2s, x1s:x2s] = blended.clip(0, 255).astype(np.uint8)

                writer.write(frame)
                idx += 1
                if idx % 50 == 0:
                    print(f"  [{idx}/{total}]")

        cap.release()
        writer.release()
        self.generator.train()
        print(f"✓ Saved: {output_path}")
        return output_path

    def generate_progress_grid_from_batch(self, epoch, batch_B, batch_A):
        self.generator.eval()
        os.makedirs(os.path.join('output', 'progress'), exist_ok=True)
        with torch.no_grad():
            bB    = batch_B.to(self.device)
            bA    = batch_A.to(self.device)
            bA_sh = bA[torch.randperm(bA.shape[0])]

            fake_self  = self.generator(bA, bA)
            fake_cross = self.generator(bB, bA)
            fake_diff  = self.generator(bB, bA_sh)

            def to_img(t):
                return ((t.permute(1, 2, 0).cpu().numpy() + 1) / 2).clip(0, 1)

            n    = min(4, bB.shape[0])
            fig, axes = plt.subplots(5, n, figsize=(4 * n, 20))
            if n == 1:
                axes = axes[:, None]
            rows = [
                (bB,         'B (video — поза)'),
                (bA,         'A (photo — лицо)'),
                (fake_self,  'G(A,A) — self recon'),
                (fake_cross, 'G(B,A) — SWAP'),
                (fake_diff,  'G(B,A_other)'),
            ]
            for row, (imgs, title) in enumerate(rows):
                for i in range(n):
                    axes[row, i].imshow(to_img(imgs[i]))
                    axes[row, i].set_title(f'{title} #{i+1}', fontsize=8)
                    axes[row, i].axis('off')
            plt.suptitle(f'Epoch {epoch} — v13', fontsize=10)
            plt.tight_layout()
            plt.savefig(
                os.path.join('output', 'progress', f'progress_epoch{epoch:03d}.jpg'), dpi=100)
            plt.close()
            print(f"✓ Progress grid: epoch {epoch}")
        self.generator.train()

    def save_progress(self, epoch):
        os.makedirs(os.path.join('output', 'trained_models'), exist_ok=True)
        torch.save(
            self.generator.state_dict(),
            os.path.join('output', 'trained_models', f'generator_epoch{epoch:03d}.pth'))
        print(f"✓ Saved weights epoch {epoch}")

    def save_model(self):
        os.makedirs(os.path.join('output', 'trained_models'), exist_ok=True)
        torch.save(
            self.generator.state_dict(),
            os.path.join('output', 'trained_models', 'generator_final.pth'))
        np.save(
            os.path.join('output', 'trained_models', 'training_history.npy'),
            self.history)
        print("✓ Model saved!")

    def load_model(self, weights_path):
        if self.generator is None:
            self.build_gan()
        self.generator.load_state_dict(
            torch.load(weights_path, map_location=self.device, weights_only=True))
        self.generator.eval()
        print(f"✓ Loaded: {weights_path}")

    def plot_training_history(self):
        fig, axes = plt.subplots(1, 5, figsize=(28, 5))
        ep = range(1, len(self.history['g_loss']) + 1)
        axes[0].plot(ep, self.history['g_loss'], color='blue',   label='G total')
        axes[0].plot(ep, self.history['d_loss'], color='red',    label='D loss')
        axes[0].set_title('GAN Loss');        axes[0].legend(); axes[0].grid(True, alpha=0.3)
        axes[1].plot(ep, self.history['id'],   color='green',   label='Identity')
        axes[1].set_title('Identity Loss');   axes[1].legend(); axes[1].grid(True, alpha=0.3)
        axes[2].plot(ep, self.history['sr'],   color='purple',  label='Self-recon')
        axes[2].set_title('Self-recon');      axes[2].legend(); axes[2].grid(True, alpha=0.3)
        axes[3].plot(ep, self.history['l1'],   color='orange',  label='L1 masked')
        axes[3].set_title('L1 masked');       axes[3].legend(); axes[3].grid(True, alpha=0.3)
        axes[4].plot(ep, self.history['pose'], color='brown',   label='Pose')
        axes[4].set_title('Pose consistency'); axes[4].legend(); axes[4].grid(True, alpha=0.3)
        plt.suptitle('FaceSwapGAN v13 — Dual Encoder + AdaIN + Multi-scale D')
        plt.tight_layout()
        os.makedirs('output', exist_ok=True)
        plt.savefig(os.path.join('output', 'training_history.png'), dpi=150)
        plt.close()
        print("✓ output/training_history.png")
