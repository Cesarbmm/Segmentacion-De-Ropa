import torch
import torch.nn as nn
import numpy as np
import cv2
import segmentation_models_pytorch as smp
from PyQt6.QtCore import QThread, pyqtSignal, QObject
from PyQt6.QtGui import QImage, QColor
from PIL import Image
import time

# =============================================================================
# LOGICA DE IA (MODELO Y UTILS)
# =============================================================================
class OptimizedSegmentationModel(nn.Module):
    def __init__(self, n_cls=6):
        super().__init__()
        self.model = smp.DeepLabV3Plus(
            encoder_name="resnet34", encoder_weights=None, 
            in_channels=3, classes=n_cls, activation=None
        )
    def forward(self, x):
        return self.model(x)

def refine_mask_logic(mask_uint8):
    kernel_close = np.ones((15, 15), np.uint8) 
    closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel_close)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_filled = np.zeros_like(closed)
    for cnt in contours:
        if cv2.contourArea(cnt) > 200:
            cv2.drawContours(mask_filled, [cnt], -1, 255, thickness=cv2.FILLED)
    blur = cv2.GaussianBlur(mask_filled, (7, 7), 0)
    _, smooth_mask = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
    return smooth_mask

def apply_refinement_to_full_mask(pred_mask):
    refined_mask = np.zeros_like(pred_mask)
    unique_classes = np.unique(pred_mask)
    for cls in unique_classes:
        if cls == 0: continue
        class_mask_b = (pred_mask == cls).astype(np.uint8) * 255
        improved = refine_mask_logic(class_mask_b)
        refined_mask[improved == 255] = cls
    return refined_mask

def colorize_mask(mask):
    colors = np.array([[0,0,0], [220,20,60], [30,144,255], [255,165,0], [138,43,226], [46,139,87]])
    return colors[np.clip(mask, 0, 5)].astype(np.uint8)

def get_dominant_color(image_rgb, mask_alpha):
    valid_pixels = image_rgb[mask_alpha > 0]
    if len(valid_pixels) == 0: return "#000000", QColor(0,0,0)
    avg_color = valid_pixels.mean(axis=0).astype(int)
    r, g, b = avg_color
    return f"#{r:02x}{g:02x}{b:02x}", QColor(r, g, b)

# =============================================================================
# WORKERS (Static & RealTime)
# =============================================================================
class SegmentationWorker(QThread):
    finished = pyqtSignal(object, object, object)
    error = pyqtSignal(str)
    
    def __init__(self, model, path):
        super().__init__()
        self.model = model; self.path = path

    def run(self):
        try:
            img_pil = Image.open(self.path).convert("RGB")
            img_np = np.array(img_pil)
            h, w = img_np.shape[:2]
            
            input_size = 256
            img_resized = cv2.resize(img_np, (input_size, input_size))
            img_float = img_resized.astype(np.float32) / 255.0
            img_norm = (img_float - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float().unsqueeze(0)

            with torch.no_grad():
                output = self.model(tensor)
                pred_small = torch.argmax(output, dim=1).squeeze().numpy()

            pred_mask = cv2.resize(pred_small.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            self.finished.emit(pred_mask, img_np, img_np)
        except Exception as e:
            self.error.emit(str(e))

class VideoWorker(QThread):
    frame_processed = pyqtSignal(QImage, float) # Image, FPS
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        self._is_running = True
        self.colors = np.array([[0,0,0], [220,20,60], [30,144,255], [255,165,0], [138,43,226], [46,139,87]])
        self.class_names = ["", "Top", "Pantalon", "Vestido", "Zapatos", "Piel"]

    def run(self):
        cap = cv2.VideoCapture(0) # 0 = Webcam
        prev_time = 0
        
        while self._is_running:
            ret, frame = cap.read()
            if not ret: break
            
            orig_h, orig_w = frame.shape[:2]
            
            # Inferencia Reducida
            input_size = 256
            img_rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_img = cv2.resize(img_rgb_full, (input_size, input_size))
            
            img_float = input_img.astype(np.float32) / 255.0
            img_norm = (img_float - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float().unsqueeze(0)
            
            with torch.no_grad():
                out = self.model(tensor)
                mask_small = torch.argmax(out, dim=1).squeeze().numpy().astype(np.uint8)
            
            # Procesamiento Full
            mask_full = cv2.resize(mask_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            
            # Overlay
            colored_mask = self.colors[np.clip(mask_full, 0, 5)].astype(np.uint8)
            colored_mask_bgr = cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR)
            
            mask_bool = mask_full > 0
            overlay = frame.copy()
            alpha = 0.5
            overlay[mask_bool] = cv2.addWeighted(frame[mask_bool], 1-alpha, colored_mask_bgr[mask_bool], alpha, 0)
            
            # UI: Etiquetas y Colores
            unique_classes = np.unique(mask_full)
            
            for cls_idx in unique_classes:
                if cls_idx == 0: continue
                
                class_mask = (mask_full == cls_idx).astype(np.uint8) * 255
                contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if not contours: continue
                largest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) < 1000: continue
                
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Dominant Color
                try:
                    roi = img_rgb_full[y:y+h, x:x+w]
                    roi_mask = class_mask[y:y+h, x:x+w]
                    mean_color = cv2.mean(roi, mask=roi_mask)[:3]
                    r, g, b = int(mean_color[0]), int(mean_color[1]), int(mean_color[2])
                    hex_color = f"#{r:02X}{g:02X}{b:02X}"
                    
                    cv2.rectangle(overlay, (x, y), (x+w, y+h), (255, 255, 255), 2)
                    label = f"{self.class_names[cls_idx]} {hex_color}"
                    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    
                    cv2.rectangle(overlay, (x, y - 25), (x + text_w + 30, y), (0, 0, 0), -1)
                    cv2.rectangle(overlay, (x + 5, y - 20), (x + 20, y - 5), (b, g, r), -1)
                    cv2.putText(overlay, label, (x + 25, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                except: pass

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
            prev_time = curr_time
            
            rgb_final = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_final.shape
            qt_img = QImage(rgb_final.data, w, h, ch*w, QImage.Format.Format_RGB888)
            self.frame_processed.emit(qt_img, fps)
            
        cap.release()

    def stop(self):
        self._is_running = False
        self.wait()