import torch
import torch.nn as nn
import numpy as np
import cv2
import segmentation_models_pytorch as smp
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage, QColor
from PIL import Image
import time

# ----- INTENTO DE IMPORTAR YOLO -----
try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False
    print("⚠️ Ultralytics no instalado. El modo multipersona fallará.")

# =============================================================================
# LOGICA DE IA (MODELO)
# =============================================================================
class OptimizedSegmentationModel(nn.Module):
    def __init__(self, n_cls=6):
        super().__init__()
        self.model = smp.DeepLabV3Plus(
            encoder_name="resnet34", 
            encoder_weights=None, 
            in_channels=3, 
            classes=n_cls, 
            activation=None
        )
        
    def forward(self, x):
        return self.model(x)

# Funciones auxiliares
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
# WORKER ESTÁTICO (Unchanged)
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

            device = next(self.model.parameters()).device
            tensor = tensor.to(device)

            with torch.no_grad():
                output = self.model(tensor)
                pred_small = torch.argmax(output, dim=1).squeeze().cpu().numpy()

            pred_mask = cv2.resize(pred_small.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            self.finished.emit(pred_mask, img_np, img_np)
        except Exception as e:
            self.error.emit(str(e))

# =============================================================================
# WORKER EN VIVO (SINGLE PERSON - DISEÑO PRO HUD)
# =============================================================================
class VideoWorker(QThread):
    frame_processed = pyqtSignal(QImage, float)
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        self._is_running = True
        
        # Configuración
        self.skip_frames = 3 
        self.frame_count = 0
        self.class_names = ["", "TOP", "PANTALON", "VESTIDO", "ZAPATOS", "PIEL"] # Mayúsculas para estilo técnico
        self.mask_colors = np.array([[0,0,0], [220,20,60], [30,144,255], [255,165,0], [138,43,226], [46,139,87]])
        
        # Memoria
        self.last_mask = None

    def draw_pro_hud(self, img, bbox, label, color_bgr):
        """Dibuja una interfaz High-Tech sobre el objeto"""
        x, y, w, h = bbox
        
        # Colores de interfaz
        border_color = (255, 255, 255) # Blanco para bordes técnicos
        bg_card_color = (10, 10, 10)   # Negro casi puro
        
        # 1. ESQUINAS TÉCNICAS (Bracket Style)
        l = int(min(w, h) * 0.2) # Longitud de la esquina
        th = 2 # Grosor
        
        # Top-Left
        cv2.line(img, (x, y), (x + l, y), color_bgr, th)
        cv2.line(img, (x, y), (x, y + l), color_bgr, th)
        # Bottom-Right
        cv2.line(img, (x+w, y+h), (x+w - l, y+h), color_bgr, th)
        cv2.line(img, (x+w, y+h), (x+w, y+h - l), color_bgr, th)

        # 2. TARJETA DE DATOS (Floating Card)
        # Posición: A la derecha del objeto, o izquierda si no cabe
        card_w, card_h = 160, 50
        card_x = x + w + 15
        card_y = y + (h // 2) - (card_h // 2)
        
        # Ajuste si se sale de la pantalla
        if card_x + card_w > img.shape[1]:
            card_x = x - card_w - 15

        # Dibujar fondo semitransparente (Simulación manual para velocidad)
        # Usamos rectángulo sólido negro primero, luego dibujamos cosas encima
        # Para transparencia real se requiere overlay + addWeighted, pero consume CPU. 
        # Hacemos un diseño sólido limpio.
        cv2.rectangle(img, (card_x, card_y), (card_x + card_w, card_y + card_h), bg_card_color, -1)
        
        # Borde fino de la tarjeta
        cv2.rectangle(img, (card_x, card_y), (card_x + card_w, card_y + card_h), (50, 50, 50), 1)

        # 3. BARRA DE COLOR (DNA Bar)
        # Una franja vertical a la izquierda de la tarjeta con el color detectado
        bar_w = 6
        cv2.rectangle(img, (card_x, card_y), (card_x + bar_w, card_y + card_h), color_bgr, -1)

        # 4. TEXTO TÉCNICO
        # Nombre de la Clase
        cv2.putText(img, label, (card_x + 15, card_y + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Código Hexadecimal
        r, g, b = color_bgr[2], color_bgr[1], color_bgr[0]
        hex_code = f"HEX: #{r:02X}{g:02X}{b:02X}"
        cv2.putText(img, hex_code, (card_x + 15, card_y + 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)

        # 5. LINEA CONECTORA (Leader Line)
        # Desde el centro del borde del bounding box hasta la tarjeta
        if card_x > x: # Tarjeta a la derecha
            cv2.line(img, (x+w, y+h//2), (card_x, y+h//2), color_bgr, 1)
            cv2.circle(img, (x+w, y+h//2), 3, color_bgr, -1)
        else: # Tarjeta a la izquierda
            cv2.line(img, (x, y+h//2), (card_x + card_w, y+h//2), color_bgr, 1)
            cv2.circle(img, (x, y+h//2), 3, color_bgr, -1)

    def run(self):
        cap = cv2.VideoCapture(0)
        prev_time = 0
        device = next(self.model.parameters()).device

        while self._is_running:
            ret, frame = cap.read()
            if not ret: break
            
            orig_h, orig_w = frame.shape[:2]
            img_rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Copia para dibujar (Overlay)
            display_frame = frame.copy()

            # --- 1. LÓGICA DE INFERENCIA ---
            if self.frame_count % self.skip_frames == 0:
                input_size = 256
                input_img = cv2.resize(img_rgb_full, (input_size, input_size))
                img_float = input_img.astype(np.float32) / 255.0
                img_norm = (img_float - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
                tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
                
                with torch.no_grad():
                    out = self.model(tensor)
                    mask_small = torch.argmax(out, dim=1).squeeze().cpu().numpy().astype(np.uint8)
                
                mask_full = cv2.resize(mask_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                self.last_mask = mask_full
            
            current_mask = self.last_mask if self.last_mask is not None else np.zeros((orig_h, orig_w), dtype=np.uint8)

            # --- 2. PROCESAMIENTO VISUAL ---
            if np.any(current_mask > 0):
                # A. Máscara sutil sobre la ropa (Alpha Blending)
                colored_mask = self.mask_colors[np.clip(current_mask, 0, 5)].astype(np.uint8)
                colored_mask_bgr = cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR)
                mask_bool = current_mask > 0
                
                # Efecto sutil: 30% color máscara, 70% imagen original
                alpha = 0.3 
                display_frame[mask_bool] = cv2.addWeighted(frame[mask_bool], 1-alpha, colored_mask_bgr[mask_bool], alpha, 0)
                
                # B. Elementos UI PRO (Iterar objetos)
                unique_classes = np.unique(current_mask)
                
                for cls_idx in unique_classes:
                    if cls_idx == 0: continue 
                    
                    # Crear máscara binaria para esta clase
                    class_mask_bin = (current_mask == cls_idx).astype(np.uint8) * 255
                    
                    # Encontrar contornos
                    contours, _ = cv2.findContours(class_mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if contours:
                        # Tomar solo el componente más grande (para evitar ruido)
                        largest_cnt = max(contours, key=cv2.contourArea)
                        
                        if cv2.contourArea(largest_cnt) > 800: # Filtro de ruido
                            # Obtener Bounding Box
                            x, y, w, h = cv2.boundingRect(largest_cnt)
                            
                            # Obtener color promedio REAL (extraído de la imagen original, no de la máscara)
                            roi = img_rgb_full[y:y+h, x:x+w]
                            roi_mask = class_mask_bin[y:y+h, x:x+w]
                            
                            if np.sum(roi_mask) > 0:
                                mean_color = cv2.mean(roi, mask=roi_mask)[:3]
                                r, g, b = int(mean_color[0]), int(mean_color[1]), int(mean_color[2])
                                detected_color_bgr = (b, g, r) # OpenCV usa BGR
                                
                                # --- LLAMADA A LA NUEVA FUNCIÓN PRO ---
                                self.draw_pro_hud(display_frame, (x, y, w, h), self.class_names[cls_idx], detected_color_bgr)

            # --- 3. SALIDA ---
            self.frame_count += 1
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
            prev_time = curr_time
            
            rgb_final = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            h_out, w_out, ch = rgb_final.shape
            qt_img = QImage(rgb_final.data, w_out, h_out, ch*w_out, QImage.Format.Format_RGB888)
            
            self.frame_processed.emit(qt_img, fps)
            
        cap.release()

    def stop(self):
        self._is_running = False
        self.wait()

# =============================================================================
# WORKER MULTI-PERSONA (Unchanged logic, just kept structure)
# =============================================================================
class MultiPersonWorker(QThread):
    frame_processed = pyqtSignal(QImage, float)
    
    def __init__(self, clothing_model):
        super().__init__()
        self.clothing_model = clothing_model 
        self._is_running = True
        
        if HAS_YOLO:
            self.yolo = YOLO('yolov8n.pt') 
        
        self.MAX_PEOPLE = 3      
        self.CONF_THRESHOLD = 0.5 
        self.SKIP_FRAMES = 2     
        self.frame_count = 0
        self.class_names = ["", "Top", "Pantalon", "Vestido", "Zapatos", "Piel"]
        self.mask_colors = np.array([[0,0,0], [220,20,60], [30,144,255], [255,165,0], [138,43,226], [46,139,87]])
        self.last_overlay = None 

    def run(self):
        if not HAS_YOLO: return

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        prev_time = 0
        device = next(self.clothing_model.parameters()).device

        while self._is_running:
            ret, frame = cap.read()
            if not ret: break
            
            overlay_final = frame.copy()
            orig_h, orig_w = frame.shape[:2]

            if self.frame_count % self.SKIP_FRAMES == 0:
                results = self.yolo.track(frame, persist=True, classes=0, verbose=False, conf=self.CONF_THRESHOLD)
                
                if results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                    
                    for i, box in enumerate(boxes[:self.MAX_PEOPLE]):
                        x1, y1, x2, y2 = box
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(orig_w, x2), min(orig_h, y2)
                        
                        if (x2-x1) < 50 or (y2-y1) < 100: continue

                        person_roi = frame[y1:y2, x1:x2]
                        roi_h, roi_w = person_roi.shape[:2]
                        
                        input_tensor = self.preprocess_roi(person_roi, device)
                        
                        with torch.no_grad():
                            out = self.clothing_model(input_tensor)
                            mask_small = torch.argmax(out, dim=1).squeeze().cpu().numpy().astype(np.uint8)
                        
                        mask_roi = cv2.resize(mask_small, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
                        
                        colored_roi_mask = self.mask_colors[np.clip(mask_roi, 0, 5)].astype(np.uint8)
                        colored_roi_mask_bgr = cv2.cvtColor(colored_roi_mask, cv2.COLOR_RGB2BGR)
                        
                        roi_overlay = overlay_final[y1:y2, x1:x2]
                        mask_bool = mask_roi > 0 
                        
                        alpha = 0.6
                        roi_overlay[mask_bool] = (
                            alpha * colored_roi_mask_bgr[mask_bool] + 
                            (1 - alpha) * roi_overlay[mask_bool]
                        ).astype(np.uint8)
                        
                        overlay_final[y1:y2, x1:x2] = roi_overlay
                        
                        cv2.rectangle(overlay_final, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        unique_in_roi = np.unique(mask_roi)
                        roi_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
                        
                        text_y_offset = y1 + 20
                        cv2.putText(overlay_final, f"Persona {i+1}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        for cls_idx in unique_in_roi:
                            if cls_idx == 0: continue
                            
                            cls_mask = (mask_roi == cls_idx).astype(np.uint8) * 255
                            mean_color = cv2.mean(roi_rgb, mask=cls_mask)[:3]
                            r, g, b = int(mean_color[0]), int(mean_color[1]), int(mean_color[2])
                            
                            label = f"- {self.class_names[cls_idx]}"
                            cv2.rectangle(overlay_final, (x1 + 5, text_y_offset - 10), (x1 + 20, text_y_offset), (b, g, r), -1)
                            cv2.rectangle(overlay_final, (x1 + 5, text_y_offset - 10), (x1 + 20, text_y_offset), (255,255,255), 1)
                            cv2.putText(overlay_final, label, (x1 + 25, text_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            
                            text_y_offset += 20

                self.last_overlay = overlay_final.copy()

            else:
                if self.last_overlay is not None:
                    overlay_final = self.last_overlay

            self.frame_count += 1
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
            prev_time = curr_time
            
            rgb_final = cv2.cvtColor(overlay_final, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_final.shape
            qt_img = QImage(rgb_final.data, w, h, ch*w, QImage.Format.Format_RGB888)
            
            self.frame_processed.emit(qt_img, fps)
            
        cap.release()

    def preprocess_roi(self, img_bgr, device):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        input_size = 256
        img_resized = cv2.resize(img_rgb, (input_size, input_size))
        img_float = img_resized.astype(np.float32) / 255.0
        img_norm = (img_float - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float().unsqueeze(0)
        return tensor.to(device)

    def stop(self):
        self._is_running = False
        self.wait()