import os
import cv2
import numpy as np
import random
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
                             QFileDialog, QScrollArea, QProgressBar, QSplitter, 
                             QCheckBox, QSplashScreen, QMessageBox)
from PyQt6.QtGui import QPixmap, QImage, QColor, QPainter, QBrush, QPen, QIcon
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QSize

# Importamos lógica necesaria
from motor_ia import SegmentationWorker, VideoWorker, apply_refinement_to_full_mask, colorize_mask, get_dominant_color

# Intentar importar el módulo 3D
try:
    from viz_3d import FashionMannequin
    HAS_3D = True
except ImportError:
    HAS_3D = False
    print("⚠️ No se encontró viz_3d.py. El modo 3D no funcionará.")

# =============================================================================
# SPLASH SCREEN
# =============================================================================
class SegmentationSplash(QSplashScreen):
    animation_finished = pyqtSignal()

    def __init__(self, image_path):
        if not os.path.exists(image_path):
            pixmap = QPixmap(600, 400); pixmap.fill(QColor("#1e1e1e"))
        else:
            pixmap = QPixmap(image_path)
            pixmap = pixmap.scaled(600, 400, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

        super().__init__(pixmap)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.mask_color = QColor(random.randint(50, 255), random.randint(50, 255), random.randint(50, 255), 200)
        self.progress = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_progress)
        self.timer.start(40) 

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.drawPixmap(0, 0, self.pixmap())
        w = self.width(); h = self.height()
        fill_height = int(h * (self.progress / 100))
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceAtop)
        painter.setBrush(QBrush(self.mask_color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(0, h - fill_height, w, fill_height)
        painter.end()

    def update_progress(self):
        self.progress += 1
        self.repaint()
        if self.progress >= 110:
            self.timer.stop()
            self.animation_finished.emit()

# =============================================================================
# MENÚ PRINCIPAL
# =============================================================================
class MainMenuWidget(QWidget):
    def __init__(self, parent_controller):
        super().__init__()
        self.controller = parent_controller
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # LOGO
        logo_path = r"C:\Users\pc\Desktop\Proyecto_Segmentacion\GUI\icons\ropa.png"
        if os.path.exists(logo_path):
            lbl_logo = QLabel()
            pix_logo = QPixmap(logo_path).scaled(120, 120, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            lbl_logo.setPixmap(pix_logo)
            lbl_logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(lbl_logo)

        # TITULO
        title = QLabel("AI FASHION STUDIO")
        title.setStyleSheet("font-size: 32px; font-weight: bold; color: white; letter-spacing: 3px; margin-bottom: 40px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        grid = QHBoxLayout()
        grid.setSpacing(30)
        
        # --- 1. Estático ---
        btn_static = QPushButton("\nAnálisis Estático")
        btn_static.setObjectName("MenuCard")
        btn_static.setFixedSize(250, 250)
        icon_static = r"C:\Users\pc\Desktop\Proyecto_Segmentacion\GUI\icons\estatica.png"
        if os.path.exists(icon_static):
            btn_static.setIcon(QIcon(icon_static))
            btn_static.setIconSize(QSize(80, 80))
        btn_static.clicked.connect(lambda: self.controller.switch_view(1))
        
        # --- 2. Real Time ---
        btn_realtime = QPushButton("\nCámara en Vivo")
        btn_realtime.setObjectName("MenuCard")
        btn_realtime.setFixedSize(250, 250)
        icon_cam = r"C:\Users\pc\Desktop\Proyecto_Segmentacion\GUI\icons\camara.png"
        if os.path.exists(icon_cam):
            btn_realtime.setIcon(QIcon(icon_cam))
            btn_realtime.setIconSize(QSize(80, 80))
        btn_realtime.clicked.connect(lambda: self.controller.switch_view(2))

        # --- 3. Modelado 3D (NUEVO) ---
        btn_3d = QPushButton("\nModelado 3D")
        btn_3d.setObjectName("MenuCard")
        btn_3d.setFixedSize(250, 250)
        
        # AQUI AGREGAMOS TU ICONO DE MANIQUI
        icon_3d = r"C:\Users\pc\Desktop\Proyecto_Segmentacion\GUI\icons\maniqui.png" 
        if os.path.exists(icon_3d):
            btn_3d.setIcon(QIcon(icon_3d))
            btn_3d.setIconSize(QSize(80, 80))
            
        btn_3d.clicked.connect(lambda: self.controller.switch_view(3))


        # --- 4. Multi-Persona (NUEVO) ---
        btn_multi = QPushButton("\nMulti-Persona")
        btn_multi.setObjectName("MenuCard")
        btn_multi.setFixedSize(250, 250)
        
        # AQUI AGREGAMOS TU ICONO DE MULTI-PERSONA
        icon_multi = r"C:\Users\pc\Desktop\Proyecto_Segmentacion\GUI\icons\multi.png" 
        if os.path.exists(icon_multi):
            btn_multi.setIcon(QIcon(icon_multi))
            btn_multi.setIconSize(QSize(80, 80))
            
        btn_multi.clicked.connect(lambda: self.controller.switch_view(4))
        
        grid.addWidget(btn_static)
        grid.addWidget(btn_realtime)
        grid.addWidget(btn_3d)
        grid.addWidget(btn_multi)
        layout.addLayout(grid)

# =============================================================================
# VISTA 2: ANÁLISIS ESTÁTICO (Clásico 2D)
# =============================================================================
class StaticAnalysisWidget(QWidget):
    def __init__(self, controller, model):
        super().__init__()
        self.controller = controller
        self.model = model
        self.raw_mask = None; self.current_mask = None; self.current_img = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self); layout.setSpacing(15); layout.setContentsMargins(20,20,20,20)
        
        header = QHBoxLayout()
        btn_back = QPushButton("←")
        btn_back.setObjectName("BtnBack")
        btn_back.setFixedSize(60, 60)
        btn_back.clicked.connect(lambda: self.controller.switch_view(0))
        header.addWidget(btn_back)
        
        title = QLabel("ANÁLISIS ESTÁTICO DETALLADO")
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin-left: 10px;")
        header.addWidget(title); header.addStretch()
        layout.addLayout(header)

        btns = QHBoxLayout()
        self.btn_load = QPushButton("  Cargar Imagen"); self.btn_load.setObjectName("BtnLoad")
        icon_load = r"C:\Users\pc\Desktop\Proyecto_Segmentacion\GUI\icons\incrementar.png"
        if os.path.exists(icon_load): self.btn_load.setIcon(QIcon(icon_load))
        self.btn_load.clicked.connect(self.upload)
        
        self.chk_refine = QCheckBox("Auto-Completar"); self.chk_refine.setChecked(True)
        self.chk_refine.toggled.connect(self.update_mask_display)
        
        self.btn_sep = QPushButton("  Separar & Analizar"); self.btn_sep.setObjectName("BtnExtract")
        self.btn_sep.setEnabled(False)
        icon_sep = r"C:\Users\pc\Desktop\Proyecto_Segmentacion\GUI\icons\color.png"
        if os.path.exists(icon_sep): self.btn_sep.setIcon(QIcon(icon_sep))
        self.btn_sep.clicked.connect(self.separate)

        # Botón Guardar
        self.btn_save = QPushButton("  Guardar Máscara")
        self.btn_save.setObjectName("BtnSave")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self.save_mask_to_disk)
        
        btns.addWidget(self.btn_load); btns.addWidget(self.chk_refine); btns.addWidget(self.btn_sep); btns.addWidget(self.btn_save); btns.addStretch()
        layout.addLayout(btns)

        self.prog = QProgressBar(); self.prog.setVisible(False); layout.addWidget(self.prog)

        split = QSplitter(Qt.Orientation.Horizontal)
        self.lbl_orig = self.mk_lbl("Original")
        self.lbl_mask = self.mk_lbl("Máscara")
        split.addWidget(self.lbl_orig); split.addWidget(self.lbl_mask); split.setSizes([500, 500])
        layout.addWidget(split, stretch=3)

        layout.addWidget(QLabel("RESULTADOS:"))
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setMinimumHeight(200)
        self.gal_content = QWidget(); self.gal_layout = QHBoxLayout(self.gal_content); self.gal_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        scroll.setWidget(self.gal_content)
        layout.addWidget(scroll, stretch=1)

    def mk_lbl(self, txt):
        l = QLabel(txt); l.setObjectName("ImageDisplay"); l.setAlignment(Qt.AlignmentFlag.AlignCenter); return l

    # --- CORRECCIÓN DEL ERROR AQUÍ ---
    def clear_gal(self):
        while self.gal_layout.count():
            item = self.gal_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def upload(self):
        f, _ = QFileDialog.getOpenFileName(self, "Img", "", "*.jpg *.png *.jpeg")
        if f:
            self.prog.setVisible(True); self.prog.setRange(0,0); self.clear_gal()
            self.worker = SegmentationWorker(self.model, f)
            self.worker.finished.connect(self.done_seg)
            self.worker.start()

    def done_seg(self, raw, _, img):
        self.prog.setVisible(False); self.raw_mask = raw; self.current_img = img
        self.update_mask_display() # Esto ya llama a show_img
        self.btn_sep.setEnabled(True)
        self.btn_save.setEnabled(True)

    def update_mask_display(self):
        if self.raw_mask is None: return
        self.current_mask = apply_refinement_to_full_mask(self.raw_mask) if self.chk_refine.isChecked() else self.raw_mask.copy()
        self.show_img(self.current_img, self.lbl_orig)
        self.show_img(colorize_mask(self.current_mask), self.lbl_mask)

    def show_img(self, img, lbl):
        h, w = img.shape[:2]
        fmt = QImage.Format.Format_RGB888
        if len(img.shape)==2: fmt=QImage.Format.Format_Grayscale8
        qi = QImage(img.data, w, h, 3*w if len(img.shape)>2 else w, fmt)
        lbl.setPixmap(QPixmap.fromImage(qi).scaled(lbl.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def save_mask_to_disk(self):
        if self.current_mask is None: return
        path, _ = QFileDialog.getSaveFileName(self, "Guardar", "", "PNG (*.png)")
        if path:
            rgb = colorize_mask(self.current_mask)
            cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            QMessageBox.information(self, "OK", "Guardado.")

    def separate(self):
        self.clear_gal()
        names = ["", "Top", "Pantalón", "Vestido", "Zapatos", "Piel"]
        for u in np.unique(self.current_mask):
            if u == 0: continue
            mb = (self.current_mask == u).astype(np.uint8)*255
            x,y,w,h = cv2.boundingRect(mb)
            cr = self.current_img[y:y+h, x:x+w]
            cm = mb[y:y+h, x:x+w]
            hex_c, _ = get_dominant_color(cr, cm)
            rgba = cv2.merge([*cv2.split(cr), cm]) if cr.shape[2]==3 else cr
            self.add_gal(rgba, names[u], hex_c)

    def add_gal(self, rgba, name, hex_c):
        w = QWidget(); w.setObjectName("GalleryItem"); w.setFixedSize(160, 220); l = QVBoxLayout(w)
        l.setSpacing(5); l.setContentsMargins(10,10,10,10)
        
        qi = QImage(rgba.data, rgba.shape[1], rgba.shape[0], 4*rgba.shape[1], QImage.Format.Format_RGBA8888)
        lb_i = QLabel(); lb_i.setPixmap(QPixmap.fromImage(qi).scaled(100,100,Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation))
        lb_i.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        lb_n = QLabel(name); lb_n.setAlignment(Qt.AlignmentFlag.AlignCenter); lb_n.setStyleSheet("font-weight:bold; color:white;")
        
        # Color Box Container
        box_c = QHBoxLayout(); box_c.setSpacing(10)
        lb_sq = QLabel(); lb_sq.setFixedSize(20,20); lb_sq.setStyleSheet(f"background:{hex_c}; border:2px solid white; border-radius:4px;")
        lb_tx = QLabel(hex_c); lb_tx.setStyleSheet("color:#ccc; font-family:monospace;")
        box_c.addStretch(); box_c.addWidget(lb_sq); box_c.addWidget(lb_tx); box_c.addStretch()
        
        l.addWidget(lb_i); l.addWidget(lb_n); l.addLayout(box_c)
        self.gal_layout.addWidget(w)

# =============================================================================
# VISTA 3: LABORATORIO 3D
# =============================================================================
class Lab3DWidget(QWidget):
    def __init__(self, controller, model):
        super().__init__()
        self.controller = controller
        self.model = model
        self.raw_mask = None; self.current_mask = None; self.current_img = None
        
        if HAS_3D:
            self.init_ui()
        else:
            self.init_error_ui()

    def init_error_ui(self):
        layout = QVBoxLayout(self)
        btn_back = QPushButton("← Regresar")
        btn_back.clicked.connect(lambda: self.controller.switch_view(0))
        layout.addWidget(btn_back)
        layout.addWidget(QLabel("ERROR: viz_3d.py no encontrado o PyVista no instalado."))

    def init_ui(self):
        layout = QVBoxLayout(self); layout.setContentsMargins(20,20,20,20)
        
        header = QHBoxLayout()
        btn_back = QPushButton("← Menú")
        btn_back.setObjectName("BtnBack")
        btn_back.setFixedSize(80, 40)
        btn_back.clicked.connect(lambda: self.controller.switch_view(0))
        header.addWidget(btn_back)
        
        title = QLabel("LABORATORIO DE DISEÑO 3D & DIGITAL TWIN")
        title.setStyleSheet("font-size: 20px; font-weight: bold; margin-left: 10px; color: white;")
        header.addWidget(title)
        header.addStretch()
        layout.addLayout(header)

        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0,0,0,0)
        
        btns = QHBoxLayout()
        self.btn_load = QPushButton(" Cargar Imagen"); self.btn_load.setObjectName("BtnLoad")
        self.btn_load.clicked.connect(self.upload)
        self.btn_process = QPushButton(" Generar Modelo 3D"); self.btn_process.setObjectName("BtnExtract")
        self.btn_process.setEnabled(False)
        self.btn_process.clicked.connect(self.process_3d)
        
        btns.addWidget(self.btn_load); btns.addWidget(self.btn_process)
        left_layout.addLayout(btns)
        
        self.lbl_orig = QLabel("Cargue una imagen")
        self.lbl_orig.setObjectName("ImageDisplay")
        self.lbl_orig.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(self.lbl_orig)
        
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0,0,0,0)
        
        self.mannequin = FashionMannequin()
        self.mannequin.setMinimumHeight(400)
        right_layout.addWidget(self.mannequin, stretch=3)
        
        right_layout.addWidget(QLabel("Muestras de Color Detectadas:"))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(150)
        self.gal_content = QWidget(); self.gal_layout = QHBoxLayout(self.gal_content); self.gal_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        scroll.setWidget(self.gal_content)
        right_layout.addWidget(scroll, stretch=1)

        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(right_panel)
        main_splitter.setSizes([400, 700])
        
        layout.addWidget(main_splitter)
        self.prog = QProgressBar(); self.prog.setVisible(False); layout.addWidget(self.prog)

    # --- CORRECCIÓN DEL ERROR TAMBIÉN AQUÍ ---
    def clear_gal(self):
        while self.gal_layout.count():
            item = self.gal_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def upload(self):
        f, _ = QFileDialog.getOpenFileName(self, "Img", "", "*.jpg *.png *.jpeg")
        if f:
            self.prog.setVisible(True); self.prog.setRange(0,0)
            self.mannequin.reset_mannequin()
            self.clear_gal()
            self.worker = SegmentationWorker(self.model, f)
            self.worker.finished.connect(self.done_seg)
            self.worker.start()

    def done_seg(self, raw, _, img):
        self.prog.setVisible(False); self.raw_mask = raw; self.current_img = img
        h, w = img.shape[:2]
        qi = QImage(img.data, w, h, 3*w, QImage.Format.Format_RGB888)
        self.lbl_orig.setPixmap(QPixmap.fromImage(qi).scaled(self.lbl_orig.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.btn_process.setEnabled(True)

    def process_3d(self):
        self.clear_gal()
        unique_classes = np.unique(self.raw_mask)
        names = ["", "Top", "Pantalón", "Vestido", "Zapatos", "Piel"]
        
        for u in unique_classes:
            if u == 0: continue
            
            mb = (self.raw_mask == u).astype(np.uint8)*255
            x,y,w,h = cv2.boundingRect(mb)
            cr = self.current_img[y:y+h, x:x+w]
            cm = mb[y:y+h, x:x+w]
            hex_c, _ = get_dominant_color(cr, cm)
            
            self.mannequin.update_clothing_color(int(u), hex_c)
            
            rgba = cv2.merge([*cv2.split(cr), cm]) if cr.shape[2]==3 else cr
            self.add_gal(rgba, names[u], hex_c)

    def add_gal(self, rgba, name, hex_c):
        w = QWidget(); w.setObjectName("GalleryItem"); w.setFixedSize(120, 140); l = QVBoxLayout(w)
        l.setContentsMargins(2,2,2,2)
        qi = QImage(rgba.data, rgba.shape[1], rgba.shape[0], 4*rgba.shape[1], QImage.Format.Format_RGBA8888)
        lb_i = QLabel(); lb_i.setPixmap(QPixmap.fromImage(qi).scaled(60,60,Qt.AspectRatioMode.KeepAspectRatio))
        lb_i.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lb_c = QLabel(); lb_c.setFixedHeight(15); lb_c.setStyleSheet(f"background:{hex_c};border:1px solid white;border-radius:2px")
        l.addWidget(lb_i); l.addWidget(QLabel(name, alignment=Qt.AlignmentFlag.AlignCenter)); l.addWidget(lb_c)
        self.gal_layout.addWidget(w)

# =============================================================================
# REAL TIME WIDGET
# =============================================================================
class RealTimeWidget(QWidget):
    def __init__(self, controller, model):
        super().__init__()
        self.controller = controller
        self.model = model
        self.video_thread = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        header = QHBoxLayout()
        btn_back = QPushButton("←")
        btn_back.setObjectName("BtnBack")
        btn_back.setFixedSize(60, 60)
        btn_back.clicked.connect(self.stop_and_exit)
        
        header.addWidget(btn_back)
        header.addWidget(QLabel("ANÁLISIS EN VIVO: Detección de Clase + Color"))
        header.addStretch()
        layout.addLayout(header)

        self.lbl_video = QLabel("Iniciando cámara...")
        self.lbl_video.setObjectName("ImageDisplay")
        self.lbl_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_video.setMinimumSize(800, 600)
        layout.addWidget(self.lbl_video, stretch=1)

        self.lbl_fps = QLabel("FPS: 0")
        self.lbl_fps.setStyleSheet("color: #007acc; font-weight: bold; font-size: 16px;")
        layout.addWidget(self.lbl_fps)

    def start_video(self):
        if self.video_thread is not None and self.video_thread.isRunning():
            return
        self.video_thread = VideoWorker(self.model)
        self.video_thread.frame_processed.connect(self.update_frame)
        self.video_thread.start()
    def add_gal(self, rgba, name, hex_c):
        w = QWidget(); w.setObjectName("GalleryItem"); w.setFixedSize(160, 220); l = QVBoxLayout(w)
        l.setSpacing(5); l.setContentsMargins(10,10,10,10)
        
        qi = QImage(rgba.data, rgba.shape[1], rgba.shape[0], 4*rgba.shape[1], QImage.Format.Format_RGBA8888)
        lb_i = QLabel(); lb_i.setPixmap(QPixmap.fromImage(qi).scaled(100,100,Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation))
        lb_i.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        lb_n = QLabel(name); lb_n.setAlignment(Qt.AlignmentFlag.AlignCenter); lb_n.setStyleSheet("font-weight:bold; color:white;")
        
        # Color Box Container
        box_c = QHBoxLayout(); box_c.setSpacing(10)
        lb_sq = QLabel(); lb_sq.setFixedSize(20,20); lb_sq.setStyleSheet(f"background:{hex_c}; border:2px solid white; border-radius:4px;")
        lb_tx = QLabel(hex_c); lb_tx.setStyleSheet("color:#ccc; font-family:monospace;")
        box_c.addStretch(); box_c.addWidget(lb_sq); box_c.addWidget(lb_tx); box_c.addStretch()
        
        l.addWidget(lb_i); l.addWidget(lb_n); l.addLayout(box_c)
        self.gal_layout.addWidget(w)

# =============================================================================
# VISTA 3: LABORATORIO 3D
# =============================================================================
class Lab3DWidget(QWidget):
    def __init__(self, controller, model):
        super().__init__()
        self.controller = controller
        self.model = model
        self.raw_mask = None; self.current_mask = None; self.current_img = None
        
        if HAS_3D:
            self.init_ui()
        else:
            self.init_error_ui()

    def init_error_ui(self):
        layout = QVBoxLayout(self)
        btn_back = QPushButton("← Regresar")
        btn_back.clicked.connect(lambda: self.controller.switch_view(0))
        layout.addWidget(btn_back)
        layout.addWidget(QLabel("ERROR: viz_3d.py no encontrado o PyVista no instalado."))

    def init_ui(self):
        layout = QVBoxLayout(self); layout.setContentsMargins(20,20,20,20)
        
        header = QHBoxLayout()
        btn_back = QPushButton("← Menú")
        btn_back.setObjectName("BtnBack")
        btn_back.setFixedSize(140, 70)
        btn_back.clicked.connect(lambda: self.controller.switch_view(0))
        header.addWidget(btn_back)
        
        title = QLabel("LABORATORIO DE DISEÑO 3D & DIGITAL TWIN")
        title.setStyleSheet("font-size: 20px; font-weight: bold; margin-left: 10px; color: white;")
        header.addWidget(title)
        header.addStretch()
        layout.addLayout(header)

        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0,0,0,0)
        
        btns = QHBoxLayout()
        self.btn_load = QPushButton(" Cargar Imagen"); self.btn_load.setObjectName("BtnLoad")
        self.btn_load.clicked.connect(self.upload)
        self.btn_process = QPushButton(" Generar Modelo 3D"); self.btn_process.setObjectName("BtnExtract")
        self.btn_process.setEnabled(False)
        self.btn_process.clicked.connect(self.process_3d)
        
        btns.addWidget(self.btn_load); btns.addWidget(self.btn_process)
        left_layout.addLayout(btns)
        
        self.lbl_orig = QLabel("Cargue una imagen")
        self.lbl_orig.setObjectName("ImageDisplay")
        self.lbl_orig.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(self.lbl_orig)
        
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0,0,0,0)
        
        self.mannequin = FashionMannequin()
        self.mannequin.setMinimumHeight(400)
        right_layout.addWidget(self.mannequin, stretch=3)
        
        right_layout.addWidget(QLabel("Muestras de Color Detectadas:"))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(150)
        self.gal_content = QWidget(); self.gal_layout = QHBoxLayout(self.gal_content); self.gal_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        scroll.setWidget(self.gal_content)
        right_layout.addWidget(scroll, stretch=1)

        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(right_panel)
        main_splitter.setSizes([400, 700])
        
        layout.addWidget(main_splitter)
        self.prog = QProgressBar(); self.prog.setVisible(False); layout.addWidget(self.prog)

    # --- CORRECCIÓN DEL ERROR TAMBIÉN AQUÍ ---
    def clear_gal(self):
        while self.gal_layout.count():
            item = self.gal_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def upload(self):
        f, _ = QFileDialog.getOpenFileName(self, "Img", "", "*.jpg *.png *.jpeg")
        if f:
            self.prog.setVisible(True); self.prog.setRange(0,0)
            self.mannequin.reset_mannequin()
            self.clear_gal()
            self.worker = SegmentationWorker(self.model, f)
            self.worker.finished.connect(self.done_seg)
            self.worker.start()

    def done_seg(self, raw, _, img):
        self.prog.setVisible(False); self.raw_mask = raw; self.current_img = img
        h, w = img.shape[:2]
        qi = QImage(img.data, w, h, 3*w, QImage.Format.Format_RGB888)
        self.lbl_orig.setPixmap(QPixmap.fromImage(qi).scaled(self.lbl_orig.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.btn_process.setEnabled(True)

    def process_3d(self):
        self.clear_gal()
        unique_classes = np.unique(self.raw_mask)
        names = ["", "Top", "Pantalón", "Vestido", "Zapatos", "Piel"]
        
        for u in unique_classes:
            if u == 0: continue
            
            mb = (self.raw_mask == u).astype(np.uint8)*255
            x,y,w,h = cv2.boundingRect(mb)
            cr = self.current_img[y:y+h, x:x+w]
            cm = mb[y:y+h, x:x+w]
            hex_c, _ = get_dominant_color(cr, cm)
            
            self.mannequin.update_clothing_color(int(u), hex_c)
            
            rgba = cv2.merge([*cv2.split(cr), cm]) if cr.shape[2]==3 else cr
            self.add_gal(rgba, names[u], hex_c)

    def add_gal(self, rgba, name, hex_c):
        w = QWidget(); w.setObjectName("GalleryItem"); w.setFixedSize(120, 140); l = QVBoxLayout(w)
        l.setContentsMargins(2,2,2,2)
        qi = QImage(rgba.data, rgba.shape[1], rgba.shape[0], 4*rgba.shape[1], QImage.Format.Format_RGBA8888)
        lb_i = QLabel(); lb_i.setPixmap(QPixmap.fromImage(qi).scaled(60,60,Qt.AspectRatioMode.KeepAspectRatio))
        lb_i.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lb_c = QLabel(); lb_c.setFixedHeight(15); lb_c.setStyleSheet(f"background:{hex_c};border:1px solid white;border-radius:2px")
        l.addWidget(lb_i); l.addWidget(QLabel(name, alignment=Qt.AlignmentFlag.AlignCenter)); l.addWidget(lb_c)
        self.gal_layout.addWidget(w)

# =============================================================================
# REAL TIME WIDGET
# =============================================================================
class RealTimeWidget(QWidget):
    def __init__(self, controller, model):
        super().__init__()
        self.controller = controller
        self.model = model
        self.video_thread = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

               
        # boton volver rojo como los demas
        header = QHBoxLayout()
        btn_back = QPushButton("←")
        btn_back.setStyleSheet("background-color: red; color: white;")
        btn_back.setObjectName("BtnBack") 
        btn_back.setFixedSize(60, 60) 
        btn_back.clicked.connect(self.stop_and_exit)
        
        header.addWidget(btn_back)
        header.addWidget(QLabel("ANÁLISIS EN VIVO: Detección de Clase + Color"))
        header.addStretch()
        layout.addLayout(header)

        self.lbl_video = QLabel("Iniciando cámara...")
        self.lbl_video.setObjectName("ImageDisplay")
        self.lbl_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_video.setMinimumSize(800, 600)
        layout.addWidget(self.lbl_video, stretch=1)

        self.lbl_fps = QLabel("FPS: 0")
        self.lbl_fps.setStyleSheet("color: #007acc; font-weight: bold; font-size: 16px;")
        layout.addWidget(self.lbl_fps)

    def start_video(self):
        if self.video_thread is not None and self.video_thread.isRunning():
            return
        self.video_thread = VideoWorker(self.model)
        self.video_thread.frame_processed.connect(self.update_frame)
        self.video_thread.start()

    def update_frame(self, qt_img, fps):
        self.lbl_video.setPixmap(QPixmap.fromImage(qt_img).scaled(self.lbl_video.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.lbl_fps.setText(f"FPS: {fps:.1f}")

    def stop_video(self):
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()

    def stop_and_exit(self):
        self.stop_video()
        self.controller.switch_view(0)


# =============================================================================
# VISTA 4: MULTI-PERSONA (YOLO + DeepLab)
# =============================================================================
class MultiPersonWidget(QWidget):
    def __init__(self, controller, model):
        super().__init__()
        self.controller = controller
        self.model = model
        self.worker = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Header
        header = QHBoxLayout()
        btn_back = QPushButton("←")
        btn_back.setStyleSheet("background-color: red; color: white;")
        btn_back.setFixedSize(60, 60)
        btn_back.clicked.connect(self.stop_and_exit)
        header.addWidget(btn_back)
        
        title = QLabel("MODO MULTI-PERSONA (CPU OPTIMIZED)")
        title.setStyleSheet("color: white; font-weight: bold; font-size: 18px;")
        header.addWidget(title)
        header.addStretch()
        layout.addLayout(header)

        # Video Area
        self.lbl_video = QLabel("Iniciando detección de múltiples sujetos...")
        self.lbl_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_video.setMinimumSize(800, 600)
        self.lbl_video.setStyleSheet("background-color: #000; border: 2px solid #333;")
        layout.addWidget(self.lbl_video, stretch=1)

        # Stats
        self.lbl_info = QLabel("FPS: 0 | Max Personas: 3")
        self.lbl_info.setStyleSheet("color: #00ff00; font-family: monospace; font-size: 14px;")
        layout.addWidget(self.lbl_info)

    def start_video(self):
        # Importamos aquí para evitar errores circulares si no está en el top
        from motor_ia import MultiPersonWorker
        
        if self.worker is not None and self.worker.isRunning(): return
        
        self.worker = MultiPersonWorker(self.model)
        self.worker.frame_processed.connect(self.update_frame)
        self.worker.start()

    def update_frame(self, qt_img, fps):
        self.lbl_video.setPixmap(QPixmap.fromImage(qt_img).scaled(
            self.lbl_video.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        ))
        self.lbl_info.setText(f"FPS: {fps:.1f} | Detección Activa")

    def stop_video(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()

    def stop_and_exit(self):
        self.stop_video()
        self.controller.switch_view(0)