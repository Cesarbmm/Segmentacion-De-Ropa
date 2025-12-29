import sys
import os
import ctypes
import torch
from PyQt6.QtWidgets import QApplication, QMainWindow, QStackedWidget
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import Qt

# Importar nuestros módulos
from estilos import STYLESHEET
from motor_ia import OptimizedSegmentationModel
from vistas import SegmentationSplash, MainMenuWidget, StaticAnalysisWidget, RealTimeWidget, Lab3DWidget

class MainController(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Fashion Studio - Ultimate Edition")
        
        icon_path = r"C:\Users\pc\Desktop\Proyecto_Segmentacion\GUI\icons\ropa.png"
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        
        self.model = None
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        
        self.load_global_model()
        
        # Inicializar vistas
        self.menu_view = MainMenuWidget(self)
        self.static_view = StaticAnalysisWidget(self, self.model)
        self.realtime_view = RealTimeWidget(self, self.model)
        self.lab3d_view = Lab3DWidget(self, self.model) # NUEVA VISTA
        
        # Añadir al Stack en orden
        self.stack.addWidget(self.menu_view)     # Index 0
        self.stack.addWidget(self.static_view)   # Index 1
        self.stack.addWidget(self.realtime_view) # Index 2
        self.stack.addWidget(self.lab3d_view)    # Index 3
        
        self.switch_view(0)

    def load_global_model(self):
        path = r"C:\Users\pc\Desktop\Proyecto_Segmentacion\best_model_mejorado.pth"
        if os.path.exists(path):
            self.model = OptimizedSegmentationModel()
            st = torch.load(path, map_location="cpu")
            state = st['model_state_dict'] if 'model_state_dict' in st else st
            self.model.load_state_dict(state)
            self.model.eval()
        else:
            print("⚠️ Modelo no encontrado.")

    def switch_view(self, index):
        self.stack.setCurrentIndex(index)
        
        # Gestión de Cámara (Solo encender si estamos en la vista 2)
        if index == 2: 
            self.realtime_view.start_video()
        elif self.realtime_view.video_thread: 
            if self.realtime_view.video_thread.isRunning():
                self.realtime_view.video_thread.stop()

if __name__ == "__main__":
    try:
        myappid = 'mycompany.fashion.segmentation.v1'
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except ImportError: pass

    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    
    app_icon_path = r"C:\Users\pc\Desktop\Proyecto_Segmentacion\GUI\icons\ropa.png"
    if os.path.exists(app_icon_path):
        app.setWindowIcon(QIcon(app_icon_path))
    
    splash_path = r"C:\Users\pc\Desktop\Proyecto_Segmentacion\GUI\icons\splash.png"
    splash = SegmentationSplash(splash_path)
    splash.show(); splash.raise_(); splash.activateWindow()
    
    main_window = MainController()
    
    def show_main_app():
        main_window.showMaximized()
        main_window.setWindowState(Qt.WindowState.WindowMaximized | Qt.WindowState.WindowActive)
        main_window.raise_(); main_window.activateWindow()
        splash.finish(main_window)
    
    splash.animation_finished.connect(show_main_app)
    sys.exit(app.exec())