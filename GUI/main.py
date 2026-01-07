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
from vistas import SegmentationSplash, MainMenuWidget, StaticAnalysisWidget, RealTimeWidget, Lab3DWidget, MultiPersonWidget

class MainController(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Fashion Studio - Ultimate Edition")
        
        # --- 1. MODO PRO DE RUTAS (Relativas y Dinámicas) ---
        self.gui_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(self.gui_dir)
        
        # Definimos rutas clave
        self.icons_dir = os.path.join(self.gui_dir, "icons")
        self.model_path = os.path.join(self.project_root, "best_model_mejorado.pth")
        
        # Icono de la ventana principal
        main_icon = os.path.join(self.icons_dir, "ropa.png")
        if os.path.exists(main_icon):
            self.setWindowIcon(QIcon(main_icon))
        
        self.model = None
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        
        self.load_global_model()
        
        # --- 2. INICIALIZAR VISTAS ---
        self.menu_view = MainMenuWidget(self)
        self.static_view = StaticAnalysisWidget(self, self.model)
        self.realtime_view = RealTimeWidget(self, self.model)      # Nombre correcto
        self.lab3d_view = Lab3DWidget(self, self.model)
        self.multi_view = MultiPersonWidget(self, self.model)      # Nombre correcto
        
        # Añadir al Stack en orden
        self.stack.addWidget(self.menu_view)     # Index 0
        self.stack.addWidget(self.static_view)   # Index 1
        self.stack.addWidget(self.realtime_view) # Index 2
        self.stack.addWidget(self.lab3d_view)    # Index 3
        self.stack.addWidget(self.multi_view)    # Index 4
        
        self.switch_view(0)

    def load_global_model(self):
        print(f"Buscando modelo en: {self.model_path}")
        
        if os.path.exists(self.model_path):
            try:
                self.model = OptimizedSegmentationModel()
                st = torch.load(self.model_path, map_location="cpu")
                # Manejo robusto del state_dict
                state = st['model_state_dict'] if isinstance(st, dict) and 'model_state_dict' in st else st
                self.model.load_state_dict(state)
                self.model.eval()
                print("✅ Modelo cargado exitosamente.")
            except Exception as e:
                print(f"❌ Error cargando modelo: {e}")
        else:
            print("⚠️ Modelo no encontrado.")

    def switch_view(self, index):
        self.stack.setCurrentIndex(index)
        
        # --- 3. GESTIÓN DE CÁMARAS ---
        
        # Caso: Entrando a Real Time (1 persona)
        if index == 2: 
            if hasattr(self, 'realtime_view'):
                self.realtime_view.start_video()

        # Caso: Entrando a Multi Persona
        elif index == 4: 
            if hasattr(self, 'multi_view'):
                self.multi_view.start_video()
                
        # Caso: Saliendo (Menú u otros)
        # NOTA: No es necesario llamar a stop() aquí porque el botón "Volver"
        # en las vistas (vistas.py) ya se encarga de detener el video 
        # antes de cambiar la vista.

if __name__ == "__main__":
    try:
        myappid = 'mycompany.fashion.segmentation.v1'
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except ImportError: pass

    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    
    # --- RUTAS DINÁMICAS PARA EL MAIN ---
    base_dir = os.path.dirname(os.path.abspath(__file__))
    icons_dir = os.path.join(base_dir, "icons")
    
    app_icon_path = os.path.join(icons_dir, "ropa.png")
    if os.path.exists(app_icon_path):
        app.setWindowIcon(QIcon(app_icon_path))
    
    splash_path = os.path.join(icons_dir, "splash.png")
    
    # Splash Screen
    splash = SegmentationSplash(splash_path)
    splash.show()
    splash.raise_()
    splash.activateWindow()
    
    # Iniciar Ventana Principal
    main_window = MainController()
    
    def show_main_app():
        main_window.showMaximized()
        main_window.setWindowState(Qt.WindowState.WindowMaximized | Qt.WindowState.WindowActive)
        main_window.raise_()
        main_window.activateWindow()
        splash.finish(main_window)
    
    splash.animation_finished.connect(show_main_app)
    sys.exit(app.exec())