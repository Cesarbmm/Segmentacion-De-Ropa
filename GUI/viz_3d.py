# viz_3d.py
import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtGui import QColor

class FashionMannequin(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # --- LAYOUT PRINCIPAL ---
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # Título Estilizado
        self.lbl_title = QLabel("DIGITAL TWIN (3D REAL-TIME)")
        self.lbl_title.setStyleSheet("""
            color: #00e5ff; 
            font-weight: bold; 
            font-size: 14px; 
            padding: 8px; 
            background: #121212; 
            border-bottom: 1px solid #333;
        """)
        self.layout.addWidget(self.lbl_title)

        # --- VISUALIZADOR 3D (PyVista) ---
        self.plotter = QtInteractor(self)
        self.layout.addWidget(self.plotter)
        
        # Configuración de Iluminación y Escena
        self.plotter.set_background("#1e1e1e") # Fondo Gris Oscuro
        # NOTA: Se eliminó SSAO y Eye Dome Lighting para corregir los errores de OpenGL
        self.plotter.enable_anti_aliasing()
        
        # Iluminación básica mejorada (para que no se vea plano sin SSAO)
        self.plotter.add_light(pv.Light(position=(0, -5, 5), intensity=0.8))
        self.plotter.add_light(pv.Light(position=(0, 5, 5), intensity=0.5))
        
        # --- CONSTRUCCIÓN DEL MANIQUÍ (EJE Z HACIA ARRIBA) ---
        # Proporciones más delgadas (Lighter)
        
        # 1. CABEZA Y CUELLO (Grupo Piel/Body)
        self.head = pv.Sphere(radius=0.10, center=(0, 0, 1.75)) # Cabeza más pequeña
        self.neck = pv.Cylinder(center=(0, 0, 1.60), direction=(0, 0, 1), radius=0.045, height=0.15)
        self.skin_head_mesh = self.head + self.neck
        
        self.actor_skin_head = self.plotter.add_mesh(
            self.skin_head_mesh, color="#F5D0A9", smooth_shading=True, specular=0.1
        )
        
        # 2. TORSO (Grupo Top)
        # Más estilizado (radio reducido de 0.15 a 0.13)
        self.torso = pv.Cylinder(center=(0, 0, 1.32), direction=(0, 0, 1), radius=0.135, height=0.55)
        shoulder_l = pv.Sphere(radius=0.11, center=(-0.14, 0, 1.52))
        shoulder_r = pv.Sphere(radius=0.11, center=(0.14, 0, 1.52))
        self.top_mesh = self.torso + shoulder_l + shoulder_r
        
        self.actor_top = self.plotter.add_mesh(
            self.top_mesh, color="#333333", smooth_shading=True
        )
        
        # 3. BRAZOS (Grupo Piel/Body)
        # Más delgados (radio 0.04)
        self.arm_l = pv.Cylinder(center=(-0.19, 0, 1.25), direction=(0, 0, 1), radius=0.04, height=0.55)
        self.arm_r = pv.Cylinder(center=(0.19, 0, 1.25), direction=(0, 0, 1), radius=0.04, height=0.55)
        self.skin_arms_mesh = self.arm_l + self.arm_r
        
        self.actor_skin_arms = self.plotter.add_mesh(
            self.skin_arms_mesh, color="#F5D0A9", smooth_shading=True
        )
        
        # 4. PIERNAS (Grupo Lower)
        # Más delgadas y separadas ligeramente
        self.leg_l = pv.Cylinder(center=(-0.07, 0, 0.55), direction=(0, 0, 1), radius=0.055, height=0.95)
        self.leg_r = pv.Cylinder(center=(0.07, 0, 0.55), direction=(0, 0, 1), radius=0.055, height=0.95)
        self.lower_mesh = self.leg_l + self.leg_r
        
        self.actor_lower = self.plotter.add_mesh(
            self.lower_mesh, color="#222222", smooth_shading=True
        )
        
        # 5. PIES (Grupo Footwear)
        self.foot_l = pv.Cube(center=(-0.07, 0.05, 0.06), x_length=0.08, y_length=0.20, z_length=0.10)
        self.foot_r = pv.Cube(center=(0.07, 0.05, 0.06), x_length=0.08, y_length=0.20, z_length=0.10)
        self.feet_mesh = self.foot_l + self.foot_r
        
        self.actor_feet = self.plotter.add_mesh(
            self.feet_mesh, color="#111111", smooth_shading=True
        )

        # 6. BASE (Plataforma Minimalista)
        self.base = pv.Cylinder(center=(0, 0, -0.02), direction=(0, 0, 1), radius=0.4, height=0.02)
        self.plotter.add_mesh(self.base, color="#00e5ff", opacity=0.3)

        # --- CÁMARA (VISTA COMPLETA) ---
        # Alejamos la cámara (Y = -4.5) para que el maniquí se vea más pequeño y completo
        self.plotter.camera_position = [(0, -4.5, 2.0), (0, 0, 0.9), (0, 0, 1)]
        self.plotter.camera.zoom(1.0) # Zoom normal

    def update_clothing_color(self, category_index, hex_color):
        """
        Actualiza el color de la parte del cuerpo correspondiente.
        """
        try:
            # 1 = TOP
            if category_index == 1: 
                self.plotter.remove_actor(self.actor_top)
                self.actor_top = self.plotter.add_mesh(self.top_mesh, color=hex_color, smooth_shading=True)
                
            # 2 = LOWER
            elif category_index == 2: 
                self.plotter.remove_actor(self.actor_lower)
                self.actor_lower = self.plotter.add_mesh(self.lower_mesh, color=hex_color, smooth_shading=True)
            
            # 3 = DRESS
            elif category_index == 3: 
                self.plotter.remove_actor(self.actor_top)
                self.plotter.remove_actor(self.actor_lower)
                self.actor_top = self.plotter.add_mesh(self.top_mesh, color=hex_color, smooth_shading=True)
                self.actor_lower = self.plotter.add_mesh(self.lower_mesh, color=hex_color, smooth_shading=True)

            # 4 = FEET
            elif category_index == 4: 
                self.plotter.remove_actor(self.actor_feet)
                self.actor_feet = self.plotter.add_mesh(self.feet_mesh, color=hex_color, smooth_shading=True)

            # 5 = BODY/SKIN
            elif category_index == 5: 
                self.plotter.remove_actor(self.actor_skin_head)
                self.plotter.remove_actor(self.actor_skin_arms)
                self.actor_skin_head = self.plotter.add_mesh(self.skin_head_mesh, color=hex_color, smooth_shading=True, specular=0.1)
                self.actor_skin_arms = self.plotter.add_mesh(self.skin_arms_mesh, color=hex_color, smooth_shading=True)
            
            self.plotter.render()
            
        except Exception as e:
            print(f"Error actualizando 3D: {e}")

    def reset_mannequin(self):
        """Devuelve el maniquí a su estado base"""
        self.update_clothing_color(1, "#333333") 
        self.update_clothing_color(2, "#222222") 
        self.update_clothing_color(4, "#111111") 
        self.update_clothing_color(5, "#F5D0A9")