# estilos.py
STYLESHEET = """
QMainWindow { 
    background-color: #1e1e1e; 
}

QLabel { 
    color: #e0e0e0; 
    font-family: 'Segoe UI', sans-serif; 
    font-size: 14px; 
}

QScrollArea { 
    border: none; 
    background-color: #1e1e1e; 
}

QLabel#ImageDisplay { 
    background-color: #2d2d2d; 
    border: 2px dashed #444; 
    border-radius: 10px; 
}

/* --- BOTONES GENERALES (Base) --- */
QPushButton { 
    background-color: #3a3a3a; 
    color: white; 
    border: none; 
    padding: 15px 25px; /* Más relleno para que se vean grandes */
    border-radius: 10px; 
    font-weight: bold; 
    font-size: 18px; /* Letra más legible */
}
QPushButton:hover { 
    background-color: #505050; 
}
QPushButton:disabled { 
    background-color: #2a2a2a; 
    color: #666; 
}

/* --- TARJETAS DEL MENÚ PRINCIPAL --- */
QPushButton#MenuCard {
    background-color: #2d2d2d;
    border: 2px solid #3d3d3d;
    border-radius: 105px;
    padding: 5px; /* Mucho espacio interno */
    text-align: center;
    font-size: 18px; /* Texto grande para los títulos */
    letter-spacing: 1px;
}
QPushButton#MenuCard:hover {
    background-color: #383838;
    border: 2px solid #007acc;
    margin-top: -5px;
}

/* --- BOTONES DE ACCIÓN LATERALES --- */
/* Tienen altura fija para que se vean uniformes con los iconos */

/* Cargar (Azul) */
QPushButton#BtnLoad { 
    background-color: #007acc; 
    text-align: left; 
    padding-left: 25px; 
    min-height: 45px; 
}
QPushButton#BtnLoad:hover { background-color: #005f9e; }

/* Separar (Verde) */
QPushButton#BtnExtract { 
    background-color: #28a745; 
    text-align: left; 
    padding-left: 25px; 
    min-height: 45px; 
}
QPushButton#BtnExtract:hover { background-color: #1e7e34; }

/* Guardar (Morado) */
QPushButton#BtnSave { 
    background-color: #6f42c1; 
    text-align: left; 
    padding-left: 25px; 
    min-height: 45px; 
}
QPushButton#BtnSave:hover { background-color: #59359a; }

/* --- OTROS COMPONENTES --- */

/* Botón Regresar (Rojo) - Más grande */
QPushButton#BtnBack { 
    background-color: #d32f2f; 
    width: 60px; 
    height: 40px;
    font-size: 20px;
    border-radius: 20px; 
    padding: 0; /* Centrar texto */
}
QPushButton#BtnBack:hover { background-color: #b71c1c; }

/* Galería y Checkbox */
QWidget#GalleryItem { 
    background-color: #2d2d2d; 
    border-radius: 8px; 
    border: 1px solid #333; 
}

QCheckBox { 
    color: #e0e0e0; 
    font-size: 14px; 
    font-weight: bold; 
    spacing: 10px; 
    padding: 5px;
}
QCheckBox::indicator { 
    width: 20px; 
    height: 20px; 
}
"""