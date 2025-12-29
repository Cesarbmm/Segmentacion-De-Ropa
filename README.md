# 👔 AI Fashion Studio - Ultimate Edition

Sistema avanzado de segmentación de prendas en tiempo real con modelado 3D interactivo. Desarrollado con Python, PyTorch y PyQt6.

## 🚀 Características

* **Análisis Estático:** Carga de imágenes, segmentación semántica y detección de color dominante (Hex).
* **Live Cam (Beta):** Segmentación en tiempo real usando webcam con superposición AR.
* **Digital Twin 3D:** Proyección de texturas detectadas sobre un maniquí 3D interactivo usando PyVista.
* **Arquitectura Modular:** Código organizado en controladores, vistas y motor de IA.

## 🛠️ Tecnologías

* **GUI:** PyQt6 (Modern Dark Theme)
* **IA:** PyTorch (DeepLabV3+ con ResNet34)
* **3D:** PyVista & PyVistaQt
* **Visión:** OpenCV & Albumentations

## 🧠 Modelo Pre-entrenado

Para utilizar la segmentación, descarga el modelo y colócalo en la carpeta raíz del proyecto:
* **[Descargar best_model_mejorado.pth](https://drive.google.com/file/d/1ca3oewDWoXFFRxhnJkD506YYdFEAPvr7/view?usp=sharing)**

## 📦 Instalación

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/Cesarbmm/Segmentacion-De-Ropa.git
   ```

2. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

3. Ejecutar la aplicación:
   ```bash
   python main.py
   ```

---

Desarrollado por **César Zapata**.
