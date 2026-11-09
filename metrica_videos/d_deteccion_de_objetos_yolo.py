import csv                                   # [MÉTRICAS]
import json                                  # [MÉTRICAS]
import time                                  # [MÉTRICAS]
from collections import Counter              # [MÉTRICAS]
from pathlib import Path                     # [MÉTRICAS]

import cv2
import numpy as np                           # [MÉTRICAS]
import torch                                 # [MÉTRICAS]
from ultralytics import YOLO

# ============================================================
# 1. CONFIGURACIÓN
# ============================================================
# Modelo YOLOv8 nano (rápido, ideal para tiempo real en CPU)
modelo = YOLO('yolov8n.pt')

TITULO_VENTANA = 'Deteccion YOLO - Animales, Personas y Vehiculos'

# IDs de las clases de COCO que queremos detectar:
# 0=persona, 2=coche, 3=moto, 7=camión,
# 14=pájaro, 15=gato, 16=perro, 17=caballo, 18=oveja, 19=vaca
CLASES_INTERES = [0, 2, 3, 7, 14, 15, 16, 17, 18, 19]

# Nombres en español para mostrar en pantalla
NOMBRES_ES = {
    0: 'Persona',   2: 'Coche',     3: 'Moto',
    7: 'Camión',    14: 'Pájaro',   15: 'Gato',
    16: 'Perro',    17: 'Caballo',  18: 'Oveja',
    19: 'Vaca',
}

# Umbral de confianza (0.5 = 50%). Súbelo si hay falsos positivos.
UMBRAL_CONFIANZA = 0.45

registro = None   # [MÉTRICAS] se crea en procesar_video()

# ============================================================
# [MÉTRICAS] Bloque agregado para la parte (e)
# ------------------------------------------------------------
# No cambia la detección: solo anota lo que pasa en cada
# fotograma y al final guarda, en resultados/d_yolo_video/:
#   metricas_por_fotograma.csv   una fila por fotograma
#   resumen_metricas.json        el resumen que usa Fermin
#   grafico_metricas.png         latencia y objetos por fotograma
# ============================================================
def _iou(a, b):
    """IoU entre cada caja de a y cada caja de b (formato x1, y1, x2, y2)."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def _objetos_que_se_mantienen(cajas_a, nombres_a, cajas_b, nombres_b, umbral=0.5):
    """Cuántos objetos del fotograma anterior siguen en el actual (misma clase, IoU >= 0.5)."""
    if len(cajas_a) == 0 or len(cajas_b) == 0:
        return 0
    misma_clase = np.array(nombres_a)[:, None] == np.array(nombres_b)[None, :]
    m = _iou(cajas_a, cajas_b) * misma_clase
    pares = 0
    while m.max() >= umbral:
        i, j = np.unravel_index(np.argmax(m), m.shape)
        pares += 1
        m[i, :] = 0
        m[:, j] = 0
    return pares


class RegistroMetricas:
    """Guarda latencia y detecciones de cada fotograma y calcula el resumen."""

    def __init__(self, carpeta, modelo, fuente, umbral, calentamiento=5, **extra):
        self.carpeta = Path(carpeta)
        self.carpeta.mkdir(parents=True, exist_ok=True)
        self.info = {'modelo': modelo, 'fuente': str(fuente), 'umbral_confianza': umbral, **extra}
        self.calentamiento = calentamiento   # los primeros fotogramas son lentos (carga del modelo)
        self.filas, self.confianzas, self.tiempos = [], [], []
        self.por_clase = Counter()
        self.anterior = None

    def agregar(self, latencia_ms, cajas, nombres, confianzas):
        cajas = np.asarray(cajas, dtype=float).reshape(-1, 4)
        nombres = list(nombres)
        confianzas = [float(c) for c in confianzas]
        estabilidad, cambio = '', ''
        if self.anterior is not None:
            cajas_ant, nombres_ant = self.anterior
            mayor = max(len(nombres), len(nombres_ant))
            if mayor:
                estabilidad = round(_objetos_que_se_mantienen(cajas, nombres, cajas_ant, nombres_ant) / mayor, 4)
            cambio = abs(len(nombres) - len(nombres_ant))
        self.anterior = (cajas, nombres)
        self.confianzas += confianzas
        self.por_clase.update(nombres)
        self.tiempos.append(time.perf_counter())
        self.filas.append({
            'fotograma': len(self.filas) + 1,
            'latencia_ms': round(latencia_ms, 2),
            'fps_modelo': round(1000 / max(latencia_ms, 1e-6), 2),
            'num_detecciones': len(nombres),
            'confianza_media': round(float(np.mean(confianzas)), 4) if confianzas else '',
            'estabilidad': estabilidad,          # fracción de objetos que siguen del fotograma anterior
            'cambio_conteo': cambio,             # |objetos ahora - objetos antes|
            'clases': ' | '.join(nombres),
        })

    def guardar(self, fps_fuente):
        if not self.filas:
            print('[MÉTRICAS] No se procesó ningún fotograma.')
            return None
        with open(self.carpeta / 'metricas_por_fotograma.csv', 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(self.filas[0]))
            w.writeheader()
            w.writerows(self.filas)

        # Los tiempos se resumen sin los fotogramas de calentamiento
        usar = self.filas[self.calentamiento:] if len(self.filas) > self.calentamiento else self.filas
        lat = np.array([f['latencia_ms'] for f in usar])
        t = self.tiempos[len(self.filas) - len(usar):]
        fps_bucle = (len(t) - 1) / (t[-1] - t[0]) if len(t) > 1 and t[-1] > t[0] else None
        fps_modelo = 1000 / lat.mean()
        factor = fps_modelo / fps_fuente
        n_det = np.array([f['num_detecciones'] for f in self.filas])
        est = [f['estabilidad'] for f in self.filas if f['estabilidad'] != '']
        cam = [f['cambio_conteo'] for f in self.filas if f['cambio_conteo'] != '']
        resumen = {
            **self.info,
            'fps_fuente': round(fps_fuente, 2),
            'fotogramas_procesados': len(self.filas),
            'fotogramas_calentamiento_excluidos_del_tiempo': len(self.filas) - len(usar),
            'latencia_media_ms': round(float(lat.mean()), 1),
            'latencia_mediana_ms': round(float(np.median(lat)), 1),
            'latencia_p95_ms': round(float(np.percentile(lat, 95)), 1),
            'latencia_min_ms': round(float(lat.min()), 1),
            'latencia_max_ms': round(float(lat.max()), 1),
            'fps_modelo': round(fps_modelo, 2),
            'fps_bucle_completo': round(fps_bucle, 2) if fps_bucle else None,   # incluye dibujar y mostrar
            'factor_tiempo_real': round(factor, 3),                              # >= 1: alcanza el ritmo de la fuente
            'fotogramas_perdidos_en_vivo_estimado_pct': round(max(0.0, 1 - factor) * 100, 1),
            'detecciones_por_fotograma': round(float(n_det.mean()), 2),
            'max_detecciones_en_un_fotograma': int(n_det.max()),
            'confianza_media': round(float(np.mean(self.confianzas)), 4) if self.confianzas else None,
            'estabilidad_temporal': round(float(np.mean(est)), 4) if est else None,
            'variacion_conteo_media': round(float(np.mean(cam)), 3) if cam else None,
            'detecciones_por_clase': dict(self.por_clase.most_common()),
            'duracion_procesamiento_s': round(self.tiempos[-1] - self.tiempos[0], 1),
        }
        (self.carpeta / 'resumen_metricas.json').write_text(
            json.dumps(resumen, indent=2, ensure_ascii=False), encoding='utf-8')
        self._grafico(resumen, fps_fuente)

        print(f"\n=== [MÉTRICAS] {resumen['modelo']} | {resumen['fotogramas_procesados']} fotogramas ===")
        print(f"Latencia media {resumen['latencia_media_ms']} ms (p95 {resumen['latencia_p95_ms']} ms)"
              f" -> {resumen['fps_modelo']} FPS del modelo")
        print(f"Fuente a {resumen['fps_fuente']} FPS -> factor de tiempo real {resumen['factor_tiempo_real']}"
              f" (en vivo se perderían ~{resumen['fotogramas_perdidos_en_vivo_estimado_pct']} % de fotogramas)")
        print(f"Objetos por fotograma {resumen['detecciones_por_fotograma']} | confianza media "
              f"{resumen['confianza_media']} | estabilidad temporal {resumen['estabilidad_temporal']}")
        print(f"Resumen en {self.carpeta / 'resumen_metricas.json'}")
        return resumen

    def _grafico(self, resumen, fps_fuente):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print('[MÉTRICAS] Instala matplotlib para generar el gráfico.')
            return
        x = [f['fotograma'] for f in self.filas]
        lat = [f['latencia_ms'] for f in self.filas]
        limite = 1000 / fps_fuente
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        a1.plot(x, lat, color='#2a78d6', linewidth=1.2)
        a1.axhline(limite, color='#8a8f98', linestyle='--', linewidth=1)
        a1.text(x[-1], limite, f' límite para {fps_fuente:.0f} FPS ({limite:.0f} ms)',
                va='bottom', ha='right', fontsize=9, color='#52514e')
        if self.calentamiento and len(x) > self.calentamiento:
            a1.axvspan(0.5, self.calentamiento + 0.5, color='#b5b4ad', alpha=0.25, linewidth=0)
            a1.text(self.calentamiento + 1.5, limite * 0.08, 'calentamiento (no se cuenta en el promedio)',
                    fontsize=8.5, color='#52514e')
        a1.set_ylim(0, max(np.percentile(lat, 99) * 1.3, limite * 1.4))
        a1.set_ylabel('latencia (ms)')
        a1.set_title(f"{resumen['modelo']}: {resumen['latencia_media_ms']:.0f} ms por fotograma "
                     f"({resumen['fps_modelo']:.1f} FPS)", loc='left', fontsize=12, fontweight='bold')
        a2.plot(x, [f['num_detecciones'] for f in self.filas], color='#52514e', linewidth=1.2)
        a2.set_ylabel('objetos detectados')
        a2.set_ylim(bottom=0)
        a2.set_xlabel('fotograma')
        for ax in (a1, a2):
            ax.grid(axis='y', color='#e4e3df')
            for s in ('top', 'right'):
                ax.spines[s].set_visible(False)
        fig.tight_layout()
        fig.savefig(self.carpeta / 'grafico_metricas.png', dpi=130)
        plt.close(fig)
# ==================== fin del bloque agregado ====================


# ============================================================
# 2. FUNCIONES
# ============================================================
def detectar_y_anotar(frame):
    """Ejecuta YOLO sobre un frame y devuelve la imagen anotada
    (cajas + panel con la lista de objetos detectados)."""

    # --- Inferencia YOLO ---
    inicio = time.perf_counter()                                  # [MÉTRICAS]
    resultados = modelo(
        frame,
        conf=UMBRAL_CONFIANZA,   # umbral de confianza
        classes=CLASES_INTERES,  # solo estas clases
        verbose=False            # no imprimir logs en consola
    )
    latencia_ms = (time.perf_counter() - inicio) * 1000           # [MÉTRICAS]
    if registro is not None:                                      # [MÉTRICAS]
        cajas = resultados[0].boxes                               # [MÉTRICAS]
        nombres = [NOMBRES_ES.get(int(c), resultados[0].names[int(c)]) for c in cajas.cls]  # [MÉTRICAS]
        registro.agregar(latencia_ms, cajas.xyxy.cpu().numpy(), nombres,                   # [MÉTRICAS]
                         cajas.conf.cpu().numpy())                                         # [MÉTRICAS]

    # --- Dibujar cajas y etiquetas ---
    # .plot() devuelve el frame con las cajas, nombres y confianza
    frame_anotado = resultados[0].plot()

    # --- Panel con la lista de objetos detectados ---
    # Contar cuántas veces aparece cada clase
    conteo = {}
    for caja in resultados[0].boxes:
        clase_id = int(caja.cls[0])
        conteo[clase_id] = conteo.get(clase_id, 0) + 1

    alto_frame, ancho_frame = frame_anotado.shape[:2]
    lineas_panel = [f"Objetos: {len(resultados[0].boxes)}"]
    for clase_id, cantidad in sorted(conteo.items(),
                                     key=lambda x: -x[1]):
        nombre = NOMBRES_ES.get(clase_id,
                                resultados[0].names.get(clase_id, '?'))
        lineas_panel.append(f"  {nombre}: {cantidad}")
    if len(lineas_panel) == 1:
        lineas_panel.append("  (ninguno)")

    # Fondo semitransparente del panel (arriba a la derecha)
    ancho_panel = 220
    alto_panel = 30 + 28 * len(lineas_panel)
    x0 = ancho_frame - ancho_panel - 10
    overlay = frame_anotado.copy()
    cv2.rectangle(overlay, (x0, 10), (ancho_frame - 10, 10 + alto_panel),
                  (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, frame_anotado, 0.4, 0,
                    frame_anotado)
    cv2.rectangle(frame_anotado, (x0, 10), (ancho_frame - 10, 10 + alto_panel),
                  (0, 200, 255), 2)

    # Texto del panel
    for i, texto in enumerate(lineas_panel):
        cv2.putText(frame_anotado, texto,
                    (x0 + 10, 40 + i * 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255) if i == 0 else (180, 255, 180),
                    2 if i == 0 else 1)

    return frame_anotado


def procesar_imagen(ruta):
    """Detecta objetos en una foto/imagen y la muestra en pantalla."""
    frame = cv2.imread(ruta)
    if frame is None:
        print(f"❌ Error: no se pudo abrir la imagen '{ruta}'.")
        return

    # Si la imagen es muy grande para la pantalla, redimensionar
    alto, ancho = frame.shape[:2]
    escala = min(1.0, 1280 / ancho, 720 / alto)
    if escala < 1.0:
        frame = cv2.resize(frame, (int(ancho * escala), int(alto * escala)))

    frame_anotado = detectar_y_anotar(frame)
    cv2.imshow(TITULO_VENTANA, frame_anotado)
    print("✅ Imagen procesada. Pulsa cualquier tecla para salir.")
    cv2.waitKey(0)


def procesar_video(fuente):
    """Bucle de detección para cámara (0) o archivo de video."""
    global registro                                               # [MÉTRICAS]
    cap = cv2.VideoCapture(fuente)
    if not cap.isOpened():
        print("❌ Error: no se pudo abrir la fuente de video.")
        return
    fps_fuente = cap.get(cv2.CAP_PROP_FPS) or 30.0                # [MÉTRICAS] la webcam a veces informa 0
    registro = RegistroMetricas(                                  # [MÉTRICAS]
        'resultados/d_yolo_video', 'YOLOv8n', fuente, UMBRAL_CONFIANZA,
        dispositivo='cuda' if torch.cuda.is_available() else 'cpu',
        clases_detectadas=[NOMBRES_ES[c] for c in CLASES_INTERES])

    # (Opcional) Guardar el resultado en un archivo .mp4
    guardar_video = False          # Cambia a True para grabar la salida
    if guardar_video:
        ancho  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        alto   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        salida = cv2.VideoWriter('salida_deteccion.mp4',
                                 cv2.VideoWriter_fourcc(*'mp4v'),
                                 fps, (ancho, alto))

    print("✅ Detección iniciada. Pulsa 'q' o ESC para salir.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("ℹ️ Fin del video.")
            break

        frame_anotado = detectar_y_anotar(frame)

        # --- Mostrar resultado ---
        cv2.imshow(TITULO_VENTANA, frame_anotado)

        # --- Guardar frame si se solicitó ---
        if guardar_video:
            salida.write(frame_anotado)

        # Salir con la tecla 'q', ESC o cerrando la ventana
        tecla = cv2.waitKey(1) & 0xFF
        ventana_cerrada = cv2.getWindowProperty(
            TITULO_VENTANA, cv2.WND_PROP_VISIBLE) < 1
        if tecla == ord('q') or tecla == 27 or ventana_cerrada:
            break

    cap.release()
    if guardar_video:
        salida.release()
    cv2.destroyAllWindows()
    registro.guardar(fps_fuente)                                  # [MÉTRICAS]
    print("🔚 Programa finalizado.")


# ============================================================
# 3. MENÚ DE SELECCIÓN DE FUENTE
# ============================================================
print("============================================")
print("   DETECCIÓN DE OBJETOS CON YOLO")
print("============================================")
print("  1 = Cámara web (tiempo real)")
print("  2 = Archivo de video (.mp4, .avi...)")
print("  3 = Imagen / foto (.jpg, .png...)")
print("============================================")

opcion = input("Elige una opción [1/2/3]: ").strip()

if opcion == '1':
    procesar_video(0)
elif opcion == '2':
    ruta = input("Ruta del video: ").strip().strip('"')
    if ruta:
        procesar_video(ruta)
    else:
        print("❌ No se indicó ninguna ruta.")
elif opcion == '3':
    ruta = input("Ruta de la imagen: ").strip().strip('"')
    if ruta:
        procesar_imagen(ruta)
    else:
        print("❌ No se indicó ninguna ruta.")
else:
    print("❌ Opción no válida.")
