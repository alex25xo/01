import argparse
import csv
import json                                  # [MÉTRICAS]
import time
from collections import Counter              # [MÉTRICAS]
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_320_fpn,
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
)


# ============================================================
# [MÉTRICAS] Bloque agregado para la parte (e)
# ------------------------------------------------------------
# No cambia la detección: solo anota lo que pasa en cada
# fotograma y al final guarda, en resultados/c_faster_rcnn_video/:
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


def elegir_fuente():
    """[MENÚ] Si no se pasa --video, pregunta como en la parte d: cámara o archivo."""
    print('============================================')
    print('   DETECCIÓN DE OBJETOS CON FASTER R-CNN')
    print('============================================')
    print('  1 = Cámara web (tiempo real)')
    print('  2 = Archivo de video (.mp4, .avi...)')
    print('============================================')
    opcion = input('Elige una opción [1/2]: ').strip()
    if opcion == '1':
        return '0'
    if opcion != '2':
        raise SystemExit('Opción no válida.')
    ruta = input('Ruta del video (Enter para buscarlo en una ventana): ').strip().strip('"')
    if not ruta:
        try:
            import tkinter as tk
            from tkinter import filedialog
            raiz = tk.Tk()
            raiz.withdraw()
            ruta = filedialog.askopenfilename(
                title='Elige un video',
                filetypes=[('Videos', '*.mp4 *.avi *.mov *.mkv *.wmv'), ('Todos los archivos', '*.*')])
            raiz.destroy()
        except Exception:
            print('No se pudo abrir la ventana para elegir el archivo.')
    if not ruta:
        raise SystemExit('No se indicó ninguna ruta.')
    if not Path(ruta).is_file():
        raise SystemExit(f'No existe el archivo: {ruta}')
    return ruta


def main():
    parser = argparse.ArgumentParser(description='Actividad 06 C - Faster R-CNN en video')
    parser.add_argument('--video', default=None, help='Indice de webcam (0) o ruta a video; sin esto aparece un menú')
    parser.add_argument('--conf', type=float, default=0.60, help='Umbral de confianza (0-1)')
    parser.add_argument('--width', type=int, default=640, help='Ancho de procesamiento')
    parser.add_argument('--guardar', action='store_true', help='Guardar video con detecciones')
    args = parser.parse_args()
    if not 0 <= args.conf <= 1:
        parser.error('--conf debe estar entre 0 y 1')
    if args.video is None:                                       # [MENÚ]
        args.video = elegir_fuente()                             # [MENÚ]
        umbral = input(f'Umbral de confianza (Enter = {args.conf}): ').strip().replace(',', '.')   # [MENÚ]
        if umbral:                                               # [MENÚ]
            args.conf = float(umbral)                            # [MENÚ]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Dispositivo: {device}')
    print('Cargando Faster R-CNN con pesos COCO (requiere internet la primera vez)...')
    weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights).to(device).eval()
    preprocess = weights.transforms()
    classes = weights.meta['categories']

    source = int(args.video) if args.video.isdigit() else args.video
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f'No se pudo abrir la fuente de video: {args.video}')

    carpeta = Path('resultados/c_faster_rcnn_video')           # [MÉTRICAS] carpeta propia de la parte c
    carpeta.mkdir(parents=True, exist_ok=True)
    fps_fuente = cap.get(cv2.CAP_PROP_FPS) or 30.0               # [MÉTRICAS] la webcam a veces informa 0
    registro = RegistroMetricas(carpeta, 'Faster R-CNN MobileNetV3-320', args.video, args.conf,  # [MÉTRICAS]
                                dispositivo=device.type, ancho_procesamiento=args.width)
    writer = None
    csv_file = open(carpeta / 'metricas.csv', 'w', newline='', encoding='utf-8')
    log = csv.writer(csv_file)
    log.writerow(['fotograma', 'latencia_ms', 'fps_procesamiento', 'num_detecciones'])
    frame_id, last_show = 0, time.perf_counter()
    print('Controles: Q = salir, S = guardar captura')

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_id += 1
            h, w = frame.shape[:2]
            target_w = min(args.width, w)
            target_h = max(1, round(h * target_w / w))
            frame = cv2.resize(frame, (target_w, target_h))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = preprocess(torch.from_numpy(rgb).permute(2, 0, 1)).to(device)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                prediction = model([tensor])[0]
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            latency_ms = elapsed * 1000
            fps_process = 1 / max(elapsed, 1e-9)

            scores = prediction['scores'].cpu().numpy()
            boxes = prediction['boxes'].cpu().numpy()
            labels = prediction['labels'].cpu().numpy()
            kept = 0
            cajas_ok, nombres_ok, conf_ok = [], [], []          # [MÉTRICAS]
            for box, label, score in zip(boxes, labels, scores):
                if score < args.conf:
                    continue
                kept += 1
                x1, y1, x2, y2 = np.asarray(box, dtype=int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 90), 2)
                name = classes[int(label)]
                cajas_ok.append(box)                             # [MÉTRICAS]
                nombres_ok.append(name)                          # [MÉTRICAS]
                conf_ok.append(score)                            # [MÉTRICAS]
                caption = f'{name}: {score:.0%}'
                cv2.putText(frame, caption, (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 220, 90), 2)
            registro.agregar(latency_ms, cajas_ok, nombres_ok, conf_ok)   # [MÉTRICAS]

            now = time.perf_counter()
            fps_display = 1 / max(now - last_show, 1e-9)
            last_show = now
            cv2.rectangle(frame, (0, 0), (min(410, target_w), 77), (20, 20, 20), -1)
            cv2.putText(frame, f'Faster R-CNN | {device.type.upper()}', (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, .59, (255, 255, 255), 2)
            cv2.putText(frame, f'Inferencia: {latency_ms:.1f} ms | FPS: {fps_display:.1f}', (8, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, .51, (255, 255, 255), 1)
            cv2.putText(frame, f'Objetos: {kept} | Conf: {args.conf:.0%}', (8, 66),
                        cv2.FONT_HERSHEY_SIMPLEX, .51, (255, 255, 255), 1)
            log.writerow([frame_id, round(latency_ms, 2), round(fps_display, 2), kept])

            if args.guardar:
                if writer is None:
                    writer = cv2.VideoWriter(str(carpeta / 'deteccion.mp4'),
                                             cv2.VideoWriter_fourcc(*'mp4v'), 15,
                                             (target_w, target_h))
                    if not writer.isOpened():
                        raise RuntimeError('No se pudo crear el video de salida')
                writer.write(frame)
            cv2.imshow('Actividad 06 C - Faster R-CNN', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('s'):
                file = str(carpeta / f'captura_{frame_id:05d}.jpg')
                cv2.imwrite(file, frame)
                print(f'Captura guardada: {file}')
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        csv_file.close()
        cv2.destroyAllWindows()
        registro.guardar(fps_fuente)                             # [MÉTRICAS]
    print(f'Listo. Metricas guardadas en {carpeta}')


if __name__ == '__main__':
    main()
