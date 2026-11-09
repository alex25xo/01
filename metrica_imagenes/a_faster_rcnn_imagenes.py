"""
PARTE (a) - Detección de objetos en IMÁGENES con Faster R-CNN
Responsable: Mestas

Archivo autocontenido: descarga lo que necesita, detecta objetos en imágenes,
guarda las imágenes anotadas y evalúa con las métricas del protocolo COCO.

    pip install -r requirements.txt
    python a_faster_rcnn_imagenes.py                        # evalúa en coco128 (128 imágenes)
    python a_faster_rcnn_imagenes.py --imagenes foto.jpg    # detecta en una imagen propia
    python a_faster_rcnn_imagenes.py --imagenes carpeta/    # o en una carpeta
    python a_faster_rcnn_imagenes.py --backend torchvision --dispositivo cuda   # GPU (Colab)

---------------------------------------------------------------------------
De R-CNN a Faster R-CNN
---------------------------------------------------------------------------
 R-CNN (2014)       Selective Search propone ~2000 regiones; CADA región se recorta
                    y pasa por una CNN -> muy lento (~50 s por imagen en CPU).
 Fast R-CNN (2015)  La CNN se aplica UNA vez a toda la imagen; las regiones se
                    recortan sobre el mapa de características (RoI Pooling).
                    Selective Search sigue siendo el cuello de botella.
 Faster R-CNN (2015) Reemplaza Selective Search por una Region Proposal Network
                    (RPN), una pequeña red que comparte las características con
                    el detector y propone regiones usando "anchors".

Faster R-CNN es un detector de DOS ETAPAS:
  imagen -> [backbone ResNet-50 + FPN] -> mapas de características
         -> ETAPA 1: RPN propone ~1000 regiones con "objeto / no objeto"
         -> RoIAlign recorta cada región a 7x7
         -> ETAPA 2: cabeza que clasifica (81 clases) y ajusta la caja
         -> NMS elimina cajas duplicadas -> detecciones finales
"""
import argparse
import contextlib
import io
import json
import math
import platform                              # [MÉTRICAS]
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parent

# --------------------------------------------------------------------------
# 1. Clases de COCO
# --------------------------------------------------------------------------
COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]
# IDs oficiales de COCO (1..90 con huecos); los usan pycocotools y torchvision
COCO80_A_ID91 = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27,
    28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53,
    54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80,
    81, 82, 84, 85, 86, 87, 88, 89, 90,
]
ID91_A_COCO80 = {cid: i for i, cid in enumerate(COCO80_A_ID91)}

# --------------------------------------------------------------------------
# 2. Descarga de datos y pesos (todo desde GitHub)
# --------------------------------------------------------------------------
URL_COCO128 = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
URL_ONNX = ("https://media.githubusercontent.com/media/onnx/models/main/validated/vision/"
            "object_detection_segmentation/faster-rcnn/model/FasterRCNN-12.onnx")


def descargar(url, destino):
    destino.parent.mkdir(parents=True, exist_ok=True)
    print(f"Descargando {destino.name} ...", flush=True)
    tmp = destino.with_suffix(destino.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    tmp.rename(destino)


def preparar_coco128():
    carpeta = RAIZ / "datos" / "coco128"
    if not carpeta.exists():
        zip_ = RAIZ / "datos" / "coco128.zip"
        descargar(URL_COCO128, zip_)
        with zipfile.ZipFile(zip_) as z:
            z.extractall(RAIZ / "datos")
        zip_.unlink()
    return carpeta / "images" / "train2017", carpeta / "labels" / "train2017"


# --------------------------------------------------------------------------
# 3. El detector Faster R-CNN
# --------------------------------------------------------------------------
class FasterRCNN:
    """Faster R-CNN ResNet-50 + FPN preentrenado en COCO.

    backend="onnx"        modelo del ONNX Model Zoo ejecutado con onnxruntime (por defecto)
    backend="torchvision" torchvision.models.detection.fasterrcnn_resnet50_fpn (PyTorch)
    Ambos son la misma arquitectura. __call__ devuelve (cajas xyxy, puntajes, clases 0..79).
    """
    nombre = "Faster R-CNN R50-FPN"
    MEDIA_BGR = np.array([102.9801, 115.9465, 122.7717], np.float32)  # medias del entrenamiento

    def __init__(self, backend="onnx", dispositivo="cpu", conf_min=0.05):
        self.backend, self.conf_min = backend, conf_min
        if backend == "onnx":
            import onnxruntime as ort
            ruta = RAIZ / "pesos" / "FasterRCNN-12.onnx"
            if not ruta.exists():
                descargar(URL_ONNX, ruta)  # 168 MB
            prov = ["CPUExecutionProvider"] if dispositivo == "cpu" else \
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self.sesion = ort.InferenceSession(str(ruta), providers=prov)
            self.entrada = self.sesion.get_inputs()[0].name
        elif backend == "torchvision":
            import torch
            from torchvision.models.detection import (FasterRCNN_ResNet50_FPN_Weights,
                                                      fasterrcnn_resnet50_fpn)
            self.torch, self.dev = torch, torch.device(dispositivo)
            self.modelo = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1,
                                                  box_score_thresh=conf_min).eval().to(self.dev)
        else:
            raise ValueError("backend debe ser 'onnx' o 'torchvision'")

    def __call__(self, img_bgr):
        if self.backend == "torchvision":
            return self._torchvision(img_bgr)
        alto, ancho = img_bgr.shape[:2]
        # (1) Reescalar: lado menor = 800 px (como en el entrenamiento)
        escala = 800.0 / min(alto, ancho)
        img = cv2.resize(img_bgr, (round(ancho * escala), round(alto * escala))).astype(np.float32)
        # (2) Normalizar: restar la media por canal; HWC -> CHW
        img = (img - self.MEDIA_BGR).transpose(2, 0, 1)
        # (3) Rellenar a múltiplo de 32 (cada nivel del FPN reduce a la mitad)
        h, w = math.ceil(img.shape[1] / 32) * 32, math.ceil(img.shape[2] / 32) * 32
        tensor = np.zeros((3, h, w), np.float32)
        tensor[:, :img.shape[1], :img.shape[2]] = img
        # (4) Backbone + RPN + RoIAlign + cabeza + NMS (dentro del grafo ONNX)
        cajas, etiquetas, puntajes = self.sesion.run(None, {self.entrada: tensor})
        # (5) Volver a las coordenadas de la imagen original
        cajas = cajas / escala
        cajas[:, [0, 2]] = cajas[:, [0, 2]].clip(0, ancho)
        cajas[:, [1, 3]] = cajas[:, [1, 3]].clip(0, alto)
        m = puntajes >= self.conf_min
        # el modelo numera 1..80 (0 = fondo) -> 0..79
        return cajas[m].astype(np.float32), puntajes[m].astype(np.float32), etiquetas[m].astype(int) - 1

    def _torchvision(self, img_bgr):
        torch = self.torch
        t = torch.from_numpy(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255
        with torch.inference_mode():
            out = self.modelo([t.to(self.dev)])[0]  # normaliza y reescala internamente
        etiq = out["labels"].cpu().numpy()
        ok = np.array([e in ID91_A_COCO80 for e in etiq], bool)
        return (out["boxes"].cpu().numpy()[ok].astype(np.float32),
                out["scores"].cpu().numpy()[ok].astype(np.float32),
                np.array([ID91_A_COCO80[e] for e in etiq[ok]], int))


# --------------------------------------------------------------------------
# 4. Dibujo de resultados
# --------------------------------------------------------------------------
def color_clase(c):
    import colorsys
    r, g, b = colorsys.hsv_to_rgb((c * 0.618034) % 1.0, 0.75, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)


def dibujar(img, cajas, puntajes, clases, titulo):
    out = img.copy()
    esc = max(0.4, min(out.shape[:2]) / 1100)
    for (x1, y1, x2, y2), s, c in zip(cajas.astype(int), puntajes, clases):
        col = color_clase(int(c))
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        txt = f"{COCO80[c]} {s:.2f}"
        (tw, th), base = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, esc, 1)
        yt = max(y1, th + base + 2)
        cv2.rectangle(out, (x1, yt - th - base - 2), (x1 + tw + 2, yt), col, -1)
        cv2.putText(out, txt, (x1 + 1, yt - base), cv2.FONT_HERSHEY_SIMPLEX, esc, (0, 0, 0), 1, cv2.LINE_AA)
    (tw, th), _ = cv2.getTextSize(titulo, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(out, (0, 0), (tw + 12, th + 14), (30, 30, 30), -1)
    cv2.putText(out, titulo, (6, th + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# --------------------------------------------------------------------------
# 5. Métricas
# --------------------------------------------------------------------------
def iou(a, b):
    """IoU entre cajas xyxy -> matriz (len(a), len(b))."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda c: (c[:, 2] - c[:, 0]) * (c[:, 3] - c[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def verdaderos_positivos(cajas_p, clases_p, cajas_g, clases_g, umbral=0.5):
    """Empareja 1 a 1 predicción-objeto real (misma clase, IoU >= umbral)."""
    m = iou(cajas_p, cajas_g)
    if m.size == 0:
        return 0
    m = np.where(clases_p[:, None] == clases_g[None, :], m, 0)
    tp = 0
    while True:
        i, j = np.unravel_index(np.argmax(m), m.shape)
        if m[i, j] < umbral:
            return tp
        tp += 1
        m[i, :] = 0
        m[:, j] = 0


def precision_recall(gts, preds, conf):
    tp = npred = ngt = 0
    for (cg, kg), (cp, sp, kp) in zip(gts, preds):
        keep = sp >= conf
        tp += verdaderos_positivos(cp[keep], kp[keep], cg, kg)
        npred += keep.sum()
        ngt += len(cg)
    p, r = (tp / npred if npred else 0.0), (tp / ngt if ngt else 0.0)
    return {"conf": conf, "precision": round(p, 4), "recall": round(r, 4),
            "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0,
            "verdaderos_positivos": int(tp), "predicciones": int(npred), "objetos_reales": int(ngt)}


def leer_etiquetas_yolo(rutas, dir_etiq):
    """Etiquetas YOLO (clase xc yc w h, normalizadas) -> dataset COCO + lista por imagen."""
    images, anns, gts = [], [], []
    for img_id, ruta in enumerate(rutas, 1):
        alto, ancho = cv2.imread(str(ruta)).shape[:2]
        images.append({"id": img_id, "file_name": ruta.name, "width": ancho, "height": alto})
        cajas, clases = [], []
        txt = dir_etiq / (ruta.stem + ".txt")
        for linea in (txt.read_text().split("\n") if txt.exists() else []):
            if not linea.strip():
                continue
            c, xc, yc, w, h = linea.split()[:5]
            w, h = float(w) * ancho, float(h) * alto
            x, y = float(xc) * ancho - w / 2, float(yc) * alto - h / 2
            anns.append({"id": len(anns) + 1, "image_id": img_id, "category_id": COCO80_A_ID91[int(c)],
                         "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0})
            cajas.append([x, y, x + w, y + h])
            clases.append(int(c))
        gts.append((np.array(cajas, np.float32).reshape(-1, 4), np.array(clases, int)))
    cats = [{"id": i, "name": n} for i, n in zip(COCO80_A_ID91, COCO80)]
    return {"images": images, "annotations": anns, "categories": cats}, gts


def evaluar_coco(gt, preds_coco):
    """mAP con pycocotools + AP por clase."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with contextlib.redirect_stdout(io.StringIO()):
        cg = COCO()
        cg.dataset = gt
        cg.createIndex()
        ev = COCOeval(cg, cg.loadRes(preds_coco), "bbox")
        ev.evaluate(); ev.accumulate(); ev.summarize()
    nombres = ["mAP_50_95", "mAP_50", "mAP_75", "AP_pequenos", "AP_medianos", "AP_grandes",
               "AR_1", "AR_10", "AR_100", "AR_pequenos", "AR_medianos", "AR_grandes"]
    resumen = {k: round(float(v), 4) for k, v in zip(nombres, ev.stats)}
    # precision[T, R, K, A, M]: umbral IoU, recall, clase, área (0 = todas), maxDets (2 = 100)
    prec = ev.eval["precision"]
    instancias = {}
    for a in gt["annotations"]:
        instancias[a["category_id"]] = instancias.get(a["category_id"], 0) + 1
    por_clase = []
    for k, cid in enumerate(ev.params.catIds):
        if cid not in instancias:
            continue
        p = prec[:, :, k, 0, 2]
        p50 = prec[0, :, k, 0, 2]
        por_clase.append({"clase": COCO80[ID91_A_COCO80[cid]], "objetos": instancias[cid],
                          "AP_50_95": round(float(p[p > -1].mean()), 4) if (p > -1).any() else 0.0,
                          "AP_50": round(float(p50[p50 > -1].mean()), 4) if (p50 > -1).any() else 0.0})
    return resumen, sorted(por_clase, key=lambda d: -d["objetos"])


# --------------------------------------------------------------------------
# 6. Gráficos (matplotlib)
# --------------------------------------------------------------------------
FONDO, TXT, TXT2, GRILLA = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
AZUL, NARANJA, VERDE = "#2a78d6", "#eb6834", "#1baf7a"


def _ejes(ax, titulo, ylabel):
    ax.set_facecolor(FONDO)
    ax.set_title(titulo, loc="left", color=TXT, fontsize=12, fontweight="bold", pad=10)
    ax.set_ylabel(ylabel, color=TXT2)
    ax.grid(axis="y", color=GRILLA, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#b5b4ad")
    ax.tick_params(colors=TXT2, length=0)


def graficos(resultado, salida):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # (a) Precisión, recall y F1 según el umbral de confianza
    b = resultado["barrido_umbral"]
    x = [d["conf"] for d in b]
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=FONDO)
    for clave, nombre, col in [("precision", "Precisión", AZUL), ("recall", "Recall", NARANJA),
                               ("f1", "F1", VERDE)]:
        y = [d[clave] for d in b]
        ax.plot(x, y, color=col, linewidth=2, marker="o", markersize=5, label=nombre)
        ax.annotate(nombre, (x[-1], y[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", color=TXT, fontsize=9)
    m = resultado["mejor_f1"]
    ax.scatter([m["conf"]], [m["f1"]], s=140, facecolor="none", edgecolor=TXT, linewidth=1.2, zorder=4)
    ax.annotate(f"mejor F1 = {m['f1']:.3f} (conf ≥ {m['conf']})", (m["conf"], m["f1"]),
                xytext=(m["conf"] - 0.2, 0.95), textcoords="data", ha="center", color=TXT, fontsize=9,
                arrowprops=dict(arrowstyle="-", color=TXT2, linewidth=0.8))
    ax.set_xlim(0.05, 1.02)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("umbral de confianza", color=TXT2)
    _ejes(ax, "Faster R-CNN: precisión vs. recall según el umbral", "valor (IoU ≥ 0,5)")
    ax.legend(frameon=False, loc="lower left", labelcolor=TXT)
    fig.tight_layout()
    fig.savefig(salida / "grafico_umbral.png", dpi=150)
    plt.close(fig)

    # (b) AP por clase (clases con al menos 10 objetos)
    pc = [d for d in resultado["ap_por_clase"] if d["objetos"] >= 10]
    pc.sort(key=lambda d: d["AP_50_95"])
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(pc) + 1.2), facecolor=FONDO)
    y = range(len(pc))
    ax.barh(y, [d["AP_50_95"] for d in pc], color=AZUL, height=0.6, edgecolor=FONDO, linewidth=2)
    for i, d in enumerate(pc):
        ax.annotate(f"{d['AP_50_95']:.2f}", (d["AP_50_95"], i), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8, color=TXT2)
    ax.set_yticks(list(y), [f"{d['clase']} ({d['objetos']})" for d in pc], color=TXT)
    ax.set_xlim(0, 1)
    ax.set_xlabel("AP@[.5:.95]", color=TXT2)
    ax.set_facecolor(FONDO)
    ax.set_title("AP por clase (clases con ≥ 10 objetos; entre paréntesis, nº de objetos)",
                 loc="left", color=TXT, fontsize=11, fontweight="bold", pad=10)
    ax.grid(axis="x", color=GRILLA, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#b5b4ad")
    ax.tick_params(colors=TXT2, length=0)
    fig.tight_layout()
    fig.savefig(salida / "grafico_ap_por_clase.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# 7. Programa principal
# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Parte (a): Faster R-CNN en imágenes")
    p.add_argument("--imagenes", default=None, help="imagen o carpeta (por defecto: coco128)")
    p.add_argument("--etiquetas", default=None, help="carpeta de etiquetas YOLO para evaluar")
    p.add_argument("--salida", default="resultados/a_faster_rcnn_imagenes")   # [MÉTRICAS] carpeta que lee la parte (e)
    p.add_argument("--backend", default="onnx", choices=["onnx", "torchvision"])
    p.add_argument("--dispositivo", default="cpu", help="cpu o cuda")
    p.add_argument("--conf", type=float, default=0.5, help="umbral para dibujar y para P/R")
    p.add_argument("--ejemplos", type=int, default=12, help="cuántas imágenes anotadas guardar (-1 = todas)")
    p.add_argument("--limite", type=int, default=None, help="procesar solo las primeras N imágenes")
    a = p.parse_args()

    if a.imagenes is None:  # por defecto: coco128 con sus etiquetas
        dir_img, dir_etq = preparar_coco128()
        a.imagenes, a.etiquetas = str(dir_img), a.etiquetas or str(dir_etq)
    ruta = Path(a.imagenes)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    rutas = [ruta] if ruta.is_file() else sorted(q for q in ruta.iterdir() if q.suffix.lower() in exts)
    rutas = rutas[:a.limite]
    salida = Path(a.salida)
    (salida / "ejemplos").mkdir(parents=True, exist_ok=True)

    detector = FasterRCNN(a.backend, a.dispositivo)
    primera = cv2.imread(str(rutas[0]))
    for _ in range(2):  # calentamiento: las primeras inferencias son más lentas
        detector(primera)

    preds, latencias = [], []
    n_ej = len(rutas) if a.ejemplos < 0 else a.ejemplos
    for i, r in enumerate(rutas):
        img = cv2.imread(str(r))
        t0 = time.perf_counter()
        cajas, puntajes, clases = detector(img)
        latencias.append((time.perf_counter() - t0) * 1000)
        preds.append((cajas, puntajes, clases))
        vis = puntajes >= a.conf
        if i < n_ej:
            cv2.imwrite(str(salida / "ejemplos" / r.name),
                        dibujar(img, cajas[vis], puntajes[vis], clases[vis],
                                f"Faster R-CNN | {latencias[-1]:.0f} ms | {vis.sum()} objetos"))
        print(f"[{i + 1}/{len(rutas)}] {r.name}: {vis.sum()} objetos en {latencias[-1]:.0f} ms", flush=True)

    lat = np.array(latencias)
    resultado = {
        "modelo": detector.nombre, "backend": a.backend, "imagenes": len(rutas),
        "latencia_media_ms": round(lat.mean(), 1), "latencia_mediana_ms": round(float(np.median(lat)), 1),
        "latencia_p95_ms": round(float(np.percentile(lat, 95)), 1), "fps": round(1000 / lat.mean(), 2),
        "dispositivo": a.dispositivo,                                          # [MÉTRICAS]
        "equipo": f"{platform.system()} | {platform.processor() or platform.machine()}",   # [MÉTRICAS]
    }
    conteo = {}
    for _, s, k in preds:
        for c in k[s >= a.conf]:
            conteo[COCO80[c]] = conteo.get(COCO80[c], 0) + 1
    resultado["objetos_detectados_por_clase"] = dict(sorted(conteo.items(), key=lambda kv: -kv[1]))

    if a.etiquetas and Path(a.etiquetas).exists():
        gt, gts = leer_etiquetas_yolo(rutas, Path(a.etiquetas))
        preds_coco = [{"image_id": i + 1, "category_id": COCO80_A_ID91[int(c)],
                       "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], "score": float(s)}
                      for i, (cj, sc, kl) in enumerate(preds)
                      for (x1, y1, x2, y2), s, c in zip(cj, sc, kl)]
        resultado["coco"], resultado["ap_por_clase"] = evaluar_coco(gt, preds_coco)
        resultado["umbral_conf"] = precision_recall(gts, preds, a.conf)
        resultado["barrido_umbral"] = [{k: v for k, v in precision_recall(gts, preds, round(0.1 * u, 1)).items()
                                        if k in ("conf", "precision", "recall", "f1")} for u in range(1, 10)]
        resultado["mejor_f1"] = max(resultado["barrido_umbral"], key=lambda d: d["f1"])
        todas = np.concatenate([s[s >= a.conf] for _, s, _ in preds])
        resultado["confianza_media"] = round(float(todas.mean()), 4) if todas.size else None
        try:
            graficos(resultado, salida)
        except ImportError:
            print("(instala matplotlib para generar los gráficos)")

    (salida / "metricas.json").write_text(json.dumps(resultado, indent=2, ensure_ascii=False))

    print(f"\n=== {resultado['modelo']} ({a.backend}) | {len(rutas)} imágenes ===")
    print(f"Latencia media {resultado['latencia_media_ms']} ms (p95 {resultado['latencia_p95_ms']} ms)"
          f" -> {resultado['fps']} FPS")
    if "coco" in resultado:
        c, u, m = resultado["coco"], resultado["umbral_conf"], resultado["mejor_f1"]
        print(f"mAP@[.5:.95] {c['mAP_50_95']:.3f} | mAP@.5 {c['mAP_50']:.3f} | mAP@.75 {c['mAP_75']:.3f}"
              f" | AR@100 {c['AR_100']:.3f}")
        print(f"AP pequeños / medianos / grandes: {c['AP_pequenos']:.3f} / {c['AP_medianos']:.3f}"
              f" / {c['AP_grandes']:.3f}")
        print(f"conf >= {a.conf}: P {u['precision']:.3f} | R {u['recall']:.3f} | F1 {u['f1']:.3f}"
              f"  ({u['verdaderos_positivos']} aciertos de {u['objetos_reales']} objetos)")
        print(f"Mejor F1 {m['f1']:.3f} con conf >= {m['conf']}")
    print(f"Resultados en {salida}/")


if __name__ == "__main__":
    main()
