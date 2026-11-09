"""
Parte b) Detección de objetos con YOLO en imágenes
Autor: Maligno
Uso:
    pip install ultralytics opencv-python pandas
    python yolo_imagenes.py --fuente imagenes/ --modelo yolo11n.pt --map
"""
import argparse
import contextlib                  # [MÉTRICAS]
import io                          # [MÉTRICAS]
import json                        # [MÉTRICAS]
import platform                    # [MÉTRICAS]
import shutil                      # [MÉTRICAS]
import time
import urllib.request              # [MÉTRICAS]
import zipfile                     # [MÉTRICAS]
from pathlib import Path

import cv2
import numpy as np                 # [MÉTRICAS]
import pandas as pd
from ultralytics import YOLO

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def detectar(modelo, carpeta, salida, conf, imgsz=640):
    salida.mkdir(parents=True, exist_ok=True)
    filas = []
    imagenes = [p for p in sorted(Path(carpeta).iterdir()) if p.suffix.lower() in EXT]
    if not imagenes:
        raise SystemExit(f"No se encontraron imágenes en {carpeta}")

    # Calentamiento (warmup)
    modelo.predict(imagenes[0], imgsz=imgsz, verbose=False)

    for ruta in imagenes:
        t0 = time.perf_counter()
        r = modelo.predict(ruta, conf=conf, imgsz=imgsz, verbose=False)[0]
        ms = (time.perf_counter() - t0) * 1000

        confs = r.boxes.conf.tolist()
        clases = [modelo.names[int(c)] for c in r.boxes.cls.tolist()]
        
        # Guardar imagen con cajas anotadas
        cv2.imwrite(str(salida / ruta.name), r.plot())

        filas.append({
            "imagen": ruta.name,
            "tiempo_ms": round(ms, 2),
            "n_detecciones": len(confs),
            "conf_media": round(sum(confs) / len(confs), 3) if confs else 0.0,
            "clases": ", ".join(sorted(set(clases))),
        })
        print(f"{ruta.name}: {len(confs)} objetos, {ms:.1f} ms")

    df = pd.DataFrame(filas)
    
    # Guardar en raíz o en carpeta de salida (se guarda en salida por orden)
    csv_path = salida / "resultados_yolo_imagenes.csv"
    df.to_csv(csv_path, index=False)
    
    print("\n=== Resumen ===")
    print(f"Tiempo medio por imagen: {df.tiempo_ms.mean():.2f} ms")
    print(f"FPS equivalente:         {1000 / df.tiempo_ms.mean():.2f}")
    print(f"Detecciones promedio:    {df.n_detecciones.mean():.2f}")
    print(f"Confianza media:         {df.conf_media.mean():.3f}")
    print(f"Resultados guardados en: {csv_path}")


def evaluar_map(modelo, imgsz=640):
    """mAP sobre COCO128 (se descarga solo, ~7 MB)."""
    print("\nEvaluando mAP en COCO128...")
    m = modelo.val(data="coco128.yaml", imgsz=imgsz, verbose=False)
    print("\n=== Métricas de precisión (COCO128) ===")
    print(f"Precision:    {m.box.mp:.3f}")
    print(f"Recall:       {m.box.mr:.3f}")
    print(f"mAP@0.50:     {m.box.map50:.3f}")
    print(f"mAP@0.5:0.95: {m.box.map:.3f}")


# ============================================================
# [MÉTRICAS] Bloque agregado para la parte (e)
# ------------------------------------------------------------
# evaluar_map() usa el método de Ultralytics y no guarda nada, así que no
# se puede comparar con la parte (a). Este bloque evalúa con EXACTAMENTE
# el mismo protocolo que usó Mestas en la parte (a):
#   - las mismas 128 imágenes de coco128 con sus etiquetas
#   - pycocotools (protocolo oficial COCO), confianza mínima 0.05, máx. 100 cajas
#   - precisión / recall / F1 con IoU >= 0.5 y confianza >= 0.5
# y guarda resultados/b_yolo_imagenes/metricas.json con el mismo formato.
# ============================================================
URL_COCO128 = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
# IDs oficiales de COCO (1..90 con huecos) de las 80 clases, en el orden que usa YOLO
COCO80_A_ID91 = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27,
    28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53,
    54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80,
    81, 82, 84, 85, 86, 87, 88, 89, 90,
]
ID91_A_COCO80 = {cid: i for i, cid in enumerate(COCO80_A_ID91)}


def _preparar_coco128():
    """Usa (o descarga) datos/coco128, la misma carpeta que la parte (a)."""
    carpeta = Path(__file__).resolve().parent / "datos" / "coco128"
    if not carpeta.exists():
        zip_ = carpeta.parent / "coco128.zip"
        zip_.parent.mkdir(parents=True, exist_ok=True)
        print("Descargando coco128 ...")
        with urllib.request.urlopen(URL_COCO128) as r, open(zip_, "wb") as f:
            shutil.copyfileobj(r, f)
        with zipfile.ZipFile(zip_) as z:
            z.extractall(carpeta.parent)
        zip_.unlink()
    return carpeta / "images" / "train2017", carpeta / "labels" / "train2017"


def _iou(a, b):
    """IoU entre cajas xyxy -> matriz (len(a), len(b))."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda c: (c[:, 2] - c[:, 0]) * (c[:, 3] - c[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def _aciertos(cajas_p, clases_p, cajas_g, clases_g, umbral=0.5):
    """Empareja 1 a 1 predicción-objeto real (misma clase, IoU >= umbral)."""
    m = _iou(cajas_p, cajas_g)
    if m.size == 0:
        return 0
    m = np.where(clases_p[:, None] == clases_g[None, :], m, 0)
    tp = 0
    while m.max() >= umbral:
        i, j = np.unravel_index(np.argmax(m), m.shape)
        tp += 1
        m[i, :] = 0
        m[:, j] = 0
    return tp


def _precision_recall(gts, preds, conf):
    tp = npred = ngt = 0
    for (cg, kg), (cp, sp, kp) in zip(gts, preds):
        keep = sp >= conf
        tp += _aciertos(cp[keep], kp[keep], cg, kg)
        npred += int(keep.sum())
        ngt += len(cg)
    p, r = (tp / npred if npred else 0.0), (tp / ngt if ngt else 0.0)
    return {"conf": conf, "precision": round(p, 4), "recall": round(r, 4),
            "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0,
            "verdaderos_positivos": tp, "predicciones": npred, "objetos_reales": ngt}


def _etiquetas_coco(rutas, dir_etq, nombres):
    """Etiquetas YOLO (clase xc yc w h, normalizadas) -> dataset COCO + cajas por imagen."""
    images, anns, gts = [], [], []
    for img_id, ruta in enumerate(rutas, 1):
        alto, ancho = cv2.imread(str(ruta)).shape[:2]
        images.append({"id": img_id, "file_name": ruta.name, "width": ancho, "height": alto})
        cajas, clases = [], []
        txt = dir_etq / (ruta.stem + ".txt")
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
    cats = [{"id": cid, "name": nombres[i]} for i, cid in enumerate(COCO80_A_ID91)]
    return {"images": images, "annotations": anns, "categories": cats}, gts


def _evaluar_coco(gt, preds_coco, nombres):
    """mAP con pycocotools (igual que la parte a) + AP por clase."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with contextlib.redirect_stdout(io.StringIO()):
        cg = COCO()
        cg.dataset = gt
        cg.createIndex()
        ev = COCOeval(cg, cg.loadRes(preds_coco), "bbox")
        ev.evaluate(); ev.accumulate(); ev.summarize()
    claves = ["mAP_50_95", "mAP_50", "mAP_75", "AP_pequenos", "AP_medianos", "AP_grandes",
              "AR_1", "AR_10", "AR_100", "AR_pequenos", "AR_medianos", "AR_grandes"]
    resumen = {k: round(float(v), 4) for k, v in zip(claves, ev.stats)}
    prec = ev.eval["precision"]          # [IoU, recall, clase, área, maxDets]
    instancias = {}
    for a in gt["annotations"]:
        instancias[a["category_id"]] = instancias.get(a["category_id"], 0) + 1
    por_clase = []
    for k, cid in enumerate(ev.params.catIds):
        if cid not in instancias:
            continue
        p, p50 = prec[:, :, k, 0, 2], prec[0, :, k, 0, 2]
        por_clase.append({"clase": nombres[ID91_A_COCO80[cid]], "objetos": instancias[cid],
                          "AP_50_95": round(float(p[p > -1].mean()), 4) if (p > -1).any() else 0.0,
                          "AP_50": round(float(p50[p50 > -1].mean()), 4) if (p50 > -1).any() else 0.0})
    return resumen, sorted(por_clase, key=lambda d: -d["objetos"])


def _color_clase(c):
    import colorsys
    r, g, b = colorsys.hsv_to_rgb((c * 0.618034) % 1.0, 0.75, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)


def _dibujar(img, cajas, puntajes, clases, nombres, titulo):
    """Mismo estilo de dibujo que la parte (a), para comparar las imágenes lado a lado."""
    out = img.copy()
    esc = max(0.4, min(out.shape[:2]) / 1100)
    for (x1, y1, x2, y2), s, c in zip(cajas.astype(int), puntajes, clases):
        col = _color_clase(int(c))
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        txt = f"{nombres[c]} {s:.2f}"
        (tw, th), base = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, esc, 1)
        yt = max(y1, th + base + 2)
        cv2.rectangle(out, (x1, yt - th - base - 2), (x1 + tw + 2, yt), col, -1)
        cv2.putText(out, txt, (x1 + 1, yt - base), cv2.FONT_HERSHEY_SIMPLEX, esc, (0, 0, 0), 1, cv2.LINE_AA)
    (tw, th), _ = cv2.getTextSize(titulo, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(out, (0, 0), (tw + 12, th + 14), (30, 30, 30), -1)
    cv2.putText(out, titulo, (6, th + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def evaluar_como_parte_a(modelo, pesos, imgsz=640, conf_pr=0.5, salida="resultados/b_yolo_imagenes", n_ejemplos=12):
    dir_img, dir_etq = _preparar_coco128()
    rutas = sorted(p for p in dir_img.iterdir() if p.suffix.lower() in EXT)
    nombres = [modelo.names[i] for i in range(80)]
    primera = cv2.imread(str(rutas[0]))
    for _ in range(2):                    # calentamiento, igual que en la parte (a)
        modelo.predict(primera, imgsz=imgsz, verbose=False)

    print(f"\nEvaluando {len(rutas)} imágenes de coco128 con el protocolo de la parte (a)...")
    (Path(salida) / "ejemplos").mkdir(parents=True, exist_ok=True)
    nombre_modelo = Path(pesos).stem.replace("yolo", "YOLO")
    preds, latencias = [], []
    for i, ruta in enumerate(rutas):
        img = cv2.imread(str(ruta))      # se lee antes de medir, como en la parte (a)
        t0 = time.perf_counter()
        r = modelo.predict(img, conf=0.05, imgsz=imgsz, max_det=100, verbose=False)[0]
        latencias.append((time.perf_counter() - t0) * 1000)
        preds.append((r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)))
        cajas, puntajes, clases = preds[-1]
        vis = puntajes >= conf_pr                # se dibuja con el mismo umbral que la parte (a)
        if i < n_ejemplos:
            cv2.imwrite(str(Path(salida) / "ejemplos" / ruta.name),
                        _dibujar(img, cajas[vis], puntajes[vis], clases[vis], nombres,
                                 f"{nombre_modelo} | {latencias[-1]:.0f} ms | {vis.sum()} objetos"))
        print(f"[{i + 1}/{len(rutas)}] {ruta.name}: {vis.sum()} objetos en {latencias[-1]:.0f} ms", flush=True)

    gt, gts = _etiquetas_coco(rutas, dir_etq, nombres)
    preds_coco = [{"image_id": i + 1, "category_id": COCO80_A_ID91[int(c)],
                   "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], "score": float(s)}
                  for i, (cj, sc, kl) in enumerate(preds) for (x1, y1, x2, y2), s, c in zip(cj, sc, kl)]
    coco, ap_por_clase = _evaluar_coco(gt, preds_coco, nombres)
    lat = np.array(latencias)
    conteo = {}
    for _, s, k in preds:
        for c in k[s >= conf_pr]:
            conteo[nombres[c]] = conteo.get(nombres[c], 0) + 1
    barrido = [{k: v for k, v in _precision_recall(gts, preds, round(0.1 * u, 1)).items()
                if k in ("conf", "precision", "recall", "f1")} for u in range(1, 10)]
    todas = np.concatenate([s[s >= conf_pr] for _, s, _ in preds])
    resultado = {
        "modelo": nombre_modelo, "backend": "ultralytics", "imagenes": len(rutas),
        "latencia_media_ms": round(float(lat.mean()), 1), "latencia_mediana_ms": round(float(np.median(lat)), 1),
        "latencia_p95_ms": round(float(np.percentile(lat, 95)), 1), "fps": round(1000 / float(lat.mean()), 2),
        "dispositivo": str(modelo.device),
        "equipo": f"{platform.system()} | {platform.processor() or platform.machine()}",
        "objetos_detectados_por_clase": dict(sorted(conteo.items(), key=lambda kv: -kv[1])),
        "coco": coco, "ap_por_clase": ap_por_clase,
        "umbral_conf": _precision_recall(gts, preds, conf_pr),
        "barrido_umbral": barrido, "mejor_f1": max(barrido, key=lambda d: d["f1"]),
        "confianza_media": round(float(todas.mean()), 4) if todas.size else None,
    }
    (Path(salida) / "metricas.json").write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")

    c, u, m = coco, resultado["umbral_conf"], resultado["mejor_f1"]
    print(f"\n=== {resultado['modelo']} | {len(rutas)} imágenes (mismo protocolo que la parte a) ===")
    print(f"Latencia media {resultado['latencia_media_ms']} ms -> {resultado['fps']} FPS")
    print(f"mAP@[.5:.95] {c['mAP_50_95']:.3f} | mAP@.5 {c['mAP_50']:.3f} | mAP@.75 {c['mAP_75']:.3f} | AR@100 {c['AR_100']:.3f}")
    print(f"AP pequeños / medianos / grandes: {c['AP_pequenos']:.3f} / {c['AP_medianos']:.3f} / {c['AP_grandes']:.3f}")
    print(f"conf >= {conf_pr}: P {u['precision']:.3f} | R {u['recall']:.3f} | F1 {u['f1']:.3f}"
          f"  ({u['verdaderos_positivos']} aciertos de {u['objetos_reales']} objetos)")
    print(f"Mejor F1 {m['f1']:.3f} con conf >= {m['conf']}")
    print(f"Guardado en {Path(salida)}/ (metricas.json y {min(n_ejemplos, len(rutas))} imágenes en ejemplos/)")
# ==================== fin del bloque agregado ====================


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fuente", default="imagenes", help="Carpeta de imágenes de entrada")
    ap.add_argument("--modelo", default="yolo11n.pt", help="Pesos de YOLO")
    ap.add_argument("--conf", type=float, default=0.25, help="Umbral de confianza")
    ap.add_argument("--imgsz", type=int, default=640, help="Resolución de inferencia")
    ap.add_argument("--salida", default="salida_yolo_imagenes", help="Carpeta de salida")
    ap.add_argument("--map", action="store_true", help="Calcular mAP en COCO128")
    ap.add_argument("--metricas", action="store_true",                       # [MÉTRICAS]
                    help="Evaluar con el mismo protocolo que la parte (a) y guardar metricas.json")
    a = ap.parse_args()

    yolo = YOLO(a.modelo)
    hay_fotos = Path(a.fuente).is_dir() and any(p.suffix.lower() in EXT for p in Path(a.fuente).iterdir())  # [MÉTRICAS]
    if hay_fotos or not a.metricas:                 # [MÉTRICAS] con --metricas no hace falta carpeta propia
        detectar(yolo, a.fuente, Path(a.salida), a.conf, a.imgsz)
    if a.map:
        evaluar_map(yolo, a.imgsz)
    if a.metricas:                                  # [MÉTRICAS]
        evaluar_como_parte_a(yolo, a.modelo, a.imgsz)