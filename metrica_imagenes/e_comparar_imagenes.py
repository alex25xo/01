"""
PARTE (e) - Comparación en IMÁGENES: Faster R-CNN (parte a) vs. YOLO (parte b)
            y, si ya existen, comparación IMAGEN vs. VIDEO con las partes c y d.
Responsable: Fermin

Lee:
    resultados/a_faster_rcnn_imagenes/metricas.json        <- python a_faster_rcnn_imagenes.py
    resultados/b_yolo_imagenes/metricas.json               <- python b_yolo_imagenes.py --metricas
    resultados/c_faster_rcnn_video/resumen_metricas.json   (opcional, parte c)
    resultados/d_yolo_video/resumen_metricas.json          (opcional, parte d)
Genera en resultados/e_comparacion/:
    comparacion_imagenes.md          tablas listas para el README
    comparacion_imagenes.png         métricas de precisión + latencia
    curva_precision_recall.png       precisión vs. recall según el umbral de confianza
    ap_por_clase_comparado.png       AP de cada clase en los dos modelos

Uso:  python e_comparar_imagenes.py
"""
import json
from pathlib import Path

RES = Path("resultados")
SALIDA = RES / "e_comparacion"
AZUL, NARANJA, GRIS, TXT, TXT2, GRILLA = "#2a78d6", "#eb6834", "#b5b4ad", "#0b0b0b", "#52514e", "#e4e3df"


def cargar(ruta):
    ruta = RES / ruta
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else None


def f(v, formato="{:.3f}"):
    return "—" if v is None else formato.format(v)


def tabla_imagenes(a, b):
    """Tabla a vs. b con una columna que dice qué modelo gana en cada métrica."""
    filas = [  # (etiqueta, función que extrae el valor, formato, mayor es mejor?)
        ("mAP@[.5:.95] (métrica principal)", lambda r: r["coco"]["mAP_50_95"], "{:.3f}", True),
        ("mAP@.5", lambda r: r["coco"]["mAP_50"], "{:.3f}", True),
        ("mAP@.75", lambda r: r["coco"]["mAP_75"], "{:.3f}", True),
        ("AP objetos pequeños", lambda r: r["coco"]["AP_pequenos"], "{:.3f}", True),
        ("AP objetos medianos", lambda r: r["coco"]["AP_medianos"], "{:.3f}", True),
        ("AP objetos grandes", lambda r: r["coco"]["AP_grandes"], "{:.3f}", True),
        ("AR@100", lambda r: r["coco"]["AR_100"], "{:.3f}", True),
        ("Precisión (conf ≥ 0,5)", lambda r: r["umbral_conf"]["precision"], "{:.3f}", True),
        ("Recall (conf ≥ 0,5)", lambda r: r["umbral_conf"]["recall"], "{:.3f}", True),
        ("F1 (conf ≥ 0,5)", lambda r: r["umbral_conf"]["f1"], "{:.3f}", True),
        ("Mejor F1", lambda r: r["mejor_f1"]["f1"], "{:.3f}", True),
        ("Umbral del mejor F1", lambda r: r["mejor_f1"]["conf"], "{}", None),
        ("Confianza media (conf ≥ 0,5)", lambda r: r.get("confianza_media"), "{:.3f}", None),
        ("Latencia media por imagen (ms)", lambda r: r["latencia_media_ms"], "{:.1f}", False),
        ("Latencia p95 (ms)", lambda r: r["latencia_p95_ms"], "{:.1f}", False),
        ("FPS", lambda r: r["fps"], "{:.2f}", True),
    ]
    n_a, n_b = a["modelo"], b["modelo"]
    L = [f"| Métrica | {n_a} (a) | {n_b} (b) | Mejor |", "|---|---|---|---|"]
    for etiqueta, sacar, formato, mayor in filas:
        va, vb = sacar(a), sacar(b)
        gana = ""
        if mayor is not None and va is not None and vb is not None:
            if abs(va - vb) <= 0.005 * max(1, abs(va), abs(vb)):   # diferencia mínima: no hay ganador
                gana = "≈ empate"
            else:
                gana = n_a if (va > vb) == mayor else n_b
        L.append(f"| {etiqueta} | {f(va, formato)} | {f(vb, formato)} | {gana} |")
    return "\n".join(L)


def tabla_clases(a, b, minimo=10):
    pa = {d["clase"]: d for d in a["ap_por_clase"]}
    pb = {d["clase"]: d for d in b["ap_por_clase"]}
    clases = [c for c in pa if c in pb and pa[c]["objetos"] >= minimo]
    clases.sort(key=lambda c: pa[c]["AP_50_95"] - pb[c]["AP_50_95"], reverse=True)
    L = [f"| Clase | Objetos | {a['modelo']} | {b['modelo']} | Diferencia (a − b) |", "|---|---|---|---|---|"]
    for c in clases:
        d = pa[c]["AP_50_95"] - pb[c]["AP_50_95"]
        L.append(f"| {c} | {pa[c]['objetos']} | {pa[c]['AP_50_95']:.3f} | {pb[c]['AP_50_95']:.3f} | {d:+.3f} |")
    return "\n".join(L), clases, pa, pb


def tabla_imagen_video(a, b, c, d):
    """Mismo tipo de modelo en imágenes y en video."""
    pares = [("Faster R-CNN", a, c), ("YOLO", b, d)]
    L = ["| Familia | Modelo en imágenes | Modelo en video | Latencia por imagen (ms) | Latencia por fotograma (ms) "
         "| FPS imagen | FPS video | Confianza media imagen | Confianza media video "
         "| Objetos por imagen | Objetos por fotograma | Estabilidad temporal (video) | Factor tiempo real (video) |",
         "|---" * 13 + "|"]
    avisos = []
    for familia, img, vid in pares:
        if img is None or vid is None:
            continue
        obj_img = sum(img.get("objetos_detectados_por_clase", {}).values()) / max(1, img["imagenes"])
        L.append(f"| {familia} | {img['modelo']} | {vid['modelo']} | {img['latencia_media_ms']:.1f} "
                 f"| {vid['latencia_media_ms']:.1f} | {img['fps']:.2f} | {vid['fps_modelo']:.2f} "
                 f"| {f(img.get('confianza_media'))} | {f(vid.get('confianza_media'))} | {obj_img:.2f} "
                 f"| {vid['detecciones_por_fotograma']:.2f} | {f(vid.get('estabilidad_temporal'))} "
                 f"| {vid['factor_tiempo_real']:.2f} |")
        if img["modelo"] != vid["modelo"]:
            avisos.append(f"{familia}: en imágenes se usó {img['modelo']} y en video {vid['modelo']}; "
                          "las diferencias de tiempo también se deben a que no es el mismo modelo.")
        if str(img.get("dispositivo", "")).startswith("cuda") != str(vid.get("dispositivo", "")).startswith("cuda"):
            avisos.append(f"{familia}: imágenes y video se ejecutaron en dispositivos distintos (CPU/GPU).")
    if len(L) == 2:
        return None, []
    return "\n".join(L), avisos


def graficos(a, b, clases, pa, pb):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n_a, n_b = a["modelo"], b["modelo"]

    def limpiar(ax, eje="y"):
        ax.grid(axis=eje, color=GRILLA, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=TXT2, length=0)

    # 1) Métricas de precisión (izquierda) y latencia (derecha): escalas distintas, paneles distintos
    metricas = [("mAP\n[.5:.95]", "mAP_50_95"), ("mAP@.5", "mAP_50"), ("mAP@.75", "mAP_75"),
                ("AP\npequeños", "AP_pequenos"), ("AP\nmedianos", "AP_medianos"), ("AP\ngrandes", "AP_grandes")]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [3.2, 1]})
    ancho = 0.38
    for k, (r, col, nom) in enumerate([(a, AZUL, n_a), (b, NARANJA, n_b)]):
        xs = [i + (k - 0.5) * ancho for i in range(len(metricas) + 1)]
        vals = [r["coco"][c] for _, c in metricas] + [r["mejor_f1"]["f1"]]
        barras = ax1.bar(xs, vals, ancho, color=col, edgecolor="white", linewidth=1.5, label=nom)
        for bar, v in zip(barras, vals):
            ax1.annotate(f"{v:.2f}", (bar.get_x() + bar.get_width() / 2, v), xytext=(0, 3),
                         textcoords="offset points", ha="center", fontsize=8, color=TXT2)
    ax1.set_xticks(range(len(metricas) + 1), [m for m, _ in metricas] + ["Mejor\nF1"], color=TXT, fontsize=9)
    ax1.set_ylim(0, 1)
    ax1.set_title("Precisión en coco128 (más alto = mejor)", loc="left", fontsize=12, fontweight="bold")
    ax1.legend(frameon=False, loc="upper left", ncol=2)
    limpiar(ax1)
    lat = [a["latencia_media_ms"], b["latencia_media_ms"]]
    barras = ax2.bar([0, 1], lat, 0.55, color=[AZUL, NARANJA])
    for bar, v in zip(barras, lat):
        ax2.annotate(f"{v:.0f} ms", (bar.get_x() + bar.get_width() / 2, v), xytext=(0, 3),
                     textcoords="offset points", ha="center", fontsize=9, color=TXT)
    ax2.set_xticks([0, 1], [" ".join(n_a.split()[:2]), " ".join(n_b.split()[:2])], fontsize=9)
    ax2.set_ylim(0, max(lat) * 1.18)
    veces = max(lat) / min(lat)
    ax2.set_title(f"Tiempo por imagen\n({veces:.0f} veces de diferencia)", loc="left", fontsize=12, fontweight="bold")
    limpiar(ax2)
    fig.tight_layout()
    fig.savefig(SALIDA / "comparacion_imagenes.png", dpi=140)
    plt.close(fig)

    # 2) Curva precisión-recall a partir del barrido de umbrales (0,1 ... 0,9)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for r, col, nom, lado in [(a, AZUL, n_a, 1), (b, NARANJA, n_b, -1)]:
        bar = r["barrido_umbral"]
        ax.plot([d["recall"] for d in bar], [d["precision"] for d in bar], color=col, linewidth=2,
                marker="o", markersize=5, label=nom)
        ax.annotate(f"conf {bar[0]['conf']}", (bar[0]["recall"], bar[0]["precision"]), xytext=(6, -12),
                    textcoords="offset points", fontsize=8, color=TXT2)
        ax.annotate(f"conf {bar[-1]['conf']}", (bar[-1]["recall"], bar[-1]["precision"]), xytext=(0, 8),
                    textcoords="offset points", fontsize=8, color=TXT2, ha="center")
        m = r["mejor_f1"]
        mp = next(d for d in bar if d["conf"] == m["conf"])
        ax.scatter([mp["recall"]], [mp["precision"]], s=150, facecolor="none", edgecolor=TXT, linewidth=1.2, zorder=4)
        ax.annotate(f"mejor F1 {m['f1']:.3f} (conf {m['conf']})", (mp["recall"], mp["precision"]),
                    xytext=(10 * lado, 6 if lado > 0 else -16), textcoords="offset points", fontsize=9,
                    color=col, fontweight="bold", ha="left" if lado > 0 else "right")
    ax.set_xlabel("recall (fracción de objetos encontrados)", color=TXT2)
    ax.set_ylabel("precisión (fracción de cajas correctas)", color=TXT2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.08)
    ax.set_title("Precisión vs. recall al variar el umbral de confianza", loc="left", fontsize=12, fontweight="bold")
    ax.legend(frameon=False, loc="lower left")
    limpiar(ax, "both")
    fig.tight_layout()
    fig.savefig(SALIDA / "curva_precision_recall.png", dpi=140)
    plt.close(fig)

    # 3) AP por clase: un punto por modelo y una línea entre ambos
    if clases:
        orden = sorted(clases, key=lambda c: pa[c]["AP_50_95"] - pb[c]["AP_50_95"])
        fig, ax = plt.subplots(figsize=(8, 0.38 * len(orden) + 1.4))
        for i, c in enumerate(orden):
            va, vb = pa[c]["AP_50_95"], pb[c]["AP_50_95"]
            ax.plot([va, vb], [i, i], color=GRIS, linewidth=2, zorder=1)
            ax.scatter([va], [i], color=AZUL, s=40, zorder=2, label=n_a if i == 0 else None)
            ax.scatter([vb], [i], color=NARANJA, s=40, zorder=2, label=n_b if i == 0 else None)
        ax.set_yticks(range(len(orden)), [f"{c} ({pa[c]['objetos']})" for c in orden], fontsize=9, color=TXT)
        ax.set_xlim(0, 1)
        ax.set_xlabel("AP@[.5:.95]", color=TXT2)
        ax.set_title("AP por clase (clases con ≥ 10 objetos)\narriba gana Faster R-CNN, abajo gana YOLO",
                     loc="left", fontsize=11, fontweight="bold")
        ax.legend(frameon=False, loc="lower right", fontsize=9)
        limpiar(ax, "x")
        fig.tight_layout()
        fig.savefig(SALIDA / "ap_por_clase_comparado.png", dpi=140)
        plt.close(fig)


def main():
    a = cargar("a_faster_rcnn_imagenes/metricas.json")
    b = cargar("b_yolo_imagenes/metricas.json")
    if a is None or b is None or "coco" not in a or "coco" not in b:
        print("Faltan resultados. Ejecuta primero:")
        print("  python a_faster_rcnn_imagenes.py")
        print("  python b_yolo_imagenes.py --metricas")
        return
    SALIDA.mkdir(parents=True, exist_ok=True)

    avisos = []
    if a.get("equipo") != b.get("equipo") or a.get("dispositivo") != b.get("dispositivo"):
        avisos.append("a y b se ejecutaron en computadoras o dispositivos distintos "
                      f"({a.get('equipo', '¿?')} / {b.get('equipo', '¿?')}): las latencias NO son comparables. "
                      "Para comparar tiempos, ejecuta ambas partes en la misma PC.")
    if a["imagenes"] != b["imagenes"]:
        avisos.append(f"Se evaluaron cantidades distintas de imágenes ({a['imagenes']} vs. {b['imagenes']}).")

    md = ["## Comparación en imágenes: Faster R-CNN (a) vs. YOLO (b)", "",
          f"Mismas {a['imagenes']} imágenes de coco128 y mismo protocolo de evaluación (pycocotools).", "",
          tabla_imagenes(a, b), ""]
    t_clases, clases, pa, pb = tabla_clases(a, b)
    md += ["### AP por clase (clases con ≥ 10 objetos)", "", t_clases, ""]

    c = cargar("c_faster_rcnn_video/resumen_metricas.json")
    d = cargar("d_yolo_video/resumen_metricas.json")
    t_iv, avisos_iv = tabla_imagen_video(a, b, c, d)
    if t_iv:
        md += ["## Imagen vs. video", "", t_iv, ""]
        avisos += avisos_iv
    else:
        md += ["*(Cuando existan los resultados de las partes c y d, aquí aparecerá la tabla imagen vs. video.)*", ""]
    if avisos:
        md += [f"> Aviso: {x}" for x in avisos] + [""]

    texto = "\n".join(md)
    (SALIDA / "comparacion_imagenes.md").write_text(texto, encoding="utf-8")
    print(texto)
    try:
        graficos(a, b, clases, pa, pb)
        print(f"Gráficos guardados en {SALIDA}/")
    except ImportError:
        print("Instala matplotlib para generar los gráficos.")


if __name__ == "__main__":
    main()
