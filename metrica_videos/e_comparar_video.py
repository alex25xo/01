"""
PARTE (e) - Comparación de las métricas de video: Faster R-CNN (parte c) vs. YOLO (parte d)
Responsable: Fermin

Lee los resúmenes que guardan las partes c y d:
    resultados/c_faster_rcnn_video/resumen_metricas.json
    resultados/d_yolo_video/resumen_metricas.json
y genera:
    resultados/e_comparacion/comparacion_video.md    tabla lista para el README
    resultados/e_comparacion/comparacion_video.png   gráfico comparativo

Uso:  python e_comparar_video.py
"""
import json
from pathlib import Path

RUTAS = {
    'Faster R-CNN (parte c)': Path('resultados/c_faster_rcnn_video/resumen_metricas.json'),
    'YOLO (parte d)': Path('resultados/d_yolo_video/resumen_metricas.json'),
}
COLORES = {'Faster R-CNN (parte c)': '#2a78d6', 'YOLO (parte d)': '#eb6834'}
FILAS = [
    ('Modelo', 'modelo', '{}'),
    ('Fuente', 'fuente', '{}'),
    ('Dispositivo', 'dispositivo', '{}'),
    ('Umbral de confianza', 'umbral_confianza', '{}'),
    ('FPS de la fuente', 'fps_fuente', '{:.1f}'),
    ('Fotogramas procesados', 'fotogramas_procesados', '{}'),
    ('Latencia media (ms)', 'latencia_media_ms', '{:.1f}'),
    ('Latencia p95 (ms)', 'latencia_p95_ms', '{:.1f}'),
    ('FPS del modelo', 'fps_modelo', '{:.2f}'),
    ('FPS del bucle completo (con dibujo y ventana)', 'fps_bucle_completo', '{:.2f}'),
    ('Factor de tiempo real (≥ 1 alcanza a la fuente)', 'factor_tiempo_real', '{:.2f}'),
    ('Fotogramas perdidos en vivo (estimado)', 'fotogramas_perdidos_en_vivo_estimado_pct', '{:.1f} %'),
    ('Objetos por fotograma', 'detecciones_por_fotograma', '{:.2f}'),
    ('Confianza media', 'confianza_media', '{:.3f}'),
    ('Estabilidad temporal (1 = sin parpadeo)', 'estabilidad_temporal', '{:.3f}'),
    ('Variación del conteo entre fotogramas', 'variacion_conteo_media', '{:.2f}'),
]


def valor(r, clave, formato):
    v = r.get(clave)
    return '—' if v is None else formato.format(v)


def main():
    datos = {n: json.loads(p.read_text(encoding='utf-8')) for n, p in RUTAS.items() if p.exists()}
    for n, p in RUTAS.items():
        if n not in datos:
            print(f'Falta {p}: ejecuta primero esa parte con un video.')
    if not datos:
        return

    salida = Path('resultados/e_comparacion')
    salida.mkdir(parents=True, exist_ok=True)
    nombres = list(datos)
    lineas = ['| Métrica | ' + ' | '.join(nombres) + ' |', '|---|' + '---|' * len(nombres)]
    for etiqueta, clave, formato in FILAS:
        lineas.append(f'| {etiqueta} | ' + ' | '.join(valor(datos[n], clave, formato) for n in nombres) + ' |')

    # Avisos: la comparación solo es justa con la misma fuente y condiciones parecidas
    avisos = []
    if len({Path(d['fuente']).name for d in datos.values()}) > 1:
        avisos.append('Se usaron fuentes distintas: para comparar, ambos deben procesar el MISMO video.')
    if len({d['umbral_confianza'] for d in datos.values()}) > 1:
        avisos.append('Los umbrales de confianza son distintos: afecta a objetos por fotograma y confianza media.')
    if len({d.get('dispositivo') for d in datos.values()}) > 1:
        avisos.append('Se ejecutaron en dispositivos distintos (CPU/GPU): los tiempos no son comparables.')
    for d in datos.values():
        if d.get('clases_detectadas'):
            avisos.append(f"{d['modelo']} solo detecta {len(d['clases_detectadas'])} clases "
                          f"({', '.join(d['clases_detectadas'])}); el otro modelo detecta las 80 de COCO.")
    texto = '\n'.join(lineas) + ('\n\n' + '\n'.join(f'> Aviso: {a}' for a in avisos) if avisos else '') + '\n'
    (salida / 'comparacion_video.md').write_text(texto, encoding='utf-8')
    print(texto)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('Instala matplotlib para generar el gráfico.')
        return
    paneles = [('FPS del modelo', 'fps_modelo', '{:.1f}'),
               ('Latencia media (ms)', 'latencia_media_ms', '{:.0f}'),
               ('Estabilidad temporal', 'estabilidad_temporal', '{:.2f}')]
    fig, ejes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (titulo, clave, formato) in zip(ejes, paneles):
        vals = [datos[n].get(clave) or 0 for n in nombres]
        barras = ax.bar(range(len(nombres)), vals, color=[COLORES.get(n, '#8a8f98') for n in nombres], width=0.55)
        for b, v in zip(barras, vals):
            ax.annotate(formato.format(v), (b.get_x() + b.get_width() / 2, v), xytext=(0, 3),
                        textcoords='offset points', ha='center', fontsize=10)
        if clave == 'fps_modelo':
            fps = datos[nombres[0]]['fps_fuente']
            ax.axhline(fps, color='#52514e', linestyle='--', linewidth=1)
            ax.text(0.02, 0.97, f'- - - FPS de la fuente: {fps:.0f}', transform=ax.transAxes, va='top', fontsize=8.5, color='#52514e')
        ax.set_title(titulo, loc='left', fontsize=11, fontweight='bold')
        ax.set_xticks(range(len(nombres)), [n.split(' (')[0] for n in nombres])
        ax.set_ylim(0, max(vals + [datos[nombres[0]]['fps_fuente'] if clave == 'fps_modelo' else 0]) * 1.2 or 1)
        ax.grid(axis='y', color='#e4e3df')
        ax.set_axisbelow(True)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    fig.suptitle('Video: Faster R-CNN vs. YOLO', x=0.01, ha='left', fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(salida / 'comparacion_video.png', dpi=130)
    plt.close(fig)
    print(f'Tabla y gráfico en {salida}/')


if __name__ == '__main__':
    main()
