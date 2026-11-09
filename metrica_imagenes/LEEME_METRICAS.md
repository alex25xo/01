# Métricas agregadas a las partes de video (c y d)

A `detector_faster_rcnn.py` (Yovani) y `d_deteccion_de_objetos_yolo.py` (Maquen) se les agregó un bloque de métricas. La detección **no cambió**: las líneas nuevas están marcadas con `# [MÉTRICAS]`. El único cambio en el código original es que la parte c ahora guarda todo en `resultados/c_faster_rcnn_video/` en vez de `resultados/`.

## Cómo ejecutarlo para que la comparación sea justa

Los dos tienen que procesar **el mismo video**, en la misma computadora y con el mismo umbral de confianza (YOLO usa 0,45):

```bash
python detector_faster_rcnn.py --video mi_video.mp4 --conf 0.45
python d_deteccion_de_objetos_yolo.py      # opción 2 y la ruta de mi_video.mp4
```

Al terminar el video, o al pulsar `q`, cada uno guarda:

| Archivo | Contenido |
|---|---|
| `resumen_metricas.json` | El resumen que usa Fermin |
| `metricas_por_fotograma.csv` | Una fila por fotograma: latencia, objetos, confianza, estabilidad |
| `grafico_metricas.png` | Latencia y objetos por fotograma |

Los archivos van en `resultados/c_faster_rcnn_video/` o en `resultados/d_yolo_video/`, según la parte.

Después, Fermin ejecuta `python e_comparar_video.py`, que junta las dos partes en `resultados/e_comparacion/comparacion_video.md` (tabla) y `comparacion_video.png` (gráfico).

## Qué significa cada métrica

| Métrica | Qué mide |
|---|---|
| Latencia media / p95 (ms) | Tiempo que tarda el modelo en un fotograma; p95 es el caso lento (el 95 % de los fotogramas tarda menos) |
| FPS del modelo | Fotogramas por segundo que procesa el modelo (1000 ÷ latencia media) |
| FPS del bucle completo | Lo mismo, pero contando también dibujar y mostrar la ventana |
| Factor de tiempo real | FPS del modelo ÷ FPS del video. Con 1 o más, alcanza a la cámara |
| Fotogramas perdidos en vivo | Estimación de cuántos fotogramas se saltaría con una cámara real |
| Objetos por fotograma | Cuántos objetos detecta, en promedio, en cada fotograma |
| Confianza media | Confianza promedio de las detecciones mostradas |
| Estabilidad temporal | Fracción de objetos que siguen detectados en el fotograma siguiente (misma clase, IoU ≥ 0,5). 1 = sin parpadeo |
| Variación del conteo | Cuánto cambia, en promedio, el número de objetos de un fotograma al siguiente |

Los 5 primeros fotogramas no se cuentan en los tiempos, porque incluyen la carga del modelo.

## Diferencias que Fermin debe mencionar

- La parte c usa **Faster R-CNN MobileNetV3-320**, una versión ligera. La parte a (imágenes) usa **Faster R-CNN ResNet-50 + FPN**. Son de la misma familia, pero no son el mismo modelo.
- La parte d usa **YOLOv8n** y solo detecta 10 clases (personas, vehículos y animales). Faster R-CNN detecta las 80 de COCO, así que "objetos por fotograma" no es del todo comparable.
- En video no hay respuestas correctas anotadas, así que no se puede calcular el mAP. Para comparar la calidad en imágenes se usa el mAP de las partes a y b; en video se usan la velocidad y la estabilidad.
