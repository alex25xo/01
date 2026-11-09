## Comparación en imágenes: Faster R-CNN (a) vs. YOLO (b)

Mismas 128 imágenes de coco128 y mismo protocolo de evaluación (pycocotools).

| Métrica | Faster R-CNN R50-FPN (a) | YOLOv8n (b) | Mejor |
|---|---|---|---|
| mAP@[.5:.95] (métrica principal) | 0.431 | 0.427 | ≈ empate |
| mAP@.5 | 0.666 | 0.580 | Faster R-CNN R50-FPN |
| mAP@.75 | 0.445 | 0.454 | YOLOv8n |
| AP objetos pequeños | 0.235 | 0.181 | Faster R-CNN R50-FPN |
| AP objetos medianos | 0.443 | 0.414 | Faster R-CNN R50-FPN |
| AP objetos grandes | 0.612 | 0.610 | ≈ empate |
| AR@100 | 0.529 | 0.494 | Faster R-CNN R50-FPN |
| Precisión (conf ≥ 0,5) | 0.670 | 0.892 | YOLOv8n |
| Recall (conf ≥ 0,5) | 0.643 | 0.357 | Faster R-CNN R50-FPN |
| F1 (conf ≥ 0,5) | 0.656 | 0.510 | Faster R-CNN R50-FPN |
| Mejor F1 | 0.661 | 0.609 | Faster R-CNN R50-FPN |
| Umbral del mejor F1 | 0.6 | 0.2 |  |
| Confianza media (conf ≥ 0,5) | 0.806 | 0.751 |  |
| Latencia media por imagen (ms) | 2172.8 | 69.1 | YOLOv8n |
| Latencia p95 (ms) | 2608.7 | 111.7 | YOLOv8n |
| FPS | 0.46 | 14.47 | YOLOv8n |

### AP por clase (clases con ≥ 10 objetos)

| Clase | Objetos | Faster R-CNN R50-FPN | YOLOv8n | Diferencia (a − b) |
|---|---|---|---|---|
| bottle | 18 | 0.432 | 0.237 | +0.196 |
| wine glass | 16 | 0.446 | 0.290 | +0.156 |
| bird | 16 | 0.612 | 0.476 | +0.136 |
| cup | 36 | 0.382 | 0.251 | +0.131 |
| spoon | 22 | 0.275 | 0.147 | +0.129 |
| handbag | 19 | 0.144 | 0.082 | +0.062 |
| traffic light | 14 | 0.200 | 0.139 | +0.061 |
| knife | 16 | 0.439 | 0.396 | +0.043 |
| teddy bear | 21 | 0.370 | 0.330 | +0.040 |
| book | 29 | 0.156 | 0.127 | +0.030 |
| chair | 35 | 0.261 | 0.233 | +0.028 |
| kite | 10 | 0.190 | 0.165 | +0.025 |
| car | 46 | 0.196 | 0.173 | +0.023 |
| bowl | 28 | 0.493 | 0.493 | +0.000 |
| person | 254 | 0.517 | 0.517 | +0.000 |
| broccoli | 11 | 0.207 | 0.212 | -0.005 |
| potted plant | 14 | 0.448 | 0.467 | -0.019 |
| elephant | 17 | 0.570 | 0.590 | -0.019 |
| truck | 12 | 0.258 | 0.291 | -0.032 |
| donut | 14 | 0.807 | 0.866 | -0.059 |
| dining table | 13 | 0.326 | 0.410 | -0.084 |
| carrot | 24 | 0.297 | 0.385 | -0.087 |
| umbrella | 18 | 0.297 | 0.392 | -0.096 |

*(Cuando existan los resultados de las partes c y d, aquí aparecerá la tabla imagen vs. video.)*
