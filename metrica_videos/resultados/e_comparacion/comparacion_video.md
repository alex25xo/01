| Métrica | Faster R-CNN (parte c) | YOLO (parte d) |
|---|---|---|
| Modelo | Faster R-CNN MobileNetV3-320 | YOLOv8n |
| Fuente | C:\Users\Usuario\Videos\pruebavideo\11.mp4 | C:\Users\Usuario\Videos\pruebavideo\11.mp4 |
| Dispositivo | cpu | cpu |
| Umbral de confianza | 0.6 | 0.45 |
| FPS de la fuente | 30.0 | 30.0 |
| Fotogramas procesados | 1304 | 1304 |
| Latencia media (ms) | 120.0 | 53.2 |
| Latencia p95 (ms) | 137.7 | 63.2 |
| FPS del modelo | 8.33 | 18.81 |
| FPS del bucle completo (con dibujo y ventana) | 7.59 | 15.51 |
| Factor de tiempo real (≥ 1 alcanza a la fuente) | 0.28 | 0.63 |
| Fotogramas perdidos en vivo (estimado) | 72.2 % | 37.3 % |
| Objetos por fotograma | 8.71 | 9.73 |
| Confianza media | 0.856 | 0.692 |
| Estabilidad temporal (1 = sin parpadeo) | 0.831 | 0.854 |
| Variación del conteo entre fotogramas | 0.94 | 0.81 |

> Aviso: Los umbrales de confianza son distintos: afecta a objetos por fotograma y confianza media.
> Aviso: YOLOv8n solo detecta 10 clases (Persona, Coche, Moto, Camión, Pájaro, Gato, Perro, Caballo, Oveja, Vaca); el otro modelo detecta las 80 de COCO.
