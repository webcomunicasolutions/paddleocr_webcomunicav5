# Despliegue en EasyPanel

Guia para desplegar PaddleOCR WebComunica v5 en EasyPanel.

## 1. Crear App desde GitHub

| Campo | Valor |
|-------|-------|
| **Repository** | `https://github.com/webcomunicasolutions/paddleocr_webcomunicav5` |
| **Branch** | `main` |
| **Dockerfile** | `Dockerfile` |

## 2. Puerto

| Puerto Interno | Descripcion |
|----------------|-------------|
| `8503` | Puerto de la aplicacion Flask |

En EasyPanel, configura el dominio/proxy para apuntar al puerto `8503`.

## 3. Variables de Entorno

Copia todas estas variables en la seccion "Environment" de EasyPanel:

```env
# Flask
TZ=Europe/Madrid
FLASK_ENV=production
FLASK_PORT=8503

# OpenCV - Deteccion de documentos
OPENCV_HSV_LOWER_H=0
OPENCV_HSV_LOWER_S=0
OPENCV_HSV_LOWER_V=140
OPENCV_HSV_UPPER_H=180
OPENCV_HSV_UPPER_S=60
OPENCV_HSV_UPPER_V=255
OPENCV_MIN_AREA_PERCENT=0.15
OPENCV_MAX_AREA_PERCENT=0.95
OPENCV_EPSILON_FACTOR=0.01
OPENCV_ERODE_ITERATIONS=1
OPENCV_DILATE_ITERATIONS=2
OPENCV_MIN_WIDTH=300
OPENCV_MIN_HEIGHT=400
OPENCV_EROSION_PERCENT=0.085
OPENCV_INNER_SCALE_FACTOR=1.06

# Rotacion automatica
ROTATION_MIN_CONFIDENCE=0.7
ROTATION_MIN_SKEW_ANGLE=0.2

# OCR - Configuracion PaddleOCR
OCR_LANG=es
OCR_VERSION=PP-OCRv3
OCR_TEXT_DETECTION_MODEL_NAME=PP-OCRv3_mobile_det
OCR_TEXT_RECOGNITION_MODEL_NAME=latin_PP-OCRv3_mobile_rec
OCR_USE_DOC_ORIENTATION=false
OCR_USE_DOC_UNWARPING=false
OCR_USE_TEXTLINE_ORIENTATION=true
OCR_TEXT_DET_THRESH=0.05
OCR_TEXT_DET_BOX_THRESH=0.2
OCR_TEXT_DET_UNCLIP_RATIO=1.5
OCR_TEXT_DET_LIMIT_SIDE_LEN=960
OCR_TEXT_DET_LIMIT_TYPE=min
OCR_TEXT_RECOGNITION_BATCH_SIZE=6
OCR_ENHANCE_LEVEL=none

# Optimizacion CPU
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
FLAGS_allocator_strategy=auto_growth
FLAGS_fraction_of_gpu_memory_to_use=0
CUDA_VISIBLE_DEVICES=
```

## 4. Volumenes

Configura estos volumenes para persistir los modelos (evita re-descarga en cada reinicio):

| Mount Path | Nombre Volumen | Descripcion |
|------------|----------------|-------------|
| `/home/n8n/.paddlex` | `paddlex-models` | Cache modelos PaddleX |
| `/home/n8n/.paddleocr` | `paddleocr-models` | Cache modelos OCR |

**Opcional** (solo si usas integracion n8n):
| `/home/n8n` | `n8n-data` | Directorio de trabajo n8n |

## 5. Recursos Minimos

| Recurso | Minimo | Recomendado |
|---------|--------|-------------|
| **Memoria RAM** | 4 GB | 6 GB |
| **CPU** | 2 cores | 4 cores |

> **Importante**: El servidor debe tener CPU con soporte AVX/AVX2. No funciona en VPS basicos sin estas instrucciones.

## 6. Healthcheck (opcional)

Si EasyPanel soporta healthchecks:

| Campo | Valor |
|-------|-------|
| **Path** | `/health` |
| **Port** | `8503` |
| **Interval** | `30s` |
| **Timeout** | `15s` |
| **Start Period** | `60s` |

## 7. Verificacion

Una vez desplegado, verifica que funciona:

```bash
# Health check
curl https://tu-dominio.easypanel.host/health

# Procesar documento
curl -X POST https://tu-dominio.easypanel.host/process \
  -F "file=@factura.pdf" \
  -F "format=layout"
```

Respuesta esperada del health:
```json
{
  "status": "healthy",
  "ocr_ready": true,
  "preprocessor_ready": true
}
```

## 8. Primer Inicio

El primer inicio tarda ~2 minutos porque descarga los modelos OCR (~500MB).
Los siguientes inicios son rapidos porque los modelos estan en cache (volumenes).

## Troubleshooting

### Error "Illegal instruction"
Tu servidor no tiene CPU con AVX. Necesitas un VPS con CPU moderna.

### Error "Out of memory"
Aumenta la memoria RAM a minimo 4GB.

### Modelos se descargan en cada reinicio
Verifica que los volumenes estan correctamente montados en `/home/n8n/.paddlex` y `/home/n8n/.paddleocr`.

### Timeout en procesamiento
Documentos muy grandes pueden tardar >60s. Aumenta el timeout del proxy en EasyPanel.

---

## Endpoints Disponibles

| Endpoint | Metodo | Descripcion |
|----------|--------|-------------|
| `/` | GET | Dashboard web para probar OCR |
| `/health` | GET | Estado del servidor |
| `/stats` | GET | Estadisticas de uso |
| `/process` | POST | Procesar documento (principal) |
| `/ocr` | POST | Compatibilidad n8n |

---

*PaddleOCR WebComunica v5 - OCR minimalista con deteccion de tablas*
