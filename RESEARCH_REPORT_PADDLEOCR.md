# Informe de Investigacion: PaddleOCR - Analisis y Mejoras Recomendadas

**Fecha:** 8 de Diciembre de 2025
**Proyecto:** paddleocr_v5_paco_base
**Investigador:** Claude Code (Opus 4.5)

---

## 1. ESTADO ACTUAL DEL PROYECTO

### 1.1 Imagen Docker Base
```
FROM ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/paddlex:paddlex3.0.1-paddlepaddle3.0.0-cpu
```

**Versiones instaladas:**
- PaddleX: 3.0.1
- PaddlePaddle: 3.0.0
- PaddleOCR: Instalado via pip (version no especificada)

### 1.2 Modelos Configurados Actualmente

| Componente | Modelo Actual | Version |
|------------|---------------|---------|
| **Deteccion de texto** | `PP-OCRv3_mobile_det` | PP-OCRv3 |
| **Reconocimiento de texto** | `latin_PP-OCRv3_mobile_rec` | PP-OCRv3 |
| **Orientacion de documento** | `PP-LCNet_x1_0_doc_ori` | - |
| **Version OCR** | `PP-OCRv3` | v3 |
| **Idioma** | `es` (espanol) | - |

### 1.3 Parametros de Deteccion Actuales (docker-compose.yml)

| Parametro | Valor Actual | Descripcion |
|-----------|--------------|-------------|
| `OCR_TEXT_DET_THRESH` | 0.25 | Umbral de deteccion de pixeles |
| `OCR_TEXT_DET_BOX_THRESH` | 0.4 | Umbral de confianza de cajas |
| `OCR_TEXT_DET_UNCLIP_RATIO` | 2.0 | Ratio de expansion de cajas |
| `OCR_TEXT_DET_LIMIT_SIDE_LEN` | 960 | Limite de lado de imagen |
| `OCR_TEXT_DET_LIMIT_TYPE` | min | Tipo de limite |
| `OCR_TEXT_RECOGNITION_BATCH_SIZE` | 6 | Tamano de batch |

---

## 2. MEJORAS DISPONIBLES

### 2.1 Nuevas Versiones de PaddleOCR (Mayo 2025)

**PaddleOCR 3.0** fue lanzado el **20 de Mayo de 2025**, incluyendo:

1. **PP-OCRv5**: Nueva generacion de reconocimiento de texto
   - **+13 puntos porcentuales** de mejora sobre PP-OCRv4
   - Soporte para **106 idiomas** (incluido espanol)
   - **+30% mejora** en reconocimiento multilingue vs PP-OCRv3

2. **PP-StructureV3**: Parsing jerarquico de documentos
   - Conversion inteligente a Markdown/JSON
   - Supera soluciones comerciales en benchmarks

3. **PP-ChatOCRv4**: Extraccion de informacion clave

### 2.2 Comparativa de Modelos de Deteccion

| Modelo | Precision (Hmean) | Tamano | Recomendacion |
|--------|-------------------|--------|---------------|
| PP-OCRv3_mobile_det | ~60% | ~3 MB | **ACTUAL** |
| PP-OCRv4_mobile_det | 63.8% | 4.7 MB | Mejora leve |
| PP-OCRv4_server_det | 69.2% | 109 MB | Mejor precision |
| **PP-OCRv5_mobile_det** | **79.0%** | **4.7 MB** | **RECOMENDADO** |
| PP-OCRv5_server_det | 83.8% | 84.3 MB | Maximo rendimiento |

### 2.3 Comparativa de Modelos de Reconocimiento (Latin/Espanol)

| Modelo | Precision | Tamano | Idiomas | Recomendacion |
|--------|-----------|--------|---------|---------------|
| latin_PP-OCRv3_mobile_rec | 76.93% | 8.7 MB | Latin | **ACTUAL** |
| **latin_PP-OCRv5_mobile_rec** | **84.7%** | **14 MB** | Latin | **RECOMENDADO** |
| PP-OCRv5_server_rec | ~90%+ | Mayor | Multi | Alto rendimiento |

**Mejora esperada: +7.77 puntos porcentuales (+10% relativo)**

### 2.4 Nueva Imagen Docker Disponible

| Version | PaddleX | PaddlePaddle | Estado |
|---------|---------|--------------|--------|
| Actual | 3.0.1 | 3.0.0 | En uso |
| **v3.2.0** | 3.2.0 | 3.1.x | **Estable** |
| v3.3.x | 3.3.x | 3.1.x | Ultima (Nov 2025) |

**Nota:** PaddleX 3.2.0+ soporta CUDA 12 y tiene mejoras de rendimiento significativas.

---

## 3. PARAMETROS OPTIMOS RECOMENDADOS

### 3.1 Para Documentos Escaneados (Facturas A4)

Basado en la documentacion oficial y mejores practicas:

| Parametro | Valor Actual | Valor Recomendado | Justificacion |
|-----------|--------------|-------------------|---------------|
| `text_det_thresh` | 0.25 | **0.25-0.3** | OK, equilibrio deteccion/ruido |
| `text_det_box_thresh` | 0.4 | **0.5-0.6** | Subir para reducir falsos positivos |
| `text_det_unclip_ratio` | 2.0 | **1.75** | Reducir para cajas mas ajustadas |
| `text_det_limit_side_len` | 960 | **1216-1536** | Aumentar para documentos A4 300dpi |
| `text_det_limit_type` | min | **min** | OK para documentos |

### 3.2 Configuracion Recomendada Completa

```yaml
# docker-compose.yml - CONFIGURACION OPTIMIZADA v5.2

# Variables OCR - PP-OCRv5 OPTIMIZADO
- OCR_LANG=es
- OCR_VERSION=PP-OCRv5

# Modelos PP-OCRv5 (MEJORA PRINCIPAL)
- OCR_TEXT_DETECTION_MODEL_NAME=PP-OCRv5_mobile_det
- OCR_TEXT_RECOGNITION_MODEL_NAME=latin_PP-OCRv5_mobile_rec

# Orientacion y preprocesamiento
- OCR_USE_DOC_ORIENTATION=false
- OCR_USE_DOC_UNWARPING=false
- OCR_USE_TEXTLINE_ORIENTATION=true

# Parametros de deteccion optimizados para facturas escaneadas
- OCR_TEXT_DET_THRESH=0.25
- OCR_TEXT_DET_BOX_THRESH=0.5
- OCR_TEXT_DET_UNCLIP_RATIO=1.75
- OCR_TEXT_DET_LIMIT_SIDE_LEN=1216
- OCR_TEXT_DET_LIMIT_TYPE=min

# Batch size
- OCR_TEXT_RECOGNITION_BATCH_SIZE=8
```

---

## 4. CAMBIOS ESPECIFICOS A IMPLEMENTAR

### 4.1 Opcion A: Solo Cambiar Modelos (Minimo Riesgo)

Cambiar en `docker-compose.yml`:

```yaml
# ANTES (PP-OCRv3)
- OCR_VERSION=PP-OCRv3
- OCR_TEXT_DETECTION_MODEL_NAME=PP-OCRv3_mobile_det
- OCR_TEXT_RECOGNITION_MODEL_NAME=latin_PP-OCRv3_mobile_rec

# DESPUES (PP-OCRv5)
- OCR_VERSION=PP-OCRv5
- OCR_TEXT_DETECTION_MODEL_NAME=PP-OCRv5_mobile_det
- OCR_TEXT_RECOGNITION_MODEL_NAME=latin_PP-OCRv5_mobile_rec
```

**Beneficio esperado:** +10-15% precision en reconocimiento

### 4.2 Opcion B: Cambiar Modelos + Parametros

Ademas de los modelos, ajustar:

```yaml
# Parametros optimizados para facturas
- OCR_TEXT_DET_BOX_THRESH=0.5          # Era 0.4
- OCR_TEXT_DET_UNCLIP_RATIO=1.75       # Era 2.0
- OCR_TEXT_DET_LIMIT_SIDE_LEN=1216     # Era 960
```

### 4.3 Opcion C: Actualizacion Completa (Maximo Rendimiento)

#### Paso 1: Actualizar Dockerfile
```dockerfile
# ANTES
FROM ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/paddlex:paddlex3.0.1-paddlepaddle3.0.0-cpu

# DESPUES (PaddleX 3.2.0 con PaddlePaddle 3.1.0)
# Nota: Verificar disponibilidad de imagen oficial
FROM ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/paddlex:paddlex3.2.0-paddlepaddle3.1.0-cpu
```

#### Paso 2: Actualizar paddleocr en Dockerfile
```dockerfile
# Instalar version especifica de PaddleOCR 3.x
RUN pip install --break-system-packages --no-cache-dir \
    paddleocr>=3.0.0
```

#### Paso 3: Usar modelos server para maxima precision
```yaml
- OCR_TEXT_DETECTION_MODEL_NAME=PP-OCRv5_server_det
- OCR_TEXT_RECOGNITION_MODEL_NAME=PP-OCRv5_server_rec
```

**Nota:** Los modelos server requieren mas RAM (~4-8GB) y son mas lentos.

---

## 5. ANALISIS DE RIESGOS

### 5.1 Riesgos de Actualizacion

| Riesgo | Probabilidad | Impacto | Mitigacion |
|--------|--------------|---------|------------|
| Incompatibilidad API | Media | Alto | Probar en entorno staging |
| Aumento uso memoria | Alta | Medio | Monitorear y ajustar limits |
| Modelos no disponibles | Baja | Alto | Verificar descarga previa |
| Regresion en ciertos docs | Media | Medio | Comparar resultados A/B |

### 5.2 Cambios en API de PaddleOCR 3.x

**Posibles incompatibilidades:**
- El parametro `ocr_version` puede ser ignorado si se especifican `model_name`
- El parametro `lang` puede ser ignorado con modelos especificos
- Nuevos parametros disponibles en v3.x

### 5.3 Recomendacion de Pruebas

1. **Backup completo** del sistema actual
2. **Crear rama de testing** con cambios
3. **Probar con set de facturas de prueba:**
   - `/mnt/c/PROYECTOS CLAUDE/paddleocr/facturas_prueba/`
4. **Comparar metricas:**
   - Precision de texto extraido
   - Tiempo de procesamiento
   - Uso de memoria
5. **Rollback plan** si hay regresion

---

## 6. TABLA COMPARATIVA FINAL

| Aspecto | Actual | Recomendado | Mejora |
|---------|--------|-------------|--------|
| **Modelo deteccion** | PP-OCRv3_mobile | PP-OCRv5_mobile | +19% Hmean |
| **Modelo reconocimiento** | latin_PP-OCRv3 | latin_PP-OCRv5 | +7.8% precision |
| **limit_side_len** | 960 | 1216 | Mejor para A4 |
| **box_thresh** | 0.4 | 0.5 | Menos falsos positivos |
| **unclip_ratio** | 2.0 | 1.75 | Cajas mas precisas |
| **PaddleX** | 3.0.1 | 3.2.0+ | Mejor rendimiento |
| **Soporte idiomas** | 80+ | 106 | Mejor espanol |

---

## 7. CONCLUSION Y RECOMENDACIONES

### 7.1 Recomendacion Inmediata (Bajo Riesgo)

**Implementar Opcion A:** Cambiar solo los nombres de modelos a PP-OCRv5

```yaml
- OCR_TEXT_DETECTION_MODEL_NAME=PP-OCRv5_mobile_det
- OCR_TEXT_RECOGNITION_MODEL_NAME=latin_PP-OCRv5_mobile_rec
```

**Beneficio:** +10-15% mejora con minimo riesgo

### 7.2 Recomendacion a Mediano Plazo

**Implementar Opcion B:** Modelos + parametros optimizados

### 7.3 Recomendacion a Largo Plazo

**Implementar Opcion C:** Actualizar a PaddleX 3.2.0+ cuando haya imagen Docker oficial estable

---

## 8. FUENTES Y REFERENCIAS

### Documentacion Oficial
- [PaddleOCR GitHub](https://github.com/PaddlePaddle/PaddleOCR)
- [PaddleOCR Documentation](https://paddlepaddle.github.io/PaddleOCR/main/en/index.html)
- [PP-OCRv5 Introduction](https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/algorithm/PP-OCRv5/PP-OCRv5.html)
- [PP-OCRv5 Multilingual](https://www.paddleocr.ai/latest/en/version3.x/algorithm/PP-OCRv5/PP-OCRv5_multi_languages.html)
- [Text Detection Module](http://www.paddleocr.ai/main/en/version3.x/module_usage/text_detection.html)
- [Text Recognition Module](http://www.paddleocr.ai/main/en/version3.x/module_usage/text_recognition.html)

### Modelos en HuggingFace
- [PP-OCRv5_server_det](https://huggingface.co/PaddlePaddle/PP-OCRv5_server_det)
- [PP-OCRv5_mobile_rec](https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_rec)
- [PP-OCRv4_server_det](https://huggingface.co/PaddlePaddle/PP-OCRv4_server_det)
- [SLANet_plus](https://huggingface.co/PaddlePaddle/SLANet_plus)

### Articulos Tecnicos
- [PaddleOCR 3.0 Technical Report](https://arxiv.org/html/2507.05595v1)
- [PP-OCRv3 Paper](https://arxiv.org/abs/2206.03001)
- [PP-StructureV2 Paper](https://ar5iv.labs.arxiv.org/html/2210.05391)

### Releases
- [PaddleOCR Releases](https://github.com/PaddlePaddle/PaddleOCR/releases)
- [PaddleX Releases](https://github.com/PaddlePaddle/PaddleX/releases)

### Discusiones y Issues
- [Best Recognition Model Discussion](https://github.com/PaddlePaddle/PaddleOCR/discussions/14369)
- [Word Detection Configuration](https://github.com/PaddlePaddle/PaddleOCR/discussions/15011)

### Recursos Adicionales
- [PaddleOCR Guide 2025](https://www.tenorshare.com/ocr/paddleocr.html)
- [PaddleX Install Guide](https://paddlepaddle.github.io/PaddleX/3.0/en/installation/installation.html)
- [PyPI PaddleOCR](https://pypi.org/project/paddleocr/)

---

*Informe generado automaticamente por Claude Code (Opus 4.5)*
