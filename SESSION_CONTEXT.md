# SESSION_CONTEXT.md - PaddleOCR WebComunica v5

## Estado Actual

**Versión:** v5.1 (2272 líneas)
**Repositorio:** https://github.com/webcomunicasolutions/paddleocr_webcomunicav5
**Puerto:** 8505
**Rendimiento:** 2.5x más rápido que versión extendida (12-18s vs 45s)

### Cambios v5.1 (08/12/2025)
**Parámetros OCR optimizados según análisis de Agentes Opus:**
- OCR_TEXT_DET_THRESH: 0.05 → 0.25 (reduce fragmentación)
- OCR_TEXT_DET_BOX_THRESH: 0.2 → 0.4 (mejor filtrado)
- OCR_TEXT_DET_UNCLIP_RATIO: 1.5 → 2.0 (menos cortes)

**Mejoras observadas:**
- Mejor separación de palabras ("CUATRO OLIVOS" vs "CUATROOLIVOS")
- +1% confianza OCR (0.905 vs 0.895)
- Tablas detectadas y formateadas correctamente

### Cambios v5.0 (08/12/2025)
- Código base minimalista de Paco (1905 líneas)
- API REST con endpoints /process, /stats, /health
- Formato layout con detección de tablas
- TABLE_HEADER_PATTERNS: 9 patrones multi-idioma (ES, EN, DE, FR, PT)
- END_TABLE_PATTERNS: 9 patrones para fin de tabla
- Formateo de tablas con pipes |col1|col2|

## Recomendaciones de Agentes Especializados

### 1. Configuración PaddleOCR (Prioridad: ALTA)

**Actualizar a PP-OCRv5** - Mejor precisión disponible

```yaml
# Parámetros recomendados (actualmente usamos valores muy bajos)
Actual:
  OCR_TEXT_DET_THRESH: 0.05      # Muy bajo, detecta ruido
  OCR_TEXT_DET_BOX_THRESH: 0.2   # Muy bajo
  OCR_TEXT_DET_LIMIT_SIDE_LEN: 960

Recomendado:
  OCR_TEXT_DET_THRESH: 0.25      # Más selectivo
  OCR_TEXT_DET_BOX_THRESH: 0.4   # Mejor filtrado
  OCR_TEXT_DET_LIMIT_SIDE_LEN: 1920  # Mayor resolución
  OCR_VERSION: PP-OCRv5          # Último modelo
```

**Impacto:** Menos ruido, mejor precisión en texto real

---

### 2. Detección de Tablas (Prioridad: ALTA)

**Headers multi-idioma expandidos:**
```python
TABLE_HEADERS_EXTENDED = {
    'es': ['CÓDIGO', 'DESCRIPCIÓN', 'CANTIDAD', 'PRECIO', 'IMPORTE', 'IVA', 'TOTAL',
           'CONCEPTO', 'UNIDADES', 'P.UNIT', 'BASE', 'DTO', 'NETO'],
    'en': ['CODE', 'DESCRIPTION', 'QUANTITY', 'PRICE', 'AMOUNT', 'VAT', 'TOTAL'],
    'de': ['ARTIKEL', 'BESCHREIBUNG', 'MENGE', 'PREIS', 'BETRAG', 'MWST'],
    'fr': ['CODE', 'DÉSIGNATION', 'QUANTITÉ', 'PRIX', 'MONTANT', 'TVA']
}
```

**Mejora clustering columnas:**
- Usar DBSCAN en lugar de clustering simple
- Detectar columnas por densidad de coordenadas X
- Mejor alineación de celdas

**Formatos de salida:**
- Markdown (actual)
- CSV
- JSON estructurado
- HTML

---

### 3. Preprocesamiento de Imágenes (Prioridad: MEDIA)

**Pipeline adaptativo según tipo de documento:**

```python
def adaptive_preprocessing(image):
    # 1. Detectar tipo de documento
    doc_type = classify_document(image)  # invoice, ticket, scan, photo

    # 2. Aplicar pipeline específico
    if doc_type == 'ticket':
        # CLAHE para mejorar contraste en tickets térmicos
        return apply_clahe(image)
    elif doc_type == 'scan':
        # Binarización adaptativa para escaneados
        return adaptive_threshold(image)
    elif doc_type == 'photo':
        # Corrección de perspectiva + iluminación
        return perspective_and_lighting(image)
    else:
        return image  # PDFs vectoriales no necesitan preprocesamiento
```

**Mejoras específicas:**
- Detección de bordes multi-estrategia (Canny + Hough + Contornos)
- CLAHE para tickets con bajo contraste
- Deskew mejorado con ángulo preciso

---

### 4. PP-Structure para Tablas (Prioridad: MEDIA-ALTA)

**Integrar TableRecognitionPipelineV2:**

```python
from paddlex import TableRecognitionPipelineV2

# Pipeline dedicado para tablas
table_pipeline = TableRecognitionPipelineV2(
    text_det_model="PP-OCRv4_mobile_det",
    text_rec_model="PP-OCRv4_mobile_rec",
    table_recognition_model="SLANet_plus"  # Mejor modelo de tablas
)

# Usar cuando se detecte tabla en documento
if has_table_region(image):
    result = table_pipeline(image)
    html_table = result['table_html']
    structured_data = parse_html_to_json(html_table)
```

**Beneficios:**
- Extracción de estructura HTML completa
- Reconocimiento de celdas fusionadas
- Mejor manejo de tablas complejas

---

### 5. Mejoras de API (Prioridad: BAJA)

**Procesamiento asíncrono:**
```python
# Para documentos grandes
POST /process/async -> {"job_id": "abc123"}
GET /process/status/{job_id} -> {"status": "processing", "progress": 45}
GET /process/result/{job_id} -> {"text": "...", "tables": [...]}
```

**Validación mejorada:**
```python
@dataclass
class ProcessingOptions:
    format: str = "normal"  # normal, layout, table
    language: str = "es"
    dpi: int = 300
    detect_tables: bool = True
    output_format: str = "json"  # json, markdown, html
```

**OpenAPI/Swagger:**
- Documentación automática en /docs
- Esquemas de request/response

---

## Priorización Recomendada

| # | Mejora | Esfuerzo | Impacto | Recomendación |
|---|--------|----------|---------|---------------|
| 1 | Parámetros OCR optimizados | Bajo | Alto | **Implementar YA** |
| 2 | Headers tabla multi-idioma | Bajo | Medio | **Implementar YA** |
| 3 | PP-OCRv5 | Medio | Alto | Próxima versión |
| 4 | PP-Structure tablas | Alto | Alto | Próxima versión |
| 5 | Preprocesamiento adaptativo | Medio | Medio | Evaluar necesidad |
| 6 | API async | Alto | Bajo | Solo si necesario |

---

## Tareas Pendientes

- [ ] Borrar repo v4 manualmente en GitHub (Settings > Delete repository)
- [ ] Decidir qué mejoras implementar primero
- [ ] Probar parámetros OCR optimizados en facturas de prueba

---

## Comandos Útiles

```bash
# Probar v5 actual
curl -X POST http://localhost:8505/process -F "file=@factura.pdf" -F "format=layout"

# Rebuild con nuevos parámetros
docker-compose down && docker-compose build && docker-compose up -d

# Ver logs
docker-compose logs -f
```

---

*Documento generado automáticamente basado en análisis de 5 agentes Opus especializados*
