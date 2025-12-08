# PaddleOCR WebComunica v5

OCR minimalista y eficiente basado en PaddleOCR 3.x con API REST profesional.

## Características

- **Minimalista**: Solo 2241 líneas de código (vs 7000+ de versiones anteriores)
- **Rápido**: 2.5x más rápido que versiones extendidas
- **API REST**: Endpoints limpios y fáciles de usar
- **Formato Layout**: Reconstrucción espacial del documento con detección de tablas
- **Tablas con Pipes**: Formateo automático `|col1|col2|col3|`
- **Compatibilidad n8n**: Mantiene endpoint `/ocr` original

## Endpoints

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/` | GET | **Dashboard web** - Interfaz visual para probar OCR |
| `/health` | GET | Estado del servidor y modelos |
| `/stats` | GET | Estadísticas de uso |
| `/process` | POST | **Principal** - Procesa documentos con formato seleccionable |
| `/ocr` | POST | Compatibilidad con n8n (archivos en /home/n8n/in/) |

## Dashboard Web

Accede a `http://localhost:8505/` para usar la interfaz visual:

- Arrastra y suelta PDF/imágenes
- Selecciona formato: Layout (tablas) o Normal
- Visualiza resultados con estadísticas

## Uso

### Endpoint /process (Recomendado)

```bash
# Formato normal (texto plano)
curl -X POST http://localhost:8505/process \
  -F "file=@factura.pdf"

# Formato layout (preserva estructura espacial + tablas con pipes)
curl -X POST http://localhost:8505/process \
  -F "file=@factura.pdf" \
  -F "format=layout"
```

### Respuesta JSON

```json
{
  "success": true,
  "format": "layout",
  "text": "... texto formateado ...",
  "stats": {
    "avg_confidence": 0.967,
    "processing_time": 18.183,
    "total_blocks": 157,
    "total_pages": 1
  },
  "timestamp": 1733664000.123
}
```

### Ejemplo de Tabla Detectada

```
|ALB.  |PED.  |CODIGO       |DESCRIPCION  |CANT. |PRECIO |DTO. |NETO  |IMPORTE |
+------+------+-------------+-------------+------+-------+-----+------+--------+
|7.740 |8.316 |C80514 DISCO |             |2     |27,90  |     |27,90 |55,80   |
```

## Instalación

### Docker (Recomendado)

```bash
# Clonar repositorio
git clone https://github.com/webcomunicasolutions/paddleocr_webcomunicav5.git
cd paddleocr_webcomunicav5

# Construir y ejecutar
docker-compose build
docker-compose up -d

# Verificar estado
curl http://localhost:8505/health
```

### Puertos por defecto

- **8505**: Puerto expuesto (mapea al 8503 interno)

## Configuración

Variables de entorno principales en `docker-compose.yml`:

| Variable | Default | Descripción |
|----------|---------|-------------|
| `FLASK_PORT` | 8503 | Puerto interno |
| `OCR_LANG` | es | Idioma OCR |
| `OCR_VERSION` | PP-OCRv3 | Versión del modelo |
| `OCR_TEXT_DET_THRESH` | 0.05 | Umbral detección texto |
| `OCR_TEXT_DET_BOX_THRESH` | 0.2 | Umbral cajas texto |

## Rendimiento

Comparativa con versión extendida (7071 líneas):

| Métrica | v5 (2241 líneas) | Versión extendida |
|---------|------------------|-------------------|
| Tiempo procesamiento | 18.2s | 45.2s |
| Confianza OCR | 0.967 | 0.956 |
| Líneas de código | 2241 | 7071 |

## Arquitectura

```
┌─────────────────────────────────────────┐
│         API REST (336 líneas)           │
│  /health  /stats  /process  /ocr        │
├─────────────────────────────────────────┤
│      Format Layout (150 líneas)         │
│  - Detección de tablas multi-idioma     │
│  - Formateo con pipes                   │
│  - Posicionamiento espacial             │
├─────────────────────────────────────────┤
│     Core OCR Paco (1755 líneas)         │
│  - PaddleOCR 3.x via PaddleX           │
│  - OpenCV preprocessing                 │
│  - Corrección perspectiva/orientación   │
│  - PDF multipágina                      │
└─────────────────────────────────────────┘
```

## Requisitos

- Docker con soporte AVX/AVX2
- 4GB RAM mínimo
- CPU con instrucciones AVX (no funciona en VPS básicos)

## Despliegue en EasyPanel

Para desplegar en EasyPanel (sin docker-compose), consulta la guia completa:

**[EASYPANEL.md](EASYPANEL.md)**

Resumen rapido:
1. **Repo**: `https://github.com/webcomunicasolutions/paddleocr_webcomunicav5`
2. **Branch**: `main`
3. **Puerto**: `8503`
4. **RAM minima**: 4GB
5. **Volumenes**:
   - `/home/n8n/.paddlex` (cache modelos)
   - `/home/n8n/.paddleocr` (cache modelos)

## Licencia

MIT License

## Creditos

- Base OCR: Proyecto de Paco (PaddleOCR optimizado)
- API REST y Layout: WebComunica Solutions
