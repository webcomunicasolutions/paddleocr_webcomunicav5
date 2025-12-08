# SESSION_CONTEXT.md - PaddleOCR WebComunica v5

## Estado Actual

**Versión:** v5.4 (estable, producción)
**Repositorio:** https://github.com/webcomunicasolutions/paddleocr_webcomunicav5
**Puerto:** 8505
**Rendimiento:** Confianza 0.927 promedio, ~7s por documento

---

## Cambios Recientes (08/12/2025)

### v5.4 - Bug Fixes y Seguridad
- **Bug 1:** os.getenv sin default corregido
- **Bug 2:** Context managers para archivos PDF
- **Bug 3:** Path traversal validación en /ocr
- **Bug 4:** OCR_CONFIG unificado con init_ocr()
- Verificado con 10 pruebas: 10/10 OK

### Investigación PP-OCRv5 (DESCARTADA)
- Creado proyecto v6 experimental en `/paddleocr_v6_ppocrv5/`
- **Resultado:** PP-OCRv5 es PEOR que PP-OCRv3
  - Confianza: -3.4%
  - Tiempo: +100% (el doble)
- **Decisión:** Mantener PP-OCRv3

---

## Configuración Actual (Óptima)

```yaml
# Modelos (NO CAMBIAR - PP-OCRv3 es mejor para nuestros documentos)
OCR_VERSION: PP-OCRv3
OCR_TEXT_DETECTION_MODEL_NAME: PP-OCRv3_mobile_det
OCR_TEXT_RECOGNITION_MODEL_NAME: latin_PP-OCRv3_mobile_rec

# Parámetros optimizados
OCR_TEXT_DET_THRESH: 0.25
OCR_TEXT_DET_BOX_THRESH: 0.4
OCR_TEXT_DET_UNCLIP_RATIO: 2.0
OCR_TEXT_DET_LIMIT_SIDE_LEN: 960
OCR_TEXT_DET_LIMIT_TYPE: min
```

---

## Métricas de Referencia v5.4

| Archivo | Confianza | Tiempo | Bloques |
|---------|-----------|--------|---------|
| ticket.pdf | 0.905 | 4.15s | 43 |
| escaneadas 400_4.pdf | 0.969 | 18.74s | 158 |
| Factura noviembre.pdf | 0.989 | 3.41s | 83 |
| **Promedio (10 archivos)** | **0.927** | **6.98s** | - |

---

## Archivos de Documentación

| Archivo | Propósito |
|---------|-----------|
| README.md | Documentación principal |
| MEJORAS.md | Historial de optimizaciones |
| REVISION_CODIGO.md | Auditoría de código |
| LECCIONES_APRENDIDAS.md | Errores y aciertos para no repetir |
| EASYPANEL.md | Guía de despliegue |

---

## NO HACER

1. **NO actualizar a PP-OCRv5** - Ya probado, es peor
2. **NO cambiar row_tolerance** de 0.7 - Causa texto corrupto
3. **NO usar parámetros muy bajos** (OCR_TEXT_DET_THRESH < 0.2) - Detecta ruido

---

## Comandos Útiles

```bash
# Health check
curl http://localhost:8505/health

# Procesar documento
curl -X POST http://localhost:8505/process \
  -F "file=@factura.pdf" -F "format=layout"

# Rebuild
cd "/mnt/c/PROYECTOS CLAUDE/paddleocr/paddleocr_v5_paco_base"
docker-compose down && docker-compose build && docker-compose up -d

# Ver logs
docker-compose logs -f
```

---

## Proyectos Relacionados

| Proyecto | Puerto | Estado | Notas |
|----------|--------|--------|-------|
| paddleocr_v5_paco_base | 8505 | **PRODUCCIÓN** | Versión estable |
| paddleocr_v6_ppocrv5 | 8506 | DETENIDO | Experimento fallido PP-OCRv5 |
| paddleocr_experimental_layout | 8503 | - | Versión anterior |

---

*Última actualización: 08/12/2025 - v5.4*
