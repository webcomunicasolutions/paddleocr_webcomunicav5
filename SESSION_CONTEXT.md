# SESSION_CONTEXT.md - PaddleOCR WebComunica v5

## Estado Actual

**Versión:** v5.5 (estable, producción)
**Repositorio:** https://github.com/webcomunicasolutions/paddleocr_webcomunicav5
**Puerto:** 8505
**Rendimiento:** Confianza 0.927 promedio, ~10s por documento

---

## IMPORTANTE: Problema de Concurrencia (LEER PRIMERO)

### El Problema
PaddleOCR **NO es thread-safe**. Cuando múltiples peticiones llegan simultáneamente, causa:
```
RuntimeError: std::exception
```

**Es un bug conocido de PaddleOCR**, no de nuestro código:
- [Issue #16238](https://github.com/PaddlePaddle/PaddleOCR/issues/16238)
- [Issue #11605](https://github.com/PaddlePaddle/PaddleOCR/issues/11605)

### La Solución (v5.5)
Usamos `threading.Semaphore(1)` para serializar peticiones OCR:

```python
# En app.py línea ~130
ocr_semaphore = threading.Semaphore(1)

# En app.py línea ~1248
with ocr_semaphore:
    page_ocr_result = ocr_instance.predict(out_png)
```

**Esto garantiza que solo 1 petición OCR se ejecuta a la vez.**

### Resultado
- **10/10 peticiones simultáneas exitosas**
- **0 errores std::exception**
- **Throughput**: ~1 documento cada 10-15 segundos

### Si necesitas más throughput
Ver **MEJORAS_FUTURAS.md** para opciones de escalado:
1. Múltiples contenedores + Nginx (recomendado)
2. PaddleServing oficial (alta carga)

---

## Cambios v5.5 (08/12/2025)

- **Semáforo**: `threading.Semaphore(1)` serializa peticiones OCR
- **UUID**: Archivos temporales con UUID evitan colisiones
- **Documentación**: MEJORAS_FUTURAS.md con opciones de escalado
- **Probado**: 10/10 peticiones simultáneas OK

---

## Configuración Actual (NO CAMBIAR)

```yaml
# Modelos - PP-OCRv3 es MEJOR que PP-OCRv5 para nuestros documentos
OCR_VERSION: PP-OCRv3
OCR_TEXT_DETECTION_MODEL_NAME: PP-OCRv3_mobile_det
OCR_TEXT_RECOGNITION_MODEL_NAME: latin_PP-OCRv3_mobile_rec

# Parámetros optimizados
OCR_TEXT_DET_THRESH: 0.25
OCR_TEXT_DET_BOX_THRESH: 0.4
OCR_TEXT_DET_UNCLIP_RATIO: 2.0
OCR_TEXT_DET_LIMIT_SIDE_LEN: 960
```

---

## NO HACER (Errores ya cometidos)

1. **NO quitar el semáforo** - Causa std::exception inmediatamente
2. **NO usar threading.Lock()** - No es suficiente, necesita Semaphore
3. **NO actualizar a PP-OCRv5** - Ya probado, es -3.4% peor
4. **NO crear pool de instancias** - Complejo, mejor múltiples contenedores

---

## Archivos de Documentación

| Archivo | Propósito |
|---------|-----------|
| README.md | Documentación principal + sección concurrencia |
| MEJORAS_FUTURAS.md | **Opciones de escalado para el futuro** |
| LECCIONES_APRENDIDAS.md | Errores y aciertos para no repetir |
| REVISION_CODIGO.md | Auditoría de código |
| EASYPANEL.md | Guía de despliegue |

---

## Comandos Útiles

```bash
# Health check
curl http://localhost:8505/health

# Procesar documento
curl -X POST http://localhost:8505/process \
  -F "file=@factura.pdf" -F "format=layout"

# Prueba de estrés (10 peticiones)
for i in {1..10}; do
  curl -s -X POST http://localhost:8505/process \
    -F "file=@/tmp/ticket.pdf" -F "format=layout" \
    -o /tmp/test_$i.json &
done
wait

# Rebuild
cd "/mnt/c/PROYECTOS CLAUDE/paddleocr/paddleocr_v5_paco_base"
docker-compose down && docker-compose build && docker-compose up -d

# Ver logs
docker-compose logs -f
```

---

## Métricas de Referencia

| Archivo | Confianza | Tiempo |
|---------|-----------|--------|
| ticket.pdf | 0.905 | ~11s |
| Factura noviembre.pdf | 0.989 | ~10s |
| escaneadas 400_4.pdf | 0.969 | ~19s |

---

## Proyectos Relacionados

| Proyecto | Puerto | Estado |
|----------|--------|--------|
| paddleocr_v5_paco_base | 8505 | **PRODUCCIÓN** |
| paddleocr_v6_ppocrv5 | 8506 | DESCARTADO (PP-OCRv5 peor) |

---

*Última actualización: 08/12/2025 - v5.5*
