# MEJORAS.md - Registro de Optimizaciones PaddleOCR v5

Este documento registra todas las mejoras aplicadas sobre la versión base v5.0, incluyendo los resultados de las pruebas antes y después de cada cambio.

---

## Resumen Ejecutivo

| Versión | Fecha | Mejora Principal | Impacto |
|---------|-------|------------------|---------|
| v5.0 | 08/12/2025 | Base minimalista de Paco + API REST | Baseline |
| v5.1 | 08/12/2025 | Parámetros OCR optimizados | +1% confianza, mejor separación palabras |
| v5.2 | 08/12/2025 | Filas mínimas >=2 + is_potential_data_row() | Detecta tablas pequeñas |
| ~~v5.X~~ | 08/12/2025 | ~~row_tolerance 0.7→1.2~~ | ❌ RECHAZADA (regresión) |

---

## Análisis de Agentes Opus (08/12/2025)

Se crearon 4 agentes especializados Opus para analizar el código y proponer mejoras:

### Agente 1 - Agrupación de Bloques
**Hallazgos:**
- `row_tolerance` actual: `avg_height * 0.7` (muy restrictivo)
- Bloques de la misma línea visual se separan incorrectamente

**Recomendaciones:**
1. Aumentar `row_tolerance` a `avg_height * 1.2`
2. Añadir merge horizontal de bloques adyacentes
3. Implementar sliding window para detección de headers

### Agente 2 - Detección de Tablas
**Hallazgos:**
- Requisito actual: mínimo 3 filas para considerar tabla
- Tablas de 2 filas (header + 1 dato) no se detectan

**Recomendaciones:**
1. Reducir filas mínimas de `>=3` a `>=2`
2. Nueva función `is_potential_data_row()` para detectar filas de datos
3. Combinar patrones de headers en regex única

### Agente 3 - Layout Espacial
**Hallazgos:**
- Versión extendida usa DBSCAN para clustering de columnas
- Mejor detección de headers (busca best match, no first match)
- Manejo de colisiones entre bloques

**Recomendaciones:**
1. Implementar DBSCAN para clustering (complejo)
2. Buscar mejor match de header, no primer match
3. Añadir manejo de colisiones

### Agente 4 - Configuración OCR (CRÍTICO)
**Hallazgo crítico:**
- `OCR_TEXT_DET_THRESH=0.05` es 6x más sensible que default (0.3)
- Causa 46 bloques fragmentados vs 6 de versión extendida

**Recomendaciones:**
- `OCR_TEXT_DET_THRESH`: 0.05 → 0.25
- `OCR_TEXT_DET_BOX_THRESH`: 0.2 → 0.4
- `OCR_TEXT_DET_UNCLIP_RATIO`: 1.5 → 2.0

---

## Mejora #1: Parámetros OCR Optimizados (v5.1)

**Fecha:** 08/12/2025
**Commit:** `add8c24`
**Basado en:** Agente 4 - Configuración OCR

### Cambios Realizados

```yaml
# docker-compose.yml - ANTES
OCR_TEXT_DET_THRESH=0.05
OCR_TEXT_DET_BOX_THRESH=0.2
OCR_TEXT_DET_UNCLIP_RATIO=1.5

# docker-compose.yml - DESPUÉS
OCR_TEXT_DET_THRESH=0.25
OCR_TEXT_DET_BOX_THRESH=0.4
OCR_TEXT_DET_UNCLIP_RATIO=2.0
```

### Pruebas Realizadas

#### Prueba 1: Factura Escaneada (escaneadas 400_4.pdf)

| Métrica | v5.0 | v5.1 | Cambio |
|---------|------|------|--------|
| Confianza | ~0.96 | 0.969 | +0.9% |
| Tiempo | ~12s | 12.3s | = |
| Bloques | 158 | 158 | = |
| Tabla detectada | Sí | Sí | = |

**Resultado tabla v5.1:**
```
|ALB.   |PED.   |CODIGO        |DESCRIPCION  |CANT. |PRECIO |DTO. |NETO  |IMPORTE |
+-------+-------+--------------+-------------+------+-------+-----+------+--------+
|7.740  |8.316  |C80514 DISCO..|             |2     |27,90  |     |27,90 |55,80   |
```

#### Prueba 2: Ticket CamScanner (ticket.pdf)

| Métrica | v5.0 | v5.1 | Cambio |
|---------|------|------|--------|
| Confianza | 0.895 | 0.905 | **+1.1%** |
| Tiempo | 4.17s | 4.70s | +0.5s |
| Separación palabras | "CUATROOLIVOS" | "CUATRO OLIVOS" | **Mejorado** |

### Conclusión Mejora #1

✅ **APROBADA** - Mejora la confianza y separación de palabras sin efectos negativos.

---

## Mejora #2: Row Tolerance Aumentado (v5.2)

**Fecha:** 08/12/2025
**Basado en:** Agente 1 - Agrupación de Bloques
**Estado:** ❌ RECHAZADA

### Cambios Probados

```python
# app.py - format_text_with_layout_simple() línea 1857

# Original
row_tolerance = avg_height * 0.7

# Probado #1
row_tolerance = avg_height * 1.2  # REGRESIÓN

# Probado #2
row_tolerance = avg_height * 0.9  # REGRESIÓN
```

### Resultados con row_tolerance = 1.2

| Archivo | Problema |
|---------|----------|
| escaneadas 400_4.pdf | Tabla con columnas duplicadas incorrectamente |
| ticket.pdf | "CUATROoOLIVOS", texto final totalmente corrupto |

**Ejemplo de corrupción:**
```
ANTES (0.7): "Conforme al R.E.P.D. y de 1a L.0. 3/2018 de P.D.P.s Pudiendo ejercer sus"
DESPUÉS (1.2): "ConformeralrR.E.P.D.ayndeu1atL.0.o3/2018odeeP.D.P.srPudiendooejercerosuso"
```

### Resultados con row_tolerance = 0.9

| Archivo | Problema |
|---------|----------|
| escaneadas 400_4.pdf | Tabla OK |
| ticket.pdf | "PuertoeRealr KM.3", texto final corrupto |

### Conclusión

❌ **RECHAZADA** - El valor original de 0.7 es óptimo. Valores mayores causan que bloques de diferentes líneas se fusionen incorrectamente, produciendo texto corrupto.

**Recomendación del Agente 1 NO aplicable** - El análisis no consideró que bloques OCR de diferentes líneas pueden tener coordenadas Y similares debido a:
- Ruido en escaneos
- Texto ligeramente inclinado
- Bloques pequeños con altura variable

---

## Mejora #3: Filas Mínimas Tabla Reducidas (v5.2)

**Fecha:** 08/12/2025
**Basado en:** Agente 2 - Detección de Tablas
**Estado:** ✅ APROBADA

### Cambios Realizados

```python
# app.py - format_text_with_layout_simple() línea 1878

# ANTES
if len(rows) >= 3:  # Mínimo 3 filas

# DESPUÉS
if len(rows) >= 2:  # Mínimo 2 filas (header + 1 dato)
```

### Pruebas Realizadas

| Archivo | v5.1 (>=3) | v5.2 (>=2) | Resultado |
|---------|------------|------------|-----------|
| escaneadas 400_4.pdf | 13 filas tabla | 13 filas tabla | Sin cambio |
| ticket.pdf | Conf 0.905 | Conf 0.905 | Sin cambio |

### Conclusión

✅ **APROBADA** - Cambio neutral para tablas grandes, permite detectar tablas de 2 filas (header + 1 dato). No causa falsos positivos porque aún requiere >=3 coincidencias de patrones de header.

---

## Mejora #4: Función is_potential_data_row() (v5.2)

**Fecha:** 08/12/2025
**Basado en:** Agente 2 - Detección de Tablas
**Estado:** ✅ APROBADA (neutral)

### Cambios Realizados

```python
# Nueva función en app.py línea 1813
def is_potential_data_row(line):
    """Detecta si una línea parece ser fila de datos de tabla."""
    has_prices = bool(re_module.search(r'\d+[,\.]\d{2}', line))
    tokens = line.split()
    has_columns = len(tokens) >= 3
    has_quantity = bool(re_module.search(r'\b[1-9]\d{0,2}\b', line))
    return has_prices and (has_columns or has_quantity)

# Integrada en línea 1916
if PRICE_PATTERN.findall(row_text) or is_potential_data_row(row_text):
    data_rows.append(row_idx)
```

### Pruebas Realizadas

| Archivo | Antes | Después | Resultado |
|---------|-------|---------|-----------|
| escaneadas 400_4.pdf | 13 filas tabla | 13 filas tabla | Sin cambio |
| ticket.pdf | Conf 0.905 | Conf 0.905 | Sin cambio |

### Conclusión

✅ **APROBADA** - Cambio neutral en documentos actuales. Añade flexibilidad para detectar filas de datos en tablas con formato no estándar que no coincidan con PRICE_PATTERN.

---

## Mejora #5: Merge Horizontal de Bloques (POSPUESTA)

**Fecha:** Pendiente
**Basado en:** Agente 1 - Agrupación de Bloques

### Cambios a Realizar

```python
# Fusionar bloques adyacentes en la misma línea
def merge_adjacent_blocks(blocks, max_gap=50):
    """Fusiona bloques que están muy cerca horizontalmente"""
    merged = []
    current = blocks[0]

    for block in blocks[1:]:
        gap = block['x'] - (current['x'] + current['width'])
        if gap < max_gap and abs(block['y'] - current['y']) < row_tolerance:
            # Fusionar
            current['text'] += ' ' + block['text']
            current['width'] = block['x'] + block['width'] - current['x']
        else:
            merged.append(current)
            current = block

    merged.append(current)
    return merged
```

### Pruebas Pendientes

- [ ] Facturas con texto fragmentado
- [ ] Verificar no romper columnas de tabla

### Resultados

_(Por completar después de aplicar)_

---

## Mejora #6: DBSCAN para Clustering (v5.6) - Opcional

**Fecha:** Pendiente
**Basado en:** Agente 3 - Layout Espacial
**Complejidad:** ALTA

### Descripción

Implementar algoritmo DBSCAN para detectar columnas automáticamente basándose en la densidad de coordenadas X de los bloques.

### Consideraciones

- Requiere añadir dependencia `sklearn`
- Aumenta complejidad del código
- Beneficio principalmente para tablas complejas

### Decisión

⏸️ **POSPUESTO** - Evaluar después de aplicar mejoras 2-5

---

## Archivos de Prueba

| Archivo | Tipo | Características |
|---------|------|-----------------|
| `escaneadas 400_4.pdf` | Factura escaneada | Tabla con múltiples columnas, números de serie |
| `ticket.pdf` | Ticket gasolinera | CamScanner, baja resolución, texto pequeño |
| `Factura 1-225-1992 JUAN JOSE SANCHEZ BERNAL.pdf` | Factura compleja | Múltiples secciones |
| `Factura noviembre.pdf` | PDF vectorial | Texto limpio, layout funciona bien |

---

## Comandos de Prueba

```bash
# Probar una factura
cd "/mnt/c/PROYECTOS CLAUDE/paddleocr/facturas_prueba"
curl -X POST http://localhost:8505/process -F "file=@archivo.pdf" -F "format=layout"

# Rebuild después de cambios en app.py
cd "/mnt/c/PROYECTOS CLAUDE/paddleocr/paddleocr_v5_paco_base"
docker-compose down && docker-compose build && docker-compose up -d

# Ver logs
docker-compose logs -f

# Health check
curl http://localhost:8505/health
```

---

## Historial de Commits

| Commit | Versión | Descripción |
|--------|---------|-------------|
| `add8c24` | v5.1 | Parámetros OCR optimizados |
| _(pendiente)_ | v5.2 | Row tolerance aumentado |
| _(pendiente)_ | v5.3 | Filas mínimas reducidas |

---

*Documento actualizado: 08/12/2025*
*Mantenido por: Claude Code + Usuario*
