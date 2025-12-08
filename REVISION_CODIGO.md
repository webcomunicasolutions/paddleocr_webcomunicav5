# REVISION_CODIGO.md - Auditoría de Código app.py

Revisión realizada por 4 Agentes Opus especializados el 08/12/2025.

---

## Resumen Ejecutivo

| Categoría | Críticos | Altos | Medios | Bajos | Total |
|-----------|----------|-------|--------|-------|-------|
| Bugs y Errores | 2 | 4 | 5 | 4 | 15 |
| Código Redundante | 1 | 0 | 6 | 4 | 11 |
| Mejores Prácticas | 3 | 3 | 4 | 3 | 13 |
| Seguridad | 1 | 2 | 6 | 5 | 14 |
| **TOTAL** | **7** | **9** | **21** | **16** | **53** |

---

## Correcciones Aplicadas

### v5.3 - Correcciones de Código (08/12/2025)

| # | Problema | Estado | Líneas |
|---|----------|--------|--------|
| 1 | Funciones inexistentes `initialize_*()` | ✅ CORREGIDO | 1346, 1351 |
| 2 | Archivos PDF sin cerrar | ✅ CORREGIDO | 1566, 1655 |
| 3 | Bare except clauses | ✅ CORREGIDO | 2055, 2252 |
| 4 | Subprocess sin timeout | ✅ CORREGIDO | Múltiples |
| 5 | Imports duplicados | ✅ CORREGIDO | 16-52 |
| 6 | Path traversal (seguridad) | ✅ CORREGIDO | 2190 |
| 7 | Límite tamaño archivo | ✅ CORREGIDO | Config Flask |
| 8 | Logging de contraseña | ✅ CORREGIDO | 773 |

---

## Detalle de Problemas Encontrados

### 1. BUG CRÍTICO: Funciones Inexistentes

**Líneas:** 1346, 1351
**Descripción:** Se llaman funciones que no existen en el código.

```python
# INCORRECTO
if not initialize_docpreprocessor():
if not initialize_ocr():

# CORRECTO
if not init_docpreprocessor():
if not init_ocr():
```

**Impacto:** Error fatal `NameError` al ejecutar `proc_mpdf_ocr()`.

---

### 2. Archivos PDF No Cerrados

**Líneas:** 1566, 1655
**Descripción:** Archivos abiertos con `open()` sin context manager.

```python
# INCORRECTO
pdf_file = open(pdf_base, 'rb')
# ... código ...
pdf_file.close()  # No se ejecuta si hay excepción

# CORRECTO
with open(pdf_base, 'rb') as pdf_file:
    # ... código ...
```

---

### 3. Bare Except Clauses

**Líneas:** 2055, 2252
**Descripción:** `except:` captura TODO incluyendo KeyboardInterrupt.

```python
# INCORRECTO
except:
    pass

# CORRECTO
except Exception:
    pass
```

---

### 4. Subprocess Sin Timeout

**Líneas:** 711, 727, 778, 785, 839, 883, etc.
**Descripción:** Comandos externos sin límite de tiempo.

```python
# INCORRECTO
result = subprocess.run(['convert', ...], capture_output=True)

# CORRECTO
result = subprocess.run(['convert', ...], capture_output=True, timeout=60)
```

---

### 5. Imports Duplicados

**Líneas:** 16-17 vs 43-52
**Descripción:** cv2, numpy importados dos veces.

```python
# Línea 16-17 (mantener)
import cv2
import numpy as np

# Línea 43-52 (eliminar duplicados)
# import cv2  <- ELIMINAR
# import numpy as np  <- ELIMINAR
```

---

### 6. Path Traversal (Seguridad)

**Líneas:** 2190
**Descripción:** Nombre de archivo no sanitizado.

```python
# INCORRECTO
temp_filename = f"proc_{int(time.time())}_{file.filename}"

# CORRECTO
from werkzeug.utils import secure_filename
safe_name = secure_filename(file.filename)
temp_filename = f"proc_{int(time.time())}_{safe_name}"
```

---

### 7. Sin Límite de Tamaño

**Descripción:** No hay límite para archivos subidos.

```python
# AÑADIR después de crear app
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB
```

---

### 8. Contraseña en Logs

**Línea:** 773
**Descripción:** NIF de empresa (usado como contraseña) se loguea.

```python
# INCORRECTO
logger.info(f"[JSON] empresaNif: {empresaNif}")

# CORRECTO
logger.info(f"[JSON] empresaNif: {'*' * len(empresaNif) if empresaNif else 'N/A'}")
```

---

## Problemas Menores (No Críticos)

### Código Redundante
- Import de `traceback` repetido en 6 lugares → Mover a imports globales
- Import de `PIL.Image` repetido en 5 lugares → Mover a imports globales
- Ordenamiento de puntos duplicado en 3 lugares → Crear función auxiliar
- Overlay OCR duplicado en 2 lugares → Crear función auxiliar

### Mejores Prácticas
- Funciones muy largas (>100 líneas): `compose_pdf_ocr()` 266 líneas
- Magic numbers sin constantes: 0.3, 6, 20 en múltiples lugares
- Falta de type hints en todo el archivo
- Docstrings incompletos

### Seguridad Menor
- Logging en nivel DEBUG en producción
- Sin headers de seguridad HTTP
- Sin rate limiting

---

## Pruebas Realizadas v5.3

| Archivo | Tipo | Confianza | Tiempo | Resultado |
|---------|------|-----------|--------|-----------|
| escaneadas 400_4.pdf | Factura escaneada | 0.969 | 11.97s | ✅ OK - Tabla detectada |
| ticket.pdf | Ticket gasolinera | 0.905 | 4.61s | ✅ OK |

**Todos los tests pasaron correctamente después de aplicar las 8 correcciones.**

---

## Historial de Versiones

| Versión | Fecha | Cambios |
|---------|-------|---------|
| v5.0 | 08/12/2025 | Base minimalista |
| v5.1 | 08/12/2025 | Parámetros OCR optimizados |
| v5.2 | 08/12/2025 | Mejoras detección tablas |
| v5.3 | 08/12/2025 | Correcciones de código (8 fixes) |

---

*Documento generado automáticamente por análisis de 4 Agentes Opus*
