# LECCIONES_APRENDIDAS.md - PaddleOCR WebComunica

Documento de lecciones aprendidas para evitar repetir errores y aprovechar lo que funciona.

---

## Fecha: 08/12/2025

### Contexto
Auditoría de código con 4 agentes Opus especializados + investigación de modelos PP-OCRv5.

---

## LO QUE FUNCIONÓ BIEN

### 1. Auditoría con Agentes Especializados
| Aspecto | Resultado |
|---------|-----------|
| 4 agentes paralelos (Bugs, Redundancia, Prácticas, Seguridad) | Encontraron 53 problemas en total |
| División de responsabilidades clara | Cada agente se enfocó en su área sin duplicar trabajo |
| Priorización por severidad (CRÍTICO/ALTO/MEDIO/BAJO) | Permitió arreglar lo importante primero |

**Recomendación:** Usar agentes especializados para auditorías de código grandes.

### 2. Verificar Antes de Cambiar
| Aspecto | Resultado |
|---------|-----------|
| 10 pruebas antes de aplicar cambios | Detectamos que v5 funcionaba bien |
| 10 pruebas después de bug fixes | Confirmamos que no hubo regresiones |
| Pruebas con archivos diversos (escaneados, vectoriales, tickets) | Cobertura amplia de casos de uso |

**Recomendación:** SIEMPRE hacer pruebas antes Y después de cambios.

### 3. Proyecto Paralelo para Experimentos
| Aspecto | Resultado |
|---------|-----------|
| Crear v6 separado para probar PP-OCRv5 | No afectó a v5 que funcionaba |
| Puertos diferentes (8505 vs 8506) | Ambos podían correr simultáneamente |
| Volúmenes Docker separados | Modelos independientes sin conflictos |

**Recomendación:** Para experimentos arriesgados, crear proyecto paralelo.

### 4. Documentación Continua
| Aspecto | Resultado |
|---------|-----------|
| MEJORAS.md con cada cambio | Registro de qué se probó y resultado |
| SESSION_CONTEXT.md | Contexto rápido para retomar trabajo |
| REVISION_CODIGO.md | Auditoría detallada con líneas de código |

**Recomendación:** Documentar mientras se trabaja, no al final.

---

## LO QUE NO FUNCIONÓ / ERRORES COMETIDOS

### 1. Confiar en Benchmarks Oficiales sin Verificar
| Error | Impacto | Lección |
|-------|---------|---------|
| Asumir que PP-OCRv5 sería mejor porque benchmarks decían +19% | Perdimos tiempo configurando v6 que resultó PEOR | **Los benchmarks son con datasets específicos, NO garantizan mejora en TU caso de uso** |

**Datos reales vs benchmarks:**
```
Benchmark oficial PP-OCRv5: +19% detección, +7.8% reconocimiento
Nuestras pruebas reales:    -3.4% confianza, +100% tiempo
```

**Recomendación:** SIEMPRE probar con TUS documentos reales antes de migrar.

### 2. Parámetros No Optimizados para Nuevo Modelo
| Error | Impacto | Lección |
|-------|---------|---------|
| Usar parámetros genéricos para PP-OCRv5 | Rendimiento subóptimo | **Cada versión de modelo puede requerir parámetros diferentes** |

**Parámetros que usamos:**
```yaml
# v5 (PP-OCRv3) - optimizado para nuestros documentos
OCR_TEXT_DET_LIMIT_SIDE_LEN: 960
OCR_TEXT_DET_BOX_THRESH: 0.4
OCR_TEXT_DET_UNCLIP_RATIO: 2.0

# v6 (PP-OCRv5) - valores genéricos sin optimizar
OCR_TEXT_DET_LIMIT_SIDE_LEN: 1216
OCR_TEXT_DET_BOX_THRESH: 0.5
OCR_TEXT_DET_UNCLIP_RATIO: 1.75
```

**Recomendación:** Si se migra a nuevo modelo, hacer tuning de parámetros con pruebas A/B.

### 3. Modelo de Reconocimiento Incorrecto para Español
| Error | Impacto | Lección |
|-------|---------|---------|
| Usar `latin_PP-OCRv5_mobile_rec` | Peor reconocimiento en español | **"latin" no es lo mismo que "español"** |

**Alternativas no probadas:**
- `es_PP-OCRv5_mobile_rec` (si existe)
- `multilingual_PP-OCRv5_mobile_rec` (si existe)
- Modelos server en vez de mobile

**Recomendación:** Investigar qué modelos específicos existen para el idioma objetivo.

### 4. No Medir Tiempo de Respuesta como Métrica Crítica
| Error | Impacto | Lección |
|-------|---------|---------|
| Solo enfocarnos en confianza OCR | v6 era 2x más lento y no lo detectamos hasta las pruebas | **El tiempo de procesamiento es TAN importante como la precisión** |

**Recomendación:** Definir métricas de éxito ANTES de experimentos:
- Confianza mínima: 0.90
- Tiempo máximo: 10s por página
- Bloques mínimos: No regresión

---

## CONCLUSIONES CLAVE

### 1. PP-OCRv3 > PP-OCRv5 (para nuestro caso)
```
┌─────────────────────────────────────────────────────────┐
│  PP-OCRv3 (v5 actual) es MEJOR que PP-OCRv5 para:      │
│  - Facturas españolas escaneadas                        │
│  - Tickets de gasolinera                                │
│  - PDFs vectoriales                                     │
│  - Documentos con tablas                                │
└─────────────────────────────────────────────────────────┘
```

### 2. "Nuevo" No Siempre es "Mejor"
- Los modelos más recientes están optimizados para benchmarks genéricos
- Documentos específicos (facturas españolas) pueden funcionar peor
- La versión "mobile" de v5 puede ser peor que "mobile" de v3

### 3. Mantener Versión Estable
- v5 con PP-OCRv3 es nuestra versión de producción
- v6 queda como referencia de lo que NO hacer
- No actualizar modelos sin pruebas exhaustivas

---

## CHECKLIST PARA FUTUROS CAMBIOS

### Antes de Cambiar Modelos OCR
- [ ] Definir métricas de éxito (confianza, tiempo, bloques)
- [ ] Crear proyecto paralelo (no tocar producción)
- [ ] Preparar 10+ documentos de prueba diversos
- [ ] Investigar modelo específico para idioma (no "latin" genérico)

### Durante Pruebas
- [ ] Ejecutar TODAS las pruebas en versión actual (baseline)
- [ ] Ejecutar TODAS las pruebas en versión nueva
- [ ] Comparar métricas lado a lado
- [ ] Verificar casos edge (tablas, tickets, escaneados)

### Después de Pruebas
- [ ] Documentar resultados en MEJORAS.md
- [ ] Si peor: descartar y documentar por qué
- [ ] Si mejor: migrar gradualmente con rollback plan
- [ ] Actualizar LECCIONES_APRENDIDAS.md

---

## RECURSOS ÚTILES

### Comandos de Prueba
```bash
# Prueba rápida con formato layout
curl -X POST http://localhost:8505/process \
  -F "file=@documento.pdf" \
  -F "format=layout"

# Ver métricas en respuesta JSON
python3 -c "import json; d=json.load(open('result.json'));
print(f'Conf: {d[\"stats\"][\"avg_confidence\"]:.3f}, Time: {d[\"stats\"][\"processing_time\"]:.2f}s')"
```

### Archivos de Referencia
| Archivo | Tipo | Confianza esperada v5 |
|---------|------|----------------------|
| ticket.pdf | Ticket escaneado | ~0.905 |
| escaneadas 400_4.pdf | Factura escaneada | ~0.969 |
| Factura noviembre.pdf | PDF vectorial | ~0.989 |

---

## HISTORIAL DE DECISIONES

| Fecha | Decisión | Razón | Resultado |
|-------|----------|-------|-----------|
| 08/12/2025 | Mantener PP-OCRv3 | v5 probado peor en 10 pruebas | CORRECTO |
| 08/12/2025 | Crear v6 paralelo | No afectar producción | CORRECTO |
| 08/12/2025 | Aplicar bug fixes v5.4 | 4 bugs críticos detectados | CORRECTO |

---

## Fecha: 21/12/2025

### Contexto
Despliegue en nuevo servidor (GESFAC). El build del contenedor fallaba con múltiples errores.

---

## PROBLEMAS DETECTADOS Y SOLUCIONADOS

### Problema 1: Mirror de pip chino no accesible

| Aspecto | Detalle |
|---------|---------|
| **Error** | `[Errno 101] Network is unreachable` al conectar a `pypi.tuna.tsinghua.edu.cn` |
| **Causa** | La imagen base de Baidu usa mirror de pip en China (Universidad Tsinghua) |
| **Impacto** | El build falla al intentar instalar dependencias Python |

**Solución en Dockerfile:**
```dockerfile
# Añadir ANTES del pip install
RUN pip config set global.index-url https://pypi.org/simple/
```

---

### Problema 2: opencv-python==4.6.0.66 no existe

| Aspecto | Detalle |
|---------|---------|
| **Error** | `No matching distribution found for opencv-python==4.6.0.66` |
| **Causa** | Esa versión específica no existe en PyPI (quizás era del mirror chino) |
| **Impacto** | El build falla al no encontrar la versión |

**Solución en Dockerfile:**
```dockerfile
# Cambiar de:
opencv-python==4.6.0.66
# A:
opencv-python
```

---

### Problema 3: NumPy 2.x incompatible con paddleocr

| Aspecto | Detalle |
|---------|---------|
| **Error** | `numpy.core.multiarray failed to import` y `PDX has already been initialized` |
| **Causa** | NumPy 2.0+ rompe compatibilidad con paquetes compilados para NumPy 1.x |
| **Impacto** | paddleocr no puede importarse, el servidor arranca pero OCR no funciona |

**Mensaje de error clave:**
```
A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.2.6 as it may crash.
```

**Solución en Dockerfile:**
```dockerfile
# Cambiar de:
numpy
# A:
"numpy<2"
```

---

## ¿POR QUÉ FUNCIONABA EN EL SERVIDOR VIEJO?

| Factor | Servidor viejo | Servidor nuevo |
|--------|---------------|----------------|
| **Contenedor** | Construido hace semanas | Construido HOY |
| **NumPy** | 1.24.x (versión de entonces) | 2.2.6 (última) |
| **pip install sin versión** | Instalaba versiones antiguas | Instala últimas versiones |

**Lección clave:** El Dockerfile sin versiones fijadas es una "bomba de tiempo". Cada rebuild puede instalar versiones diferentes.

---

## DOCKERFILE FINAL CORREGIDO

```dockerfile
# FIX 1: Cambiar a PyPI oficial (la imagen base usa mirror chino)
RUN pip config set global.index-url https://pypi.org/simple/

# Instalar dependencias Python
RUN python3.10 -m pip install --upgrade pip && \
    pip install --break-system-packages --no-cache-dir \
    "numpy<2" \              # FIX 3: Forzar NumPy 1.x
    decord \
    opencv-python \          # FIX 2: Sin versión específica
    paddleocr \
    pdf2image==1.16.3 \
    ...
```

---

## COMMITS RELACIONADOS

| Commit | Descripción |
|--------|-------------|
| `49e6a1e` | fix: Dockerfile compatible con servidores fuera de China |
| `1404d80` | fix: Restaurar paddleocr en pip install (necesario) |
| `857c271` | fix: Forzar numpy<2 (paddleocr incompatible con NumPy 2.x) |
| `499a479` | fix: v5.6 Semáforo en doc_preprocessor (concurrencia) |

---

### Problema 4: std::exception con múltiples documentos simultáneos

| Aspecto | Detalle |
|---------|---------|
| **Error** | `std::exception` cuando llegan varios documentos a la vez |
| **Causa** | `doc_preprocessor.predict()` no tenía semáforo, solo `ocr_instance.predict()` |
| **Impacto** | Errores intermitentes con reintentos, procesamiento más lento |

**Solución en app.py (v5.6):**
```python
def fix_orientation(img_path, doc_preprocessor):
    ...
    # v5.6: Semáforo para doc_preprocessor (PaddlePaddle no es thread-safe)
    with ocr_semaphore:
        output = doc_preprocessor.predict(img_path, batch_size=1)
```

**Explicación:**
- PaddlePaddle NO es thread-safe
- Había DOS modelos: `doc_preprocessor` y `ocr_instance`
- Solo `ocr_instance` estaba protegido con semáforo
- Ahora AMBOS usan el mismo semáforo

---

## RECOMENDACIONES FUTURAS

1. **Fijar versiones críticas** - Especialmente numpy, opencv, paddleocr
2. **Probar rebuilds periódicamente** - Para detectar incompatibilidades antes de producción
3. **Documentar la imagen base** - La de Baidu tiene configuraciones ocultas (mirror chino)

---

*Documento creado: 08/12/2025*
*Última actualización: 21/12/2025*
*Mantenido por: Equipo WebComunica + Claude Code*
