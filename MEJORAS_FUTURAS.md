# MEJORAS_FUTURAS.md - PaddleOCR WebComunica v5

Documento con opciones de mejora y escalado para cuando el proyecto crezca.

---

## Estado Actual (v5.5)

- **Solución**: `threading.Semaphore(1)` serializa peticiones OCR
- **Probado**: 10/10 peticiones simultáneas exitosas
- **Throughput**: ~1 documento cada 10-15 segundos
- **RAM**: 4GB

---

## Opciones de Mejora para el Futuro

### Opción 1: Múltiples Contenedores + Balanceador (RECOMENDADA)

**Cuándo usar**: Tráfico medio (10-50 req/min)

**Arquitectura**:
```
                    ┌─────────────────┐
                    │   Nginx/Traefik │
                    │   (Balanceador) │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  OCR Container  │ │  OCR Container  │ │  OCR Container  │
│    Puerto 8505  │ │    Puerto 8506  │ │    Puerto 8507  │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

**Implementación con docker-compose**:
```yaml
version: '3.8'
services:
  ocr1:
    image: paddleocr_v5_paco_base_paddlepaddle-v5
    ports:
      - "8505:8503"
    deploy:
      resources:
        limits:
          memory: 4G

  ocr2:
    image: paddleocr_v5_paco_base_paddlepaddle-v5
    ports:
      - "8506:8503"
    deploy:
      resources:
        limits:
          memory: 4G

  nginx:
    image: nginx:alpine
    ports:
      - "8500:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - ocr1
      - ocr2
```

**nginx.conf**:
```nginx
upstream ocr_servers {
    least_conn;  # Enviar al servidor con menos conexiones
    server ocr1:8503;
    server ocr2:8503;
}

server {
    listen 80;
    location / {
        proxy_pass http://ocr_servers;
        proxy_connect_timeout 300s;
        proxy_read_timeout 300s;
    }
}
```

**Ventajas**:
- Fácil de implementar
- Escala horizontalmente
- Cada contenedor es independiente
- Sin cambios en código

**Recursos**: 4GB RAM x N contenedores

---

### Opción 2: Pool de Instancias OCR (Mayor Complejidad)

**Cuándo usar**: Si necesitas más throughput sin más contenedores

**Concepto**: Crear N instancias de PaddleOCR en memoria

```python
from queue import Queue
from paddleocr import PaddleOCR

class OCRPool:
    def __init__(self, size=4):
        self.pool = Queue(maxsize=size)
        for _ in range(size):
            self.pool.put(PaddleOCR(lang='es'))

    def get(self):
        return self.pool.get(timeout=300)

    def release(self, instance):
        self.pool.put(instance)

# Uso
ocr_pool = OCRPool(size=4)

def process():
    ocr = ocr_pool.get()
    try:
        return ocr.ocr(image)
    finally:
        ocr_pool.release(ocr)
```

**Ventajas**:
- N peticiones simultáneas en un solo contenedor
- Menor overhead de red

**Desventajas**:
- Mayor uso de RAM (~4GB x N instancias)
- Mayor complejidad de código
- Si una instancia se corrompe, afecta al pool

**Recursos**: 4GB x N instancias en un solo contenedor

---

### Opción 3: PaddleServing (Alta Carga)

**Cuándo usar**: Alto tráfico (>100 req/min), máximo rendimiento

**Qué es**: Servidor oficial de Baidu optimizado para inferencia

**Instalación**:
```bash
# Instalar plugin de serving
paddlex --install serving

# Lanzar servidor OCR
paddlex --serve --pipeline OCR --port 8080 --device cpu
```

**Características**:
- Diseñado para alta concurrencia
- Soporte C++, Java, Go, Node.js, PHP
- Basado en NVIDIA Triton (opcional)
- Batching automático de peticiones
- Monitoreo y métricas integradas

**Documentación**:
- https://www.paddleocr.ai/main/en/version3.x/deployment/serving.html
- https://github.com/PaddlePaddle/Paddle-Inference-Demo

**Ventajas**:
- Máximo rendimiento posible
- Soporte oficial de Baidu
- Escalabilidad empresarial

**Desventajas**:
- Requiere refactorización completa
- Curva de aprendizaje
- Arquitectura diferente

---

### Opción 4: Cola de Mensajes (Procesamiento Asíncrono)

**Cuándo usar**: Procesamiento batch, no necesitas respuesta inmediata

**Arquitectura**:
```
Usuario → API → Redis/RabbitMQ → Worker OCR → Resultado en BD
                     ↑
              N Workers procesando
```

**Implementación con Celery**:
```python
from celery import Celery

app = Celery('ocr', broker='redis://localhost:6379/0')

@app.task
def process_document(file_path):
    # Procesar con OCR
    result = ocr_instance.predict(file_path)
    # Guardar resultado en BD
    save_result(result)
    return result

# En API
@app.route('/process', methods=['POST'])
def process():
    task = process_document.delay(file_path)
    return {'task_id': task.id}

@app.route('/result/<task_id>')
def get_result(task_id):
    task = process_document.AsyncResult(task_id)
    if task.ready():
        return {'result': task.result}
    return {'status': 'processing'}
```

**Ventajas**:
- Escala infinitamente
- No bloquea al usuario
- Reintentos automáticos
- Ideal para procesamiento batch

**Desventajas**:
- Mayor complejidad de arquitectura
- Requiere BD para resultados
- No es tiempo real

---

## Comparativa de Opciones

| Opción | Complejidad | Throughput | RAM | Caso de Uso |
|--------|-------------|------------|-----|-------------|
| Semáforo actual | Baja | 1 req/vez | 4GB | <10 req/min |
| Múltiples contenedores | Baja | N req/vez | 4GB x N | 10-50 req/min |
| Pool de instancias | Media | N req/vez | 4GB x N | 10-50 req/min |
| PaddleServing | Alta | Alto | Variable | >100 req/min |
| Cola de mensajes | Alta | Ilimitado | Variable | Batch processing |

---

## Recomendación de Migración

1. **Fase 1 (Actual)**: Semáforo - Suficiente para inicio
2. **Fase 2**: Múltiples contenedores + Nginx - Cuando necesites más throughput
3. **Fase 3**: PaddleServing - Cuando tengas tráfico empresarial

---

## Monitoreo Recomendado

Para cualquier opción, añadir:

```python
# Endpoint de métricas
@app.route('/metrics')
def metrics():
    return {
        'requests_total': server_stats['total_requests'],
        'requests_failed': server_stats['failed_requests'],
        'avg_processing_time': server_stats.get('avg_time', 0),
        'queue_size': ocr_semaphore._value  # Peticiones esperando
    }
```

---

## Referencias

- [PaddleOCR Threading Issues](https://github.com/PaddlePaddle/PaddleOCR/issues/16238)
- [PaddleServing Docs](https://www.paddleocr.ai/main/en/version3.x/deployment/serving.html)
- [Celery Documentation](https://docs.celeryq.dev/)
- [Nginx Load Balancing](https://nginx.org/en/docs/http/load_balancing.html)

---

*Documento creado: 08/12/2025*
*Versión actual: v5.5 con Semáforo*
