# SLM Industrial Gateway

Gateway HTTP compatible con OpenAI para un modelo de lenguaje pequeño (SLM)
adaptado por dominio a jerga industrial y minera, con ejecución de
herramientas analíticas (DuckDB, detección de anomalías, cálculo de RUL) y
guardrails de seguridad. Diseñado para desplegarse on-prem, sin dependencia
de red para la ruta de inferencia.

## Arquitectura

El sistema se compone de seis módulos independientes bajo `src/`, integrados
por la capa de API a través de un pipeline multi-agente:

```
                        ┌──────────────────────────────┐
                        │        src/api (FastAPI)      │
                        │  /v1/chat/completions          │
                        │  /v1/models  /health  /metrics │
                        └───────────────┬────────────────┘
                                        │ delega en
                        ┌───────────────▼─────────────────┐
                        │  src/agents (AgentOrchestrator)   │
                        │  RouterAgent -> AnalyticsAgent     │
                        │  (+PdMAgent, advisor) -> Verifier- │
                        │  Agent -> SafetyComplianceAgent    │
                        │  (bloqueante)                      │
                        └────┬──────────────┬──────────┬────┘
                             ▼               ▼          ▼
                   ┌───────────────┐ ┌──────────────┐ ┌──────────────────┐
                   │  src/engine    │ │  src/tools    │ │  src/guardrails  │
                   │  LLMServer     │ │  ToolRegistry │ │  validate_sql_   │
                   │  (llama.cpp,   │ │  + industrial │ │  query,          │
                   │  GGUF, fallback│ │  _tools       │ │  validate_json_  │
                   │  GPU→CPU)      │ │  (DuckDB,     │ │  output          │
                   └───────────────┘ │  Z-score, RUL)│ └──────────────────┘
                                      └──────────────┘

        src/evaluation (offline, no se ejecuta en el request path)
        FaithfulnessEvaluator (DeepEval, LLM juez externo) para evaluación
        batch — distinto del chequeo de fidelidad local y sin red que hace
        VerifierAgent en cada solicitud (ver "Flujo de datos" más abajo).

        src/training (offline, no se ejecuta en el gateway)
        QLoRA (Unsloth + TRL SFTTrainer) sobre data/domain_dataset/
        → produce adaptadores LoRA que se cargan como el modelo GGUF
          cuantizado que consume src/engine.
```

`src/tools` depende de `src/guardrails` (todo query SQL y todo argumento de
tool pasa por un validador antes de ejecutarse). `src/agents` depende de los
tres módulos de runtime (`engine`, `tools`, `guardrails`) y es el único punto
de entrada que usa `src/api` para procesar una conversación: la API ya no
arma el prompt ni despacha tools directamente, delega todo el flujo a
`AgentOrchestrator`. `src/evaluation` y `src/training` son pipelines offline,
desacoplados del servicio HTTP.

### Flujo de datos de `/v1/chat/completions`

```mermaid
flowchart LR
    Client(["Cliente industrial"]) --> Gateway["FastAPI Gateway<br/>/v1/chat/completions"]
    Gateway --> InGuard{"Input Guardrail<br/>validación Pydantic"}
    InGuard -- invalido --> Err400a["HTTP 400"]
    InGuard -- valido --> Router["RouterAgent<br/>LLM Engine, GGUF air-gapped"]
    Router -- texto en lenguaje natural --> Verifier
    Router -- tool call JSON --> Analytics["AnalyticsAgent<br/>DuckDB / Z-score / RUL"]
    Analytics -- SQL peligroso o tool invalida --> Err400b["HTTP 400"]
    Analytics -- resultado OK --> PdM["PdMAgent<br/>interpreta RUL/anomalía, advisor"]
    PdM -.->|si urgente| MaintAlert["maintenance_alert<br/>(metadata, no bloquea)"]
    PdM --> Router2["RouterAgent<br/>segunda pasada"]
    Router2 --> Verifier{"VerifierAgent<br/>formato + fidelidad numérica"}
    Verifier -- vacio/invalido (modelo) --> Err500["HTTP 500"]
    Verifier -- tool-call sin resolver o<br/>valor no sustentado --> Err400c["HTTP 400"]
    Verifier -- valido y fiel --> Safety{"SafetyComplianceAgent<br/>límites físicos de diseño"}
    Safety -- valor fuera de límite --> Err400d["HTTP 400<br/>SAFETY_ALERT"]
    Safety -- dentro de límites --> Response(["Respuesta HTTP<br/>contrato OpenAI"])
```

1. **Validación de entrada**: Pydantic valida el payload (`ChatCompletionRequest`);
   se rechaza con `400` si `messages` está vacío o no cumple el esquema.
   `src/api` traduce los `ChatMessage` del contrato HTTP a `AgentMessage`
   internos y delega en `AgentOrchestrator.run()` (`src/agents`). Antes de
   tocar el modelo, el orquestador pasa el último mensaje del usuario por
   `RouterAgent.check_threat()` (`detect_threat`): patrones deterministas de
   inyección de prompt, SQL o comandos se rechazan con `400` sin costo de red
   ni de inferencia. `RouterAgent.classify_intent()` (clasificación
   ANALYTICS_REQUIRED/DIRECT_QA con una llamada barata al SLM) existe y está
   probado (`tests/test_router_agent.py`) pero todavía no se invoca en este
   camino caliente — ver el docstring de `src/agents/orchestrator.py`.
2. **RouterAgent**: construye un prompt que incluye el listado de tools
   disponibles (`ToolRegistry.list_tools()`) y las instrucciones de formato de
   tool call, e invoca `LLMServer.generate()` (`src/engine`, backend
   llama.cpp sobre el modelo GGUF local). La respuesta se intenta parsear
   como `{"tool": "<nombre>", "arguments": {...}}`; si no es JSON válido para
   ese esquema, se trata como respuesta final en lenguaje natural y se salta
   al paso 4.
3. **AnalyticsAgent (si el RouterAgent pidió una tool)**: despacha la llamada
   vía `ToolRegistry.dispatch()` (`src/tools`), que valida los argumentos
   contra el `args_schema` de la tool y —para `query_duckdb`— además contra
   `validate_sql_query` (`src/guardrails`): solo se permiten
   `SELECT/WITH/EXPLAIN/DESCRIBE/SHOW`, una única sentencia, sin palabras
   clave destructivas. Si la tool fue `calculate_rul` o `sensor_anomaly_check`,
   `PdMAgent.assess()` (`src/agents/pdm_agent.py`) interpreta el resultado
   crudo contra un umbral de urgencia fijo (RUL ≤ 24h, o ≥ 2 lecturas
   anómalas) y, si corresponde, adjunta `maintenance_alert` al resultado
   final — **advisor, no bloqueante**: un RUL bajo es información operativa
   para planificar mantenimiento, no una condición insegura que deba impedir
   la respuesta. El resultado de la tool se inyecta como un mensaje de rol
   `tool` y se vuelve a invocar al `RouterAgent` para que redacte la
   respuesta final con ese contexto (un único hop: si el SLM pide otra tool
   en esta segunda pasada, el `VerifierAgent` la rechaza en el paso 4 en vez
   de encadenar).
4. **VerifierAgent**: antes de responder, rechaza (`500`, `GenerationError`)
   una respuesta final vacía o de tipo inválido —una falla del modelo, no de
   contenido—; rechaza (`400`) una respuesta final que en realidad es un
   tool-call sin resolver; y rechaza (`400`, `FaithfulnessError`) una
   respuesta que menciona valores numéricos que no aparecen en el resultado
   crudo de la tool ejecutada. Este último chequeo es una heurística local
   (comparación de números, sin LLM juez ni red) — un filtro barato en el
   camino caliente, no un reemplazo del `FaithfulnessMetric` de DeepEval
   (`src/evaluation`), que sigue siendo la evaluación de referencia pero solo
   corre offline/batch. Cualquier otro error de guardrail o de ejecución de
   tool (SQL peligroso, tool desconocida, argumentos inválidos) también se
   traduce a `400`.
5. **SafetyComplianceAgent**: la última puerta, después de que `VerifierAgent`
   ya aprobó el texto final. Audita cada valor numérico con unidad
   (potencia MW, presión PSI, temperatura °C) mencionado en la respuesta
   contra una matriz FIJA de límites de diseño físico del sitio
   (`OPERATIONAL_LIMITS`, `src/agents/safety_agent.py`) — inmutable en
   runtime, sin variable de entorno que la relaje. Si algún valor supera su
   límite, lanza `SafetyAlertError` (`SAFETY_ALERT`) y la respuesta **nunca**
   llega al cliente, ni siquiera parcialmente; `src/api` lo traduce a `400`
   igual que el resto de los `GuardrailError`.
6. **Respuesta JSON**: se devuelve un `ChatCompletionResponse` con el mismo
   contrato que la API de OpenAI, y se registran métricas de Prometheus
   (tokens generados, latencia por token, resultado de la solicitud).

Todas las tools están protegidas por guardrails; ninguna tool ejecuta SQL de
escritura ni comandos del sistema.

## Benchmarks

### Benchmark Real en Entorno Edge / CPU x86_64 (Qwen2.5-1.5B Q4_K_M)

Medido de verdad, no un placeholder: `Qwen/Qwen2.5-1.5B-Instruct-GGUF`
(`qwen2.5-1.5b-instruct-q4_k_m.gguf`, ~1.04 GB) descargado con
`huggingface_hub`, corrido con `python -m src.engine.benchmarks
--model-path data/models/model.gguf` sobre **CPU forzada explícitamente**
(`n_gpu_layers=0`, ver `_run_cli` en `src/engine/benchmarks.py`) -- este
equipo no tiene GPU dedicada, así que no hay ninguna cifra de GPU en esta
sección, medida ni estimada.

![Quantization Benchmark](outputs/reports/quant_benchmark.png)

| Métrica | Valor (última corrida) | Rango sobre 5 corridas |
|---|---:|---:|
| TTFT | 4,709 ms | 3,431 – 4,736 ms |
| Throughput | 8.11 tok/s | 5.50 – 8.11 tok/s |
| RAM residente (RSS) | 1,144 MB | 1,143.5 – 1,144.4 MB |
| VRAM | N/A | sin GPU dedicada; forzado a CPU |

*Hardware: AMD Ryzen 5 2500U (4 núcleos / 8 hilos, sin GPU dedicada), Windows.
Prompt de 128 tokens de salida máx. (ver `DEFAULT_BENCHMARK_PROMPT`),
`n_threads` autodetectado por llama.cpp. El rango viene de 5 corridas
independientes en la misma máquina, no de una sola medición: TTFT varió
~38% entre la corrida más rápida y la más lenta, y el throughput ~47% --
varianza real de un laptop compartido con otros procesos, no de una GPU/CPU
de benchmarking dedicada, y se reporta así en vez de citar solo la corrida
más favorable. El archivo completo de la última corrida (la que grafica la
imagen de arriba) queda versionado en
[`outputs/reports/benchmark_result.json`](outputs/reports/benchmark_result.json)
para que cualquiera pueda verificar el número exacto sin volver a correr el
benchmark.*

Para reproducir:

```bash
python -m huggingface_hub download Qwen/Qwen2.5-1.5B-Instruct-GGUF \
    qwen2.5-1.5b-instruct-q4_k_m.gguf --local-dir data/models
cp data/models/qwen2.5-1.5b-instruct-q4_k_m.gguf data/models/model.gguf

python -m src.engine.benchmarks --model-path data/models/model.gguf \
    --model-label "Qwen2.5-1.5B-Instruct Q4_K_M"

python scripts/generate_plots.py   # regenera quant_benchmark.png desde el JSON guardado
```

### Métricas de evaluación del agente

> **Los números de esta sub-sección siguen siendo ilustrativos.** A
> diferencia del benchmark de arriba, correr `src/evaluation.FaithfulnessEvaluator`
> de verdad requiere un modelo juez externo (por defecto `gpt-4o-mini` vía
> API) que este comando no invoca -- no es un límite de CPU/GPU sino de
> credenciales de red, fuera del alcance de esta corrida. Reemplazar estos
> valores exige correr el evaluador aparte sobre un set de prueba real, más
> las tasas de SQL Safety / JSON Validity de `src.guardrails` sobre intentos
> de dispatch de tools registrados.

![Eval Metrics](outputs/reports/eval_metrics.png)

| Métrica            | Valor | Qué mide |
|---------------------|------:|----------|
| Faithfulness         | 0.93  | El `FaithfulnessMetric` de DeepEval: cuánto de la respuesta está sustentado por el contexto recuperado. |
| Answer Relevancy     | 0.89  | Qué tan pertinente es la respuesta final respecto a la pregunta del usuario. |
| SQL Safety Rate      | 1.00  | Fracción de intentos de `query_duckdb` con SQL peligroso correctamente bloqueados por `validate_sql_query`. |
| JSON Validity        | 0.97  | Fracción de tool calls emitidas por el SLM que parsean como `ToolCallEnvelope` sin error de esquema. |

Para regenerar ambos gráficos: el de cuantización lee
`outputs/reports/benchmark_result.json` si existe (la corrida real de arriba)
y cae al placeholder ilustrativo si no; el de métricas de evaluación sigue
siendo siempre el placeholder.

```bash
python scripts/generate_plots.py
```

## Estructura del repositorio

```
src/
  engine/       LLMServer (GGUF/llama.cpp), benchmarks (CLI real: t/s, TTFT, RAM/VRAM)
  api/          Gateway FastAPI, contrato OpenAI, métricas Prometheus
  agents/       Pipeline multi-agente: RouterAgent -> AnalyticsAgent (si
                aplica, con evaluación PdMAgent advisor) -> VerifierAgent
                (fidelidad numérica + formato) -> SafetyComplianceAgent
                (límites físicos, bloqueante) -- los cuatro, locales y sin
                red en cada solicitud
  tools/        ToolRegistry, herramientas industriales (query_duckdb,
                sensor_anomaly_check, calculate_rul)
  guardrails/   Validación de SQL y de salidas JSON estrictas
  evaluation/   Evaluador de fidelidad/alucinación (DeepEval, offline)
  training/     Pipeline QLoRA (dataset_prep.py, finetune.py)
data/
  domain_dataset/  Dataset sintético ChatML de dominio (telemetría/sensores)
  models/          Pesos GGUF locales (no versionado; ver instalación)
scripts/
  quantize.py       Verifica/carga modelos GGUF cuantizados (Q4_K_M, Q8_0)
  generate_plots.py Genera los gráficos de `outputs/reports/` (ver Benchmarks)
outputs/
  reports/        Gráficos versionados que embeben el README (PNG), más
                   benchmark_result.json (última corrida real de CPU)
monitoring/
  prometheus.yml  Configuración de scraping para el contenedor Prometheus
tests/
  test_engine.py, test_api.py, test_agents.py, test_router_agent.py,
  test_analytics_agent.py, test_pdm_agent.py, test_safety_agent.py,
  test_tools.py, test_guardrails.py, test_eval.py, test_training.py,
  test_integration.py (ver `pytest -v` para el listado completo y vigente)
```

## Guía de despliegue air-gapped

El servicio de inferencia (`src/engine` + `src/api`) no requiere red en
tiempo de ejecución: el modelo GGUF es un archivo local y llama.cpp corre
in-process. Los únicos puntos que sí asumen red por defecto son (a) la
instalación de dependencias Python y (b) el módulo de evaluación si se deja
apuntando a un juez alojado en la nube. Para un entorno sin salida a
Internet:

### 1. Dependencias Python (wheelhouse)

En una máquina con red (misma versión de Python/plataforma que el destino):

```bash
pip download -r requirements.txt -d wheelhouse/
```

Copiar `wheelhouse/` y `requirements.txt` al entorno aislado, e instalar ahí
sin acceso a PyPI:

```bash
pip install --no-index --find-links=wheelhouse/ -r requirements.txt
```

### 2. Montaje local de los pesos GGUF

El modelo nunca se descarga en tiempo de ejecución: `MODEL_PATH` apunta a un
archivo `.gguf` que ya tiene que estar en disco.

1. Copiar el archivo a `data/models/model.gguf` (o la ruta que indique
   `MODEL_PATH`); en Docker Compose esa carpeta se monta como volumen
   de solo lectura (`./data/models:/app/data/models:ro`), así que el `.gguf`
   nunca se copia dentro de la imagen ni queda expuesto por accidente.
2. Verificar el archivo (cabecera GGUF válida y cuantización detectada) y
   opcionalmente cargarlo en memoria para confirmar que arranca:

   ```bash
   python scripts/quantize.py data/models/model.gguf --load
   ```

### 3. Arranque offline vía Docker Compose

En la máquina con red, pre-cargar las imágenes base (no hay Dockerfile para
Prometheus: se usa la imagen oficial tal cual):

```bash
docker pull python:3.11-slim
docker pull prom/prometheus:v3.0.1
docker save python:3.11-slim prom/prometheus:v3.0.1 -o base-images.tar
```

En el destino aislado:

```bash
docker load -i base-images.tar
docker compose build   # usa solo wheelhouse/ y las imagenes ya cargadas, sin red
docker compose up -d
```

`docker compose build` no debe disparar ningún acceso a red si el
`Dockerfile` y `requirements.txt` están fijados a wheels ya presentes en
`wheelhouse/`; si el build intenta salir a Internet, es señal de una
dependencia sin pin exacto en `requirements.txt`.

### 4. Configuración de guardrails

Los guardrails (`src/guardrails/validators.py`) son intencionalmente
**código, no configuración**: la lista de verbos SQL permitidos
(`ALLOWED_SQL_VERBS`) y de palabras clave bloqueadas (`BLOCKED_SQL_KEYWORDS`)
son constantes fijas, sin variable de entorno ni flag que las relaje en
producción. Esto es deliberado en un entorno air-gapped: no hay superficie de
configuración en runtime que un operador pueda aflojar por error (o que un
prompt del propio SLM pueda intentar manipular). Para ampliar qué puede hacer
el agente, la vía soportada es agregar una tool nueva y explícita en
`src/tools/industrial_tools.py`, no relajar el validador SQL existente.

### 5. Evaluación offline sin salida a Internet

`src/evaluation` usa DeepEval con un modelo juez configurable
(`FaithfulnessEvaluator(judge_model=...)`). Por defecto apunta a
`gpt-4o-mini` (API externa). En un entorno aislado, apuntar
`OPENAI_BASE_URL`/`OPENAI_API_KEY` a un endpoint OpenAI-compatible servido
localmente (por ejemplo, este mismo gateway u otro servidor local), o
simplemente omitir la evaluación de fidelidad en producción: en CI,
`tests/test_eval.py` corre con las métricas de DeepEval mockeadas, sin red.

## Configuración (variables de entorno)

| Variable              | Default                          | Usado por         |
|-----------------------|-----------------------------------|-------------------|
| `MODEL_NAME`          | `local-slm`                       | `src/api` (metadata de `/v1/models`) |
| `MODEL_PATH`          | `data/models/model.gguf`          | `src/api` → `src/engine.LLMServer` |
| `MODEL_N_CTX`         | `4096`                            | `src/engine.LLMServer` (ventana de contexto) |
| `DEEPEVAL_JUDGE_MODEL`| `gpt-4o-mini`                      | `src/evaluation.FaithfulnessEvaluator` |
| `OPENAI_API_KEY`      | (vacío)                           | Cliente del modelo juez de DeepEval |
| `OPENAI_BASE_URL`     | (vacío)                           | Cliente del modelo juez de DeepEval (endpoint local en despliegues aislados) |

## Ejecutar localmente

```bash
pip install -r requirements.txt
export MODEL_PATH=data/models/model.gguf   # o el path real al .gguf
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000
```

## Ejecutar con Docker Compose

```bash
docker compose up -d --build
```

Expone:
- API: `127.0.0.1:8000` (`/v1/chat/completions`, `/v1/models`, `/health`, `/metrics`)
- Prometheus: `127.0.0.1:9090`, con scraping ya configurado hacia
  `slm-api:8000/metrics` (ver `monitoring/prometheus.yml`)

Ambos puertos están atados a `127.0.0.1` deliberadamente: el servicio no se
expone a la red pública por defecto.

## Pruebas

```bash
pytest -v
```

Por módulo:

```bash
pytest tests/test_engine.py -v        # LLMServer, fallback GPU→CPU, benchmarks
pytest tests/test_api.py -v           # Contrato HTTP, LLM mockeado
pytest tests/test_agents.py -v        # AnalyticsAgent/VerifierAgent + AgentOrchestrator + API end-to-end
pytest tests/test_router_agent.py -v  # RouterAgent: guardrail de entrada + clasificación de intención
pytest tests/test_analytics_agent.py -v  # AnalyticsAgent en aislamiento (dispatch real, SQL peligroso)
pytest tests/test_pdm_agent.py -v     # PdMAgent: interpretación de RUL/anomalía, advisor (nunca lanza)
pytest tests/test_safety_agent.py -v  # SafetyComplianceAgent: límites físicos de diseño, bloqueante
pytest tests/test_tools.py -v         # Tools industriales + ToolRegistry
pytest tests/test_guardrails.py -v    # Validación de SQL y de JSON de salida
pytest tests/test_eval.py -v          # Evaluador de fidelidad (métricas mockeadas)
pytest tests/test_training.py -v      # Dataset ChatML, config y args de QLoRA
pytest tests/test_integration.py -v   # End-to-end: API + tools + guardrails reales
```

`tests/test_integration.py` monta un `TestClient` de FastAPI y ejercita el
flujo completo descrito arriba con DuckDB y los guardrails corriendo de
verdad; solo `LLMServer.generate` se simula, porque no hay pesos GGUF en CI.

`tests/test_training.py` no requiere GPU: valida el dataset, el formateo
ChatML y la construcción de `QLoRATrainingConfig`/`SFTConfig`, pero no
entrena. El entrenamiento real (`src/training/finetune.py`) requiere una GPU
con soporte CUDA y `unsloth` + `bitsandbytes` instalados.

## Notas de compatibilidad de dependencias

`unsloth` fija transitivamente el resto del stack de fine-tuning. En
concreto, `unsloth==2026.9.5` requiere `trl<=0.24.0`, mientras que
`trl>=1.0` requiere `transformers>=4.56.2` pero es incompatible con ese techo
de `unsloth`. `requirements.txt` fija `trl==0.24.0` (no la línea 1.x) junto
con `transformers==4.56.2`, `peft==0.21.0`, `accelerate==1.15.0`,
`datasets==4.3.0`, `bitsandbytes==0.50.2` y `torch==2.12.1`, todos dentro de
los rangos que `unsloth` declara soportar. Antes de subir cualquiera de estas
versiones, revisar la metadata de distribución de `unsloth` para confirmar
que el nuevo rango sigue siendo compatible.

## Seguridad y guardrails

- **SQL**: `query_duckdb` solo ejecuta una única sentencia `SELECT/WITH/
  EXPLAIN/DESCRIBE/SHOW`; se bloquean `DROP/DELETE/ALTER/INSERT/UPDATE/
  CREATE/ATTACH/EXEC/...` aunque aparezcan en subconsultas o tras comentarios.
- **Salidas estructuradas**: toda tool valida sus argumentos contra un
  esquema Pydantic en modo estricto (sin coerción de tipos), y rechaza
  campos desconocidos.
- **Guardrail de salida del SLM**: se rechaza una respuesta final vacía antes
  de devolverla al cliente.
- **Fidelidad de la respuesta final (`VerifierAgent`)**: si se ejecutó una
  tool, se rechaza cualquier respuesta final que mencione un valor numérico
  ausente del resultado crudo de esa tool, y cualquier respuesta final que en
  realidad sea un tool-call sin resolver. Es una heurística local (comparación
  de números, no un LLM juez) pensada como red de seguridad en tiempo real;
  no reemplaza al `FaithfulnessMetric` de DeepEval (`src/evaluation`), que es
  más riguroso pero solo corre offline/batch.
- **Límites físicos de diseño (`SafetyComplianceAgent`)**: última puerta del
  pipeline, después de `VerifierAgent`. Rechaza cualquier respuesta final que
  proponga un valor de potencia, presión o temperatura por encima del límite
  de diseño fijo del sitio (`OPERATIONAL_LIMITS`, inmutable en runtime, sin
  variable de entorno que lo relaje) — la respuesta nunca llega al cliente,
  ni parcialmente. No confundir con `PdMAgent`: ese es advisor (adjunta
  `maintenance_alert` sin bloquear nada), este bloquea.
