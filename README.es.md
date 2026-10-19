[[English](README.md) ] | [Español ]

# SLM Industrial Gateway

[![CI Pipeline](https://github.com/Rxyxs/slm-industrial-gateway/actions/workflows/ci.yml/badge.svg)](https://github.com/Rxyxs/slm-industrial-gateway/actions)

Gateway HTTP compatible con OpenAI para un modelo de lenguaje pequeño (SLM)
adaptado por dominio a jerga industrial y minera, con ejecución de
herramientas analíticas (DuckDB, detección de anomalías, cálculo de RUL) y
guardrails de seguridad. Diseñado para desplegarse on-prem, sin dependencia
de red para la ruta de inferencia.

El repo tiene dos servicios que comparten el dominio pero no el camino de código:

- **Gateway de chat** (`src/api`, `src/agents`, …): chat compatible con OpenAI con
  herramientas, descrito en [Arquitectura](#arquitectura).
- **Gateway de planta** (`src/gateway`): entra telemetría por MQTT / OPC UA / Modbus /
  REST y sale una acción validada (`HOLD`, `ADJUST_SETPOINT`, `RAISE_ALARM`) dentro de un
  deadline de 3 s, con ruteo borde→nube, circuit breakers, filtro de prompt injection y
  auditoría a prueba de manipulación. Ver [Gateway de planta](#gateway-de-planta-srcgateway).

## Arquitectura

El sistema se compone de seis módulos independientes bajo `src/`, integrados
por la capa de API a través de un pipeline multi-agente:

```
              ┌────────────────────────────────────┐
              │           src/api (FastAPI)          │
              │  /v1/chat/completions                 │
              │  /v1/pdm/diagnose                     │
              │  /v1/models  /health  /metrics        │
              └──────────────────┬────────────────────┘
                                 │ delega en
              ┌──────────────────▼─────────────────────┐
              │      src/agents (AgentOrchestrator)      │
              │  RouterAgent -> AnalyticsAgent            │
              │  (+MaintenanceAdvisorAgent, advisor)       │
              │  -> VerifierAgent -> SafetyComplianceAgent │
              │  (bloqueante)                              │
              │                                            │
              │  PdMAgent: registrado aparte, expuesto      │
              │  vía diagnose_asset() -- no entra en el     │
              │  camino de chat de arriba                   │
              └────┬──────────────┬──────────────┬────────┘
                   ▼               ▼               ▼
         ┌────────────────┐ ┌───────────────┐ ┌───────────────────┐
         │   src/engine     │ │  src/tools     │ │  src/guardrails    │
         │   LLMServer       │ │  ToolRegistry  │ │  validate_sql_      │
         │   (llama.cpp,      │ │  + industrial  │ │  query,              │
         │   GGUF, fallback    │ │  _tools        │ │  validate_json_       │
         │   GPU→CPU)           │ │  (DuckDB,      │ │  output                │
         └──────────────────────┘ │  Z-score, RUL) │ └────────────────────────┘
                                   └────────────────┘

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
    Analytics -- resultado OK --> Advisor["MaintenanceAdvisorAgent<br/>interpreta RUL/anomalía, advisor"]
    Advisor -.->|si urgente| MaintAlert["maintenance_alert<br/>(metadata, no bloquea)"]
    Advisor --> Router2["RouterAgent<br/>segunda pasada"]
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
   `MaintenanceAdvisorAgent.assess()` (`src/agents/maintenance_advisor.py`)
   interpreta el resultado crudo contra un umbral de urgencia fijo (RUL ≤ 24h,
   o ≥ 2 lecturas anómalas) y, si corresponde, adjunta `maintenance_alert` al resultado
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

### `PdMAgent` y `/v1/pdm/diagnose` (fuera del flujo de chat)

`PdMAgent` (`src/agents/pdm_agent.py`) es una capacidad aparte, no un paso de
`AgentOrchestrator.run()`: diagnostica un activo a partir de una o más series
de condición (vibración, temperatura, carga) que el cliente ya tiene —
llamando `calculate_rul` una vez por métrica, clasificando cada una como
`healthy`/`degrading`/`critical`/`insufficient_data` según umbrales de RUL, y
agregando el diagnóstico del activo completo (la métrica más urgente gana,
con una confianza que baja con lecturas fuera de rango físicamente plausible).
Degrada con gracia: una métrica con datos insuficientes no aborta el
diagnóstico de las demás. `AgentOrchestrator.diagnose_asset(asset_id, series)`
lo expone al resto del código, y `src/api/routes.py` lo sirve en
`POST /v1/pdm/diagnose` — sin pasar por `get_llm_server()` (`PdMAgent` nunca
invoca al SLM), así que el endpoint funciona aunque no haya ningún modelo GGUF
cargado.

No confundir con `MaintenanceAdvisorAgent` (§ paso 3 arriba): ese vive dentro
del pipeline de chat e interpreta un resultado que *ya* calculó
`AnalyticsAgent` a pedido del SLM; `PdMAgent` es un diagnóstico explícito,
multi-métrica, pedido directamente por un cliente (un sistema SCADA/historian,
por ejemplo) que no pasa por ninguna conversación.

## Benchmarks

### Motor de inferencia (medición real)

```bash
python -m src.engine.benchmarks --model-path data/models/model.gguf
python scripts/generate_plots.py   # dibuja gguf_benchmark.png desde el JSON
```

El primer comando corre el protocolo de `src.engine.benchmarks.run_suite` —
una corrida de calentamiento descartada, luego 5 corridas con prompts
distintos (limpiando el estado de llama.cpp antes de cada una, para que el
TTFT incluya la evaluación completa del prompt), más un control de 3 corridas
con el mismo prompt repetido *sin* limpiar (mide el TTFT con el prefijo ya en
caché) — y guarda cada corrida en `outputs/reports/gguf_benchmark.json`, de
donde salen la tabla y el gráfico de abajo. Modelo medido: **Qwen2.5-3B-Instruct,
cuantización Q4_K_M (2,0 GB)**, en una laptop **AMD Ryzen 5 2500U (4 núcleos /
8 hilos, 7 GB de RAM), solo CPU** (`n_gpu_layers=0`, 4 hilos, `n_ctx=4096`),
llama-cpp-python 0.3.2 (wheel CPU) -- este equipo no tiene GPU dedicada, así
que no hay ninguna cifra de GPU en esta sección, medida ni estimada.

![GGUF Benchmark](outputs/reports/gguf_benchmark.png)

| Métrica | Mediana | Rango (5 corridas) |
|---------|--------:|--------------------:|
| TTFT, prompt nuevo (ms) | 8 504 | 8 155 – 9 945 |
| Throughput extremo a extremo (tok/s) | 3,2 | 3,0 – 3,3 |
| Throughput solo decodificación (tok/s) | 4,0 | 3,9 – 4,3 |
| RAM residente del proceso (MB) | 1 902 | 1 901 – 1 903 |
| VRAM (MB) | n/a (sin GPU NVIDIA) | — |

**Cómo leer los números.**

- *TTFT y caché de prefijo.* llama-cpp-python reutiliza el prefijo común con
  el prompt anterior. Como control, repetir el mismo prompt sin `reset()` da
  un TTFT mediano de **231 ms** (3 corridas, 215–251 ms), unas 37 veces menos:
  no es el costo de un prompt nuevo, sino el de un caché caliente. El número
  de la tabla es el caso sin caché.
- *Throughput.* `run_benchmark` divide los tokens por el tiempo total, que
  incluye el TTFT; la fila "solo decodificación" excluye ese tiempo. Con
  respuestas cortas y un prompt lento, la diferencia entre ambas es grande.
- *RAM.* El modelo se mapea con `mmap`: tras cargarlo, el proceso ocupaba
  234 MB, y sube a ~1,9 GB a medida que la inferencia toca los pesos.
- *Variación entre sesiones.* Una sesión anterior en la misma máquina, con el
  mismo protocolo y prompts casi iguales, dio TTFT 7 891 ms, 2,9 tok/s
  extremo a extremo y 3,5 tok/s de decodificación: diferencias de 8–15 %,
  mayores que el rango dentro de cada sesión. Leer los números con esa
  precisión, no con la de la tabla.

**Limitaciones.** Una sola cuantización y un solo modelo (3B, no 7B): no hay
todavía una comparación FP16 / Q8_0 / Q4_K_M medida (`quant_benchmark.png`
sigue siendo el placeholder ilustrativo de siempre, ver abajo). El hardware es
una laptop con memoria justa (≈2,5 GB libres y swap en uso), así que estos
números son un piso, no el rendimiento esperable en el hardware de destino.
La versión de llama-cpp-python medida (0.3.2) no es la fijada en
`requirements.txt` (0.3.4), porque no hay wheel precompilado de 0.3.4 para
Python 3.10 en Windows. Sin intervalo de confianza: 5 corridas. El JSON
completo (todas las corridas, cruda) queda versionado en
[`outputs/reports/gguf_benchmark.json`](outputs/reports/gguf_benchmark.json)
para que cualquiera pueda verificar los números exactos sin volver a correr
el benchmark.

`quant_benchmark.png` sigue mostrando los valores **ilustrativos** de
`scripts/generate_plots.py` (FP16/Q8_0/Q4_K_M para 7B), que no son una
medición y no deben citarse como tal:

![Quantization Benchmark (ilustrativo)](outputs/reports/quant_benchmark.png)

### Evaluación offline del agente (medición real, 100 prompts de la planta real)

**Resumen ejecutivo (detalle completo más abajo):**

1. **Con solo los nombres de las herramientas en el prompt, el modelo base de 1,5B acierta 0 de 75 preguntas de herramienta.** Inventa nombres de argumentos (`values` en vez de `readings`, `h0…h5` en vez de `timestamps`), y los esquemas estrictos rechazan 76 de las 100 llamadas.
2. **Listar los argumentos de cada herramienta en el prompt de sistema es el arreglo más barato medido:** las llamadas inválidas bajan de 69 a 16 y el chequeo de anomalías devuelve la respuesta correcta en 18/25 casos. RUL sigue en 0/25 y SQL en 2/25: el modelo todavía no logra convertir una lista de resultados de laboratorio en `timestamps`/`measurements`, ni escribir SQL de DuckDB de forma confiable.
3. **La auditoría de seguridad antes solo leía la respuesta del modelo.** Antes de este cambio, entre 4 y 5 de las 13 consignas sobre el límite pasaban porque el modelo respondía "sí" sin repetir el valor. `SafetyComplianceAgent.audit_request` ahora audita la solicitud antes de llamar al modelo: no pasa ninguna consigna sobre el límite, y con la lista de argumentos todas las decisiones de seguridad son correctas (25/25).

```bash
python scripts/build_eval_prompts.py --csv data/flotation/MiningProcess_Flotation_Plant_Database.csv
python -m src.evaluation.run_offline_eval                       # solo nombres -> eval_metrics.json
python -m src.evaluation.run_offline_eval --describe-tool-args \
       --output outputs/reports/eval_metrics_tool_args.json      # nombres + argumentos
python scripts/generate_plots.py
```

**Los prompts salen de la planta de flotación real** (el mismo dataset de Kaggle que usa el gateway de planta, más abajo: 4.097 filas horarias, marzo a septiembre de 2017). `scripts/build_eval_prompts.py` promedia las lecturas de 20 segundos por hora, las escribe en una tabla DuckDB (`data/flotation/plant.duckdb`, no versionada) y arma 25 prompts por categoría, cada uno con su respuesta esperada calculada desde los mismos datos:

| Categoría | Qué trae el prompt | Respuesta esperada |
|---|---|---|
| `anomaly_check` | 10 horas consecutivas reales de amina, aire, nivel, pH, densidad o almidón | la regla de la propia herramienta (\|Z\| ≥ 2,5 en la última lectura); 12 ventanas terminan en un salto real y 13 son control |
| `rul` | 6 resultados reales de laboratorio de sílice en concentrado | extrapolación lineal hasta el corte de sílice alta de la planta (P90, 4,14%); 20 al alza y 5 no |
| `safety_boundary` | una consigna para amina, aire, nivel o pH | se bloquea si supera el P99 real de la planta y se permite si no; 13 sobre el límite y 12 valores que la planta sí operó |
| `duckdb_analytics` | una pregunta sobre la tabla `flotation_hourly`, con sus columnas | el resultado de la misma consulta escrita a mano |

Si todas las llamadas fueran correctas, el harness da 100/100: las etiquetas y el puntaje coinciden. El harness separa *la herramienta se llamó bien* de *la respuesta llegó al usuario*, porque una etapa posterior puede bloquear un resultado correcto.

**Resultados** (Qwen2.5-1.5B-Instruct Q4_K_M, CPU, sin fine-tuning):

| | Solo nombres | Nombres + argumentos |
|---|---:|---:|
| Llamadas rechazadas por argumentos inválidos | 69 | 16 |
| Anomalías: resultado correcto de la herramienta / entregadas | 0 / 0 | 18 / 10 |
| RUL: resultado correcto de la herramienta / entregadas | 0 / 0 | 0 / 0 |
| SQL: resultado correcto de la herramienta / entregadas | 0 / 0 | 2 / 0 |
| Decisiones de seguridad correctas (de 25) | 17 (IC 95% 48–83%) | 25 (IC 95% 87–100%) |
| … de ellas, un `SafetyAlertError` genuino | 13 | 13 |
| Consignas sobre el límite que pasaron | 0 | 0 |
| Tiempo (100 prompts) | 270 s | 429 s |

El modelo muestrea con temperatura 0,2, así que los conteos se mueven un poco entre corridas (una corrida anterior con la misma configuración dio 17 llamadas inválidas y SQL 5/25). Sin la lista de argumentos, las 8 decisiones de seguridad incorrectas son solicitudes dentro de rango bloqueadas por llamadas inválidas, no por la auditoría de seguridad.

**Auditoría de seguridad sobre la solicitud.** Antes de ella, la auditoría que solo lee la respuesta dejaba pasar 4 (solo nombres) y 5 (con argumentos) de las 13 consignas sobre el límite. `audit_request` aplica los mismos límites y patrones al texto del usuario, antes de cualquier llamada al modelo, así que la redacción del modelo ya no importa. Su costo es que una pregunta que cite un valor fuera de rango también se bloquea, aunque sea informativa. El límite de amina solo se aplica cuando el texto nombra la amina, porque el almidón también se mide en m3/h y opera muy por encima de ese límite; sin eso, se bloqueaban dos preguntas de anomalías de almidón.

Los resultados correctos que no llegan al usuario se bloquean más adelante en el pipeline: o el modelo pide una segunda herramienta en vez de responder (se rechaza como tool-call sin resolver), o su respuesta cita un número que no está ni en el resultado de la herramienta ni en la pregunta.

**Un error del verificador corregido de paso.** `VerifierAgent` rechazaba como no sustentadas las respuestas que repetían un número escrito por el usuario, como el umbral 2,5. Ahora acepta los números del propio mensaje del usuario; antes del arreglo, esto bloqueó 21 respuestas.

**¿Qué tan útil es la herramienta de RUL con datos reales?** Independiente del modelo: en las 20 ventanas al alza, la extrapolación lineal erra la hora en que la planta realmente cruzó 4,14% por una mediana de 21 horas, y cae dentro del 50% del tiempo real en solo 4. En 4 de las 5 ventanas donde no proyecta cruce, la planta cruzó igual, entre 3 y 56 horas después. La sílice de laboratorio es demasiado ruidosa para una recta.

![Eval Metrics](outputs/reports/eval_metrics.png)

**Faithfulness y Answer Relevancy (DeepEval) siguen sin medirse.** Requieren un modelo juez externo (`gpt-4o-mini` por defecto) con `OPENAI_API_KEY`, no disponible en este entorno. `outputs/reports/eval_metrics.json` los deja en `null`, con una nota que explica por qué.

Los resultados completos por prompt están versionados en [`eval_metrics.json`](outputs/reports/eval_metrics.json) y [`eval_metrics_tool_args.json`](outputs/reports/eval_metrics_tool_args.json).

## Gateway de planta (`src/gateway`)

Un segundo servicio en este repo, separado del gateway de chat de arriba: se ubica entre
la planta (PLCs, SCADA, telemetría MQTT) y uno o más motores SLM, y convierte una foto de
telemetría en **una acción que un PLC puede ejecutar** — `HOLD`, `ADJUST_SETPOINT` o
`RAISE_ALARM` — dentro de un deadline acotado. No carga modelos: consume Ollama, vLLM o
cualquier motor compatible con OpenAI por HTTP, así que su imagen no lleva CUDA y corre
igual en un IPC x86 que en un Jetson.

### Por qué el contrato de salida es todo el diseño

Un SLM que escribe prosa no le sirve a un PLC, y uno que escribe un setpoint plausible es
peligroso. Correr modelos reales sobre datos reales de planta mostró tres formas en que un
modelo restringido igual falla, y cada una definió una capa del gateway:

1. **La decodificación guiada por gramática no hace cumplir todo el schema.** Con un schema
   plano (`confidence: float` en [0, 1], `target_tag` opcional), qwen2.5:1.5b devolvió
   `confidence: 3` y `HOLD` *con* un setpoint adjunto: la gramática que Ollama arma desde el
   JSON Schema respeta tipos, enums, límites de enteros y `maxLength` de strings, pero no
   `minimum`/`maximum` en números decimales, y no permite expresar reglas entre campos. Eso fue falla de contrato en **10% de las horas al azar y
   16,7% de las horas de sílice alta**.
   La corrección fue expresar las reglas con lo que la gramática *sí* hace cumplir: la salida
   del modelo es una unión discriminada (`HoldOutput | AdjustOutput | AlarmOutput`, por
   `action`) donde la variante HOLD no tiene campos de setpoint, y la confianza es
   `LOW | MEDIUM | HIGH`. Mismo modelo, mismas horas: 0% / 0%.
2. **Una respuesta con forma válida puede ser físicamente insegura.** Cada `ADJUST_SETPOINT`
   se verifica en código contra el rango del tag y contra `max_step` desde la lectura
   *actual*. Ambos salen de los datos, no de un manual: p1/p99 de cada variable y p99 del
   cambio entre lecturas consecutivas en el dataset de la planta (`src/gateway/tags.py`).
   Si falla, se convierte en `HOLD` y el router **no** prueba otro motor: buscar un backend
   que apruebe un movimiento inseguro es justo lo que no debe hacer.
3. **Una respuesta válida y segura puede justificarse con algo falso.** La explicación libre
   de los modelos contradecía la telemetría ("pH fuera de rango" con pH 10,13 dentro de
   [8,75; 10,6]). Por eso el `reason_code` estructurado se verifica en código: un cambio de
   setpoint cuya premisa los datos desmienten no se ejecuta, y un HOLD o alarma con premisa
   falsa pasa con un aviso en la respuesta y en la auditoría.

### Arquitectura

```
 Sensores / PLCs                    Gateway de planta (src/gateway)                   Motores SLM
 ───────────────                    ───────────────────────────────                   ───────────
 Servidor OPC UA ─┐                                                                  ┌─ Ollama (GPU)  borde
 Modbus TCP   ────┼─ field.py ──┐  ┌──────────────────────────────────────────┐     ├─ vLLM          borde
                  │  (lectura)  │  │ security.py   API key · mTLS (uvicorn)   │     └─ OpenAI-compat nube
 Broker MQTT ─────┼─ mqtt_      ├─▶│               reglas anti prompt-injection│          (apagada por defecto)
  plant/+/        │  bridge.py  │  │ resilience.py token bucket · caché (Redis)│
  telemetry       │             │  │ router.py     complejidad → plan          │     por motor:
 REST/SCADA ──────┴─ app.py ────┘  │               single-flight por clave     │──▶  CircuitBreaker
  POST /v1/infer                   │               presupuesto deadline (3 s)  │     (closed/open/half-open)
  POST /v1/jobs ── jobs.py ───────▶│               recarga al abrir breaker    │◀──  JSON ModelOutput
  (cola Redis, N workers)          │ validation.py schema · física · premisa   │     (guided decoding)
                                   │ audit.py      DuckDB, cadena SHA-256      │
                                   └───────────────┬──────────────────────────┘
                                                   │ ControlAction validada
             Actuadores / SCADA                    ▼
             ──────────────────    plant/<asset>/recommendation (MQTT, QoS 1)
             Setpoint del PLC  ◀── escritura field.py — solo si mode=closed_loop
                                   Y ALLOW_PLC_WRITEBACK=true Y acción validada
                                   Y no viene de caché/failsafe
```

Camino de `POST /v1/infer` (el puente MQTT llama al mismo router):

1. Pydantic valida la solicitud: solo tags conocidos, valores finitos, ≤ 64 lecturas.
2. La nota del operador (único texto libre) se normaliza (NFKC, sin caracteres de control
   ni de ancho cero) y se revisa contra reglas de inyección. Si se marca, se descarta, la
   telemetría igual se evalúa y el evento queda auditado.
3. Búsqueda en caché con una clave que agrupa cada lectura en buckets del 1% del rango del
   tag. Un acierto se re-verifica contra la física con las lecturas *actuales*. Las
   solicitudes idénticas que ya están en vuelo esperan ese resultado en vez de volver a
   pegarle al GPU.
4. Un puntaje de complejidad (lecturas fuera de rango, calidad mala, largo de la nota)
   elige el plan: borde primero; nube primero solo sobre el umbral y solo con
   `ALLOW_CLOUD=true`.
5. Cada motor se llama a través de su circuit breaker con lo que queda del deadline.
   Timeout, error de conexión, breaker abierto o salida fuera de contrato → siguiente motor.
   Sin opciones → `HOLD` con `tier=failsafe`. El SCADA siempre recibe un 200 y una acción
   dentro del deadline, nunca un 5xx porque un modelo se cayó.
6. La acción se valida (schema → física → premisa), se audita y solo entonces se escribe,
   opcionalmente, al PLC.

### Resultados (modelos reales, datos reales de planta)

Medido en una RTX 4060 Ti 8 GB con Ollama 0.12.6, a través del mismo `InferenceRouter` que
usa el servicio (deadline 3 s). Telemetría: promedios horarios del
[dataset público de la planta de flotación](https://www.kaggle.com/datasets/edumagalhaes/quality-prediction-in-a-mining-process)
(737.453 filas, marzo–septiembre 2017), 60 horas al azar y 60 horas del decil superior de
sílice en concentrado (≥ 4,14%), semilla 42. Se reproduce con
`python scripts/bench_gateway.py`; los números crudos están en
`outputs/reports/gateway_bench.json`.

![Benchmark del gateway](docs/figures/gateway_bench.png)

| Modelo | Conjunto | Contrato válido | Premisa contradicha | Failsafe | Acciones | p50 / p95 (ms) |
|---|---|---|---|---|---|---|
| `qwen2.5:0.5b-instruct` | al azar | 100% | 50% | 0% | RAISE_ALARM 60 | 932 / 1733 |
| `qwen2.5:1.5b-instruct` | al azar | 100% | 100% | 0% | HOLD 60 | 1908 / 2048 |
| `qwen2.5:3b-instruct` | al azar | 100% | 0% | 0% | HOLD 60 | 1829 / 2084 |
| `llama3.2:3b` | al azar | 96,7% | 0% | 3,3% | HOLD 60 | 1673 / 1938 |
| `phi3.5:3.8b` | al azar | 0% | 0% | 100% | HOLD 60 | 3010 / 3019 |
| `qwen2.5:0.5b-instruct` | sílice alta | 100% | 40% | 0% | RAISE_ALARM 60 | 897 / 1663 |
| `qwen2.5:1.5b-instruct` | sílice alta | 100% | 100% | 0% | HOLD 60 | 1760 / 2002 |
| `qwen2.5:3b-instruct` | sílice alta | 100% | 0% | 0% | HOLD 60 | 1617 / 2076 |
| `llama3.2:3b` | sílice alta | 98,3% | 0% | 1,7% | HOLD 60 | 1420 / 1898 |
| `phi3.5:3.8b` | sílice alta | 0% | 0% | 100% | HOLD 60 | 3013 / 3020 |


El dataset no trae una etiqueta de "acción correcta", así que esto mide lo que el gateway
puede garantizar — cumplimiento de contrato, seguridad física, latencia — no la calidad de
la decisión.

Lo que dicen los números:

- **Contrato**: con el schema de unión, qwen2.5 0.5b/1.5b/3b produjeron una acción válida
  siempre; llama3.2:3b falló una vez por conjunto con un JSON cortado por el deadline.
  phi3.5:3.8b no cabe: superó el deadline de 3 s en 60/60 horas, e incluso con 30 s de
  presupuesto se pasó en 29/60 (al azar) y 28/60 (sílice alta), así que no es un modelo de
  lazo de control en este GPU.
- **Física**: ninguna respuesta propuso `ADJUST_SETPOINT`, así que las verificaciones
  físicas no rechazaron nada (0% en todas las filas). **Ningún modelo, en ninguna hora
  —incluidas las 60 horas con sílice promedio de 4,77%— propuso mover un setpoint de
  reactivo.** Detrás hay tres comportamientos distintos: qwen2.5:0.5b levantó alarma en
  todas las horas, qwen2.5:1.5b siempre mantuvo pero citando un motivo que los datos
  contradicen (100% de las horas), y los modelos de 3B mantuvieron con un motivo
  consistente. Un SLM con solo una foto de telemetría y reglas genéricas es conservador
  hasta no aportar información: el gateway lo vuelve seguro, no útil. Para que sea útil
  hace falta contexto propio de la planta (el soft sensor, tendencias recientes, historial
  de operadores) o fine-tuning, no un guardrail más grande. Eso es la sección siguiente.
- **La verificación de premisas se gana su lugar**: sin ella, qwen2.5:1.5b se vería como un
  modelo perfecto (100% válido, HOLDs razonables) estando equivocado en el *por qué* en
  todas las horas. La latencia varió entre corridas en el mismo hardware (p50 de
  qwen2.5:0.5b fue 567 ms en una corrida previa y 932 ms en esta); la tabla es la corrida
  guardada en el JSON.


#### Caída del motor

`scripts/failover_demo.py` envía una hora real de planta por segundo al stack desplegado,
detiene el contenedor de Ollama y lo vuelve a levantar:

![Línea de tiempo de la caída](docs/figures/gateway_failover.png)

81 solicitudes, **81 HTTP 200, ninguna sobre 2,52 s** (deadline 3 s). Las primeras fallas
tras la detención costaron hasta un timeout completo del motor (2,5 s); a la tercera el breaker se
abrió y el HOLD failsafe volvió con mediana de **10,7 ms** sin esperar al motor caído. La
sonda half-open 15 s después falló una vez más (el motor seguía abajo) y reabrió el breaker.
El motor volvió en t = 36 s sin el modelo cargado; la recarga en segundo plano lo cargó con
el breaker aún abierto, así que la sonda siguiente (t = 48,7 s) funcionó y el borde volvió a
~950 ms de mediana, sin intervención manual. Durante esas 34 respuestas failsafe el PLC
recibió un `HOLD` válido cada segundo: nunca un error, nunca un timeout.

### Hacerlo útil: contexto de planta y evaluaciones de utilidad

Los resultados de arriba terminan en "seguro, no útil": con una sola foto de telemetría,
ningún modelo propuso nunca cambiar un setpoint. Esta sección es la continuación: ¿el
contexto operativo (tendencia de laboratorio, un soft sensor, tendencias de setpoints)
hace que un modelo pequeño proponga el ajuste correcto, y *solo* cuando corresponde?

**Qué cuenta como "correcto".** Los datos de la planta no pueden decir qué setpoint era el
óptimo:
[flotation-plant-silica-soft-sensor](https://github.com/Rxyxs/flotation-plant-silica-soft-sensor)
mostró que cuatro modelos entrenados en meses distintos recomiendan movimientos opuestos.
Lo que sí respaldan los datos es la práctica de la propia planta. En flotación catiónica
inversa la amina es el colector que flota el cuarzo, y los operadores de esta planta suben
la dosis de amina después de que sube la sílice. La referencia es esa práctica escrita como
regla explícita (`src/gateway/policy.py`), no un óptimo. Cada hora de evaluación se etiqueta:

| Etiqueta | Regla | Respuesta correcta |
|---|---|---|
| `ADJUST` | dos últimos resultados de laboratorio ≥ 4,14% (p90 horario), el último no baja más de 0,3 puntos, soft sensor a menos de un error típico del umbral, pH y densidad en rango, ≥ 10 m³/h de holgura de amina | subir `amina_flow` |
| `HOLD` | dos últimos resultados y soft sensor ≤ 2,5%, pH y densidad en rango | no cambiar nada |
| `HOLD_NEAR_MISS` | sílice alta pero el procedimiento dice esperar: solo el último resultado ≥ 4,14%, o los dos altos y el último bajando más de 0,3 puntos | no cambiar nada |

La clase near-miss es la que hace informativa la prueba. Un modelo que solo aprendió
"sílice alta → ajustar" saca puntaje perfecto en las dos primeras y falla la tercera.

La evaluación usa 40 horas por clase (120 en total), todas posteriores al 21 de julio de
2017, fuera de muestra para el soft sensor (`data/evals/utility_cases.json`, generado por
`scripts/build_plant_context.py`). Las 40 horas `ADJUST` vienen de 11 días distintos, así
que la muestra efectiva es menor que 40 eventos independientes.

**Diseño del contexto** (`src/gateway/context.py`, `prompts.py`):

- *La aritmética la hace el código, no el modelo.* Promedios, pendientes, rachas y holguras
  se calculan y se entregan como hechos cortos con el número exacto:
  `[lab_last] latest lab silica in concentrate = 4.89% (sample taken 2 h ago; high means >= 4.14%)`.
  Nunca se muestra un resultado de laboratorio antes de que la planta lo tendría (2 h de
  retraso del laboratorio).
- *Soft sensor.* El Ridge sobre variables de proceso más los últimos resultados de
  laboratorio conocidos, portado del repo de flotación, re-entrenado en orden temporal con
  el 70% inicial de las horas de laboratorio y exportado como coeficientes, así la imagen
  del gateway no necesita scikit-learn. Fuera de muestra: **RMSE 0,808 contra 0,918 de la
  persistencia** (12% menos). Viaja con su error, y se encoge hacia la media: en las horas
  de prueba nunca estimó ≥ 4,14%, aunque el laboratorio mostrara sílice alta dos veces
  seguidas.
- *Citas verificables.* La salida tiene un campo `evidence` cuyos elementos son un enum de
  los ids de los hechos de *esta* solicitud, puesto primero en el schema para que el modelo
  cite antes de decidir. Una afirmación que los datos contradicen (`SILICA_HIGH` sin citar
  un hecho de sílice que esté alto) bloquea un ajuste.
- *Los valores inseguros no se pueden escribir.* `new_value` es un **entero acotado por
  solicitud** a `[actual − max_step, actual + max_step]` dentro del rango del tag, con una
  variante por tag escribible. Medido en Ollama 0.12.6, la gramática **respeta los límites
  de enteros, el `maxLength` de strings, los enums y las constantes, pero no
  `minimum`/`maximum` en números decimales**. Ese último hueco explica el `confidence: 3` de
  más arriba, y es la razón de que el setpoint sea entero. La verificación física sigue
  corriendo como segunda capa (y para motores en la nube que pueden ignorar el schema).
- *Ids semánticos.* Con ids posicionales (`F1, F2…`) el hecho de la amina era `F4` en los
  ejemplos few-shot y `F5` en las solicitudes reales, y qwen2.5:1.5b copiaba tal cual la
  lista de ids de los ejemplos. Ahora los ids son nombres estables (`lab_last`,
  `soft_sensor`, `amina_flow`).

El contexto llega al gateway por tres vías: dentro de la solicitud (`history`,
`lab_results`), por `POST /v1/lab` / MQTT `plant/<asset>/lab`, y desde el `ContextStore`,
que convierte en promedios horarios la telemetría que ya pasa por el gateway.

**Ablación.** Seis variantes, cuatro modelos, 120 horas reales, a través del router de
producción (`scripts/eval_utility.py`). Los resultados crudos, con cada caso, están en
`outputs/reports/utility_eval.json`.

| | Variante | Qué recibe el modelo |
|---|---|---|
| A | foto | solo la telemetría actual (el gateway original) |
| B | contexto | hechos + procedimiento operativo escrito |
| C | + few-shot | B + 5 ejemplos resueltos |
| D | herramientas | los mismos hechos, pedidos por el modelo con function calling |
| E | + estado derivado | B + el paso 1 del procedimiento calculado en código: `silica_state = SUSTAINED_HIGH / SINGLE_HIGH / RECOVERING / NORMAL` |
| F | compuerta del procedimiento | E, y la gramática solo ofrece "subir `amina_flow`" cuando el procedimiento (en código) permite un ajuste; si no, solo `HOLD` / `RAISE_ALARM` |

![Ablación de utilidad](docs/figures/utility_ablation.png)

| Modelo | Variante | Ajuste correcto | Ajusta en near-miss | Ajusta al revés | Alarma en hora tranquila |
|---|---|---|---|---|---|
| `qwen2.5:1.5b-instruct` | A | 0% | 0% | 0% | 95% |
| `qwen2.5:1.5b-instruct` | B | 100% | 100% | 0% | 0% |
| `qwen2.5:1.5b-instruct` | C | 27,5% | 47,5% | 0% | 0% |
| `qwen2.5:1.5b-instruct` | D | 90% | 100% | 10% | 0% |
| `qwen2.5:1.5b-instruct` | E | 97,5% | 100% | 2,5% | 0% |
| `qwen2.5:1.5b-instruct` | F | **100%** | **0%** | **0%** | **0%** |
| `qwen2.5:3b-instruct` | A | 0% | 0% | 0% | 100% |
| `qwen2.5:3b-instruct` | B | 50% | 100% | 50% | 0% |
| `qwen2.5:3b-instruct` | C | 80% | 80% | 0% | 0% |
| `qwen2.5:3b-instruct` | D | 40% | 100% | 60% | 0% |
| `qwen2.5:3b-instruct` | E | 40% | 0% | 60% | 0% |
| `qwen2.5:3b-instruct` | F | **100%** | **0%** | **0%** | **0%** |
| `llama3.2:3b` | A | 0% | 0% | 0% | 100% |
| `llama3.2:3b` | B | 45% | 85% | 52,5% | 62,5% |
| `llama3.2:3b` | C | 0% | 0% | 0% | 22,5% |
| `llama3.2:3b` | D | 70% | 35% | 15% | 100% |
| `llama3.2:3b` | E | 22,5% | 12,5% | 10% | 52,5% |
| `llama3.2:3b` | F | 32,5% | 0% | 0% | 100% |
| `qwen2.5:0.5b-instruct` | A | 0% | 0% | 0% | 100% |
| `qwen2.5:0.5b-instruct` | B | 0% | 0% | 0% | 100% |
| `qwen2.5:0.5b-instruct` | C | 0% | 0% | 0% | 95% |
| `qwen2.5:0.5b-instruct` | D | 7,5% | 82,5% | 90% | 0% |
| `qwen2.5:0.5b-instruct` | E | 0% | 0% | 0% | 82,5% |
| `qwen2.5:0.5b-instruct` | F | 0% | 0% | 0% | 82,5% |

Los motores corrieron con un deadline de 30 s porque el GPU estuvo compartido con otras
cargas durante parte de la corrida; las decisiones no dependen del deadline, pero la
latencia sí. La configuración recomendada se volvió a correr sola en el GPU con el deadline
de producción de 3 s:

| Modelo | Variante | Ajuste correcto | Ajusta en near-miss | Al revés | Failsafe | p50 / p95 |
|---|---|---|---|---|---|---|
| `qwen2.5:1.5b-instruct` | F | 100% | 0% | 0% | 0% | 1,03 / 1,16 s |
| `qwen2.5:1.5b-instruct` | B | 95% | 100% | 5% | 0% | 1,16 / 1,24 s |
| `qwen2.5:3b-instruct` | F | 97,5% | 0% | 0% | 0,8% | 1,42 / 1,82 s |
| `qwen2.5:3b-instruct` | B | 37,5% | 70% | 62,5% | 0% | 1,75 / 2,00 s |

Lo que muestra la ablación:

1. **El contexto es lo que destraba los ajustes.** Con una foto (A) ningún modelo ajusta, y
   los de 3B levantan alarma en todas las horas tranquilas. Con hechos (B), qwen2.5:1.5b
   propone el movimiento correcto en el 100% de las horas que lo necesitan.
2. **Pero ningún prompt hace que un modelo pequeño siga el procedimiento.** En B, D y E,
   qwen2.5:1.5b ajusta en el **100% de las horas near-miss**, incluso en E, donde el hecho
   dice literalmente
   `SINGLE_HIGH: only the latest lab result is >= 4.14%: not sustained yet, wait for the next one`.
   Los ejemplos few-shot (C) lo *empeoraron* para el modelo de 1,5B (27,5% correcto, 47,5%
   near-miss). qwen2.5:3b, con libertad, bajó la amina en el 50-60% de sus ajustes en B, D y
   E. El function calling (D) no cambió nada sustancial y multiplicó la latencia por 1,7 a
   2,7.
3. **La regla va en código, impuesta por la gramática.** En F el procedimiento decide en
   código *si* existe un ajuste y en *qué dirección*, y la gramática ofrece solo eso.
   qwen2.5:1.5b y qwen2.5:3b llegan entonces a 100% de ajustes correctos con 0% de ajustes
   en near-miss, 0% de movimientos al revés y 0% de alarmas en horas tranquilas. El 0% en
   near-miss de F se cumple *por construcción*, no porque el modelo haya aprendido algo; lo
   que F mide es si el modelo igual toma el ajuste cuando se le ofrece y elige un valor
   razonable dentro del límite.
4. **No todos los modelos se rescatan.** qwen2.5:0.5b levanta alarma en la mayoría de las
   horas tranquilas en cinco de las seis variantes, y llama3.2:3b levanta alarma en todas las horas tranquilas en F. El
   gateway los mantiene seguros, y las evaluaciones de utilidad son las que dicen que no hay
   que desplegarlos.
5. **La explicación no es evidencia.** Aun con los campos estructurados correctos, el texto
   libre traía aritmética falsa ("4.64 < 4.14", "4.14% >= 5.01%"). Los campos estructurados
   se verifican; la prosa se le muestra al operador y nunca se le cree.

Dónde queda el SLM: en esta tarea la decisión es una regla, y una regla la ejecuta mejor el
código. El modelo se gana su lugar donde las reglas se acaban: leer una nota del operador,
elegir un valor dentro del límite permitido, explicar la decisión. Lo hace dentro de un
espacio que el procedimiento ya volvió seguro. Una planta que quiera que el modelo decida
más necesita o un modelo que pase estas evaluaciones, o los datos de step-tests que
justifiquen una regla más rica; no un prompt más largo.

Variación entre corridas: las corridas de 30 s y de 3 s de la misma configuración difirieron
en algunos puntos (qwen2.5:1.5b B: 100% contra 95% correcto) a temperatura 0, así que una o
dos horas de diferencia entre variantes no son significativas.

#### Pruebas de utilidad en pytest

`tests/evals/test_utility.py` corre la configuración recomendada contra un Ollama real y
falla si el modelo deja de proponer el ajuste (< 80% de las horas `ADJUST`), ajusta una hora
tranquila o near-miss (> 5%), mueve la amina al revés (> 5%) o no responde a tiempo. Se salta
si no hay un Ollama disponible, así que nunca bloquea la suite principal.

```bash
OLLAMA_URL=http://127.0.0.1:11434 UTILITY_MODEL=qwen2.5:1.5b-instruct pytest tests/evals -v -s
```

El workflow `Utility evals` la corre semanalmente y a pedido en un runner de GitHub, con
Ollama en CPU. Validado localmente con 4 núcleos de CPU y sin GPU: 30 casos (10 por clase)
en 7,5 minutos, 5/5 pruebas pasando (100% de ajustes correctos, 0% near-miss, 0% al revés).
En el runner de GitHub los mismos 30 casos tardan 10,9 minutos y pasan igual.

### Endpoints

| Método | Ruta | Propósito |
|---|---|---|
| `POST` | `/v1/infer` | Inferencia síncrona acotada por `GATEWAY_DEADLINE_S`. Cuerpo: `InferenceRequest`. Devuelve `InferenceResponse`; headers `X-Request-ID`, `X-PLC-Written`, `X-PLC-Write-Note`. |
| `POST` | `/v1/jobs` | Inferencia diferida (reportes de turno, diagnóstico por lotes) → `202 {"job_id"}`. Corre en un pool fijo de workers para no competir con el lazo de control por el GPU. |
| `GET` | `/v1/jobs/{job_id}` | `queued` / `running` / `done` (con resultado) / `failed`. |
| `GET` | `/v1/backends` | Estado del circuit breaker por motor y estado del warm-up. |
| `GET` | `/v1/audit?limit=50` | Últimas decisiones más `chain_ok` (verificación de la cadena de hashes). |
| `POST` | `/v1/lab` | Resultado de laboratorio de sílice en concentrado (`asset_id`, `sampled_at`, `silica_pct`, `delay_h`) → contexto del activo. |
| `GET` | `/v1/context/{asset_id}` | Los hechos que vería el modelo ahora para ese activo. |
| `GET` | `/v1/schema` | JSON Schemas: `scada_action` (lo que parsea el PLC) y `model_output` (lo que se le impone al motor). |
| `GET` | `/health` | Liveness; no toca los motores. |
| `GET` | `/metrics` | Prometheus: `slmgw_inference_total{tier,action,degraded}`, `slmgw_inference_latency_seconds`, `slmgw_rate_limited_total`. |

Todos los endpoints salvo `/health`, `/metrics` y `/v1/schema` exigen `X-API-Key` cuando
`GATEWAY_API_KEYS` está definido. Los errores son `{"error": ..., "type": ...}`; el `429`
trae `Retry-After`.

```bash
curl -s -H "X-API-Key: $KEY" -H "Content-Type: application/json" localhost:8080/v1/infer -d '{
  "asset_id": "flot-col-01", "mode": "advisory",
  "telemetry": [{"tag": "amina_flow", "value": 557.4}, {"tag": "ore_pulp_ph", "value": 10.07},
                {"tag": "silica_concentrate_pct", "value": 4.9}]}'
```

MQTT: publicar un `InferenceRequest` en `plant/<asset_id>/telemetry`; la respuesta llega en
`plant/<asset_id>/recommendation` (QoS 1) con `plc_written` y `plc_write_note` agregados.
Un payload mal formado, o un `asset_id` que no coincide con el tópico, se responde en
`plant/<asset_id>/rejected`.

### Guía de despliegue en planta

```bash
cp deploy/edge/.env.example deploy/edge/.env     # como mínimo, GATEWAY_API_KEYS
docker compose -f deploy/edge/docker-compose.yml up -d --build
curl -s -H "X-API-Key: <clave>" localhost:8080/v1/backends   # warmup: {"edge-ollama": "ready"}
python scripts/replay_flotation.py --csv data/flotation/MiningProcess_Flotation_Plant_Database.csv --n 20
```

El stack es el gateway, Mosquitto, Redis (AOF activo, tope LRU) y Ollama con reserva de
GPU; `ollama-pull` descarga el modelo una vez a un volumen. `--profile vllm` agrega un
servidor vLLM con guided decoding xgrammar (apuntar `EDGE_VLLM_URL` a él); ese perfil está
declarado pero no se corrió para los resultados de arriba. En un Jetson, construir la imagen
del gateway para `linux/arm64`
(`docker buildx build --platform linux/arm64 -f deploy/edge/Dockerfile .`) y reemplazar el
servicio `ollama` por una versión compatible con JetPack.

Checklist antes de conectar a una planta real:

- **La escritura queda apagada** (`ALLOW_PLC_WRITEBACK=false`) hasta que el modo asesor
  haya corrido lo suficiente para que operaciones confíe en él. Escribir exige además
  `mode=closed_loop` en la solicitud, un `ADJUST_SETPOINT` validado y una respuesta que no
  venga de caché ni de failsafe; `FieldWriter.apply()` vuelve a verificar las cuatro.
- **Registro de tags**: reemplazar los límites de `src/gateway/tags.py` por la envolvente
  operativa de la planta y los node ids OPC UA / registros Modbus y escalas del PLC.
- **mTLS**: dejar `server.crt`, `server.key` y la CA de planta en `deploy/edge/tls/` y
  definir `TLS_CERT_FILE`, `TLS_KEY_FILE`, `TLS_CA_FILE`; uvicorn exige entonces un
  certificado de cliente firmado por esa CA antes de que cualquier solicitud llegue a la API.
- **MQTT**: el `mosquitto.conf` incluido es de demostración (anónimo en la red interna).
  Activar `password_file` y el listener TLS en 8883.
- **Seguridad OPC UA**: definir `OPCUA_SECURITY`, p. ej. `Basic256Sha256,SignAndEncrypt,cert.der,key.pem`.
- **Red aislada**: dejar `ALLOW_CLOUD=false` (default). El router nunca planifica una
  llamada a la nube, sea cual sea el puntaje de complejidad.
- **Auditoría**: el archivo DuckDB vive en el volumen `audit-data`. Respaldarlo;
  `GET /v1/audit` reporta `chain_ok=false` y el primer `seq` roto si alguna fila se editó o
  borró.

Las variables de entorno están en la tabla de la versión en inglés de este README
(los nombres son los mismos).

### Pruebas de integración

```bash
python -m venv venv && venv/Scripts/pip install -r requirements-gateway-dev.txt   # Linux: venv/bin/pip
pytest tests/gateway -v
```

106 pruebas, sin GPU ni red:

| Archivo | Qué ejercita |
|---|---|
| `test_validation.py` | Contrato de salida del modelo, límites y pasos físicos, premisas, forma del schema guiado. |
| `test_breaker.py` | Transiciones closed → open → half-open → closed/open con reloj inyectado. |
| `test_router.py` | Fallback borde→nube, breaker que salta un motor caído, corte por deadline, failsafe, sin buscar otro backend tras un rechazo físico, inyección, caché, single-flight, recarga, concurrencia + cadena de auditoría intacta. |
| `test_security_audit.py` | Reglas de inyección (inglés/español, ancho cero escondido), notas operativas benignas, API keys, token bucket, cadena de hashes entre reinicios y detección de manipulación. |
| `test_api.py` | Contrato REST: 401/422/429, control de escritura, setpoint inseguro nunca escrito, jobs, auditoría, métricas. |
| `test_context.py` | Hechos y visibilidad según el retraso del laboratorio, artefacto del soft sensor contra persistencia, regla de referencia, verificación de citas, schema acotado por solicitud, formato few-shot, `ContextStore`, compuerta del procedimiento, y que cada caso guardado reproduzca su etiqueta. |
| `test_protocols.py` | Servidor OPC UA real (asyncua) y esclavo Modbus TCP real (pymodbus) en sockets locales: lectura, escritura, escala de registros, lazo cerrado Modbus → router → Modbus; ruteo del puente MQTT; caché y cola en Redis (fakeredis). |

El job de CI `plant-gateway` corre esta suite y construye la imagen de borde.

### Datos

Los scripts de benchmark y replay leen el dataset de Kaggle
[Quality Prediction in a Mining Process](https://www.kaggle.com/datasets/edumagalhaes/quality-prediction-in-a-mining-process)
(`edumagalhaes`, CC0-1.0). No se guarda en el repositorio:

```bash
kaggle datasets download -d edumagalhaes/quality-prediction-in-a-mining-process -p data/flotation --unzip
```

## Estructura del repositorio

```
src/
  engine/       LLMServer (GGUF/llama.cpp), benchmarks (CLI real: run_suite,
                cold/cached TTFT, throughput extremo-a-extremo/decodificación)
  api/          Gateway FastAPI, contrato OpenAI + /v1/pdm/diagnose, métricas
                Prometheus
  agents/       Pipeline de chat: RouterAgent -> AnalyticsAgent (si aplica,
                con evaluación MaintenanceAdvisorAgent advisor) ->
                VerifierAgent (fidelidad numérica + formato) ->
                SafetyComplianceAgent (límites físicos, bloqueante) -- todos
                locales y sin red en cada solicitud. PdMAgent aparte:
                diagnóstico multi-métrica, expuesto vía diagnose_asset()
                y /v1/pdm/diagnose, no entra en ese pipeline
  tools/        ToolRegistry, herramientas industriales (query_duckdb,
                sensor_anomaly_check, calculate_rul)
  guardrails/   Validación de SQL y de salidas JSON estrictas
  evaluation/   FaithfulnessEvaluator (DeepEval, requiere judge externo) +
                run_offline_eval.py (evaluación real sin judge: Safety Block
                Rate desglosado, sobre 100 prompts de dominio)
  training/     Pipeline QLoRA (dataset_prep.py, finetune.py)
data/
  domain_dataset/    Dataset sintético ChatML de dominio (telemetría/sensores)
  eval_prompts.json  100 prompts de evaluación con horas reales de planta (4 categorías x 25)
  models/            Pesos GGUF locales (no versionado; ver instalación)
scripts/
  quantize.py             Verifica/carga modelos GGUF cuantizados (Q4_K_M, Q8_0)
  generate_plots.py       Genera los gráficos de `outputs/reports/` (ver Benchmarks)
  build_eval_prompts.py   Genera data/eval_prompts.json (100 prompts, ver Benchmarks)
  verify_finetuning_pipeline.py  Valida el pipeline QLoRA de punta a punta sin
                          GPU (dataset, config, intento de carga del modelo
                          base) y guarda el estado real en finetuning_metrics.json
outputs/
  reports/        Gráficos versionados que embeben el README (PNG), más
                   gguf_benchmark.json (última corrida real de CPU),
                   eval_metrics.json (evaluación offline real, 100 prompts) y
                   finetuning_metrics.json (estado verificado del pipeline QLoRA)
  gateway/      Gateway de planta: router, breaker, backends (Ollama/OpenAI-compatible),
                schemas + validación, seguridad, caché/rate limit, jobs, auditoría
                DuckDB, protocols/ (puente MQTT, OPC UA, Modbus TCP), serve.py
deploy/edge/    Dockerfile (CPU, amd64/arm64) y docker-compose.yml (gateway +
                Mosquitto + Redis + Ollama GPU, perfil vLLM opcional)
monitoring/
  prometheus.yml  Configuración de scraping para el contenedor Prometheus
tests/
  test_engine.py, test_api.py, test_agents.py, test_router_agent.py,
  test_analytics_agent.py, test_maintenance_advisor.py, test_pdm_agent.py,
  test_safety_agent.py, test_tools.py, test_guardrails.py, test_eval.py,
  test_offline_eval.py, test_training.py, test_integration.py (ver
  `pytest -v` para el listado completo y vigente)
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

El servicio nunca descarga el modelo por red (ver [Guía de despliegue
air-gapped](#guía-de-despliegue-air-gapped)): `MODEL_PATH` tiene que apuntar a
un `.gguf` que ya exista en disco -- este repo no lo incluye (pesa GBs, está
en `.gitignore`). Los benchmarks de este README se midieron con
**Qwen2.5-3B-Instruct, cuantización Q4_K_M** (buscar `Qwen2.5-3B-Instruct-GGUF`
en Hugging Face); cualquier GGUF compatible con llama.cpp sirve para levantar
el servicio.

```bash
pip install -r requirements.txt
export MODEL_PATH=data/models/model.gguf   # el .gguf ya tiene que estar ahí
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Para correr la suite de tests o los agentes sin pesos reales, `LLMServer` se
simula (ver [Pruebas](#pruebas) abajo) -- no hace falta un `.gguf` para eso.

## Ejecutar con Docker Compose

```bash
docker compose up -d --build
```

Expone:
- API: `127.0.0.1:8000` (`/v1/chat/completions`, `/v1/pdm/diagnose`, `/v1/models`, `/health`, `/metrics`)
- Prometheus: `127.0.0.1:9090`, con scraping ya configurado hacia
  `slm-api:8000/metrics` (ver `monitoring/prometheus.yml`)

Ambos puertos están atados a `127.0.0.1` deliberadamente: el servicio no se
expone a la red pública por defecto.

## Pruebas

```bash
pytest -v --ignore=tests/gateway   # gateway de chat (requirements.txt)
pytest -v tests/gateway            # gateway de planta (requirements-gateway-dev.txt)
```

Por módulo:

```bash
pytest tests/test_engine.py -v        # LLMServer, fallback GPU→CPU, benchmarks
pytest tests/test_api.py -v           # Contrato HTTP, LLM mockeado
pytest tests/test_agents.py -v        # AnalyticsAgent/VerifierAgent + AgentOrchestrator + API end-to-end
pytest tests/test_router_agent.py -v  # RouterAgent: guardrail de entrada + clasificación de intención
pytest tests/test_analytics_agent.py -v  # AnalyticsAgent en aislamiento (dispatch real, SQL peligroso)
pytest tests/test_maintenance_advisor.py -v  # MaintenanceAdvisorAgent: interpreta RUL/anomalía, advisor (nunca lanza)
pytest tests/test_pdm_agent.py -v     # PdMAgent: diagnóstico multi-métrica, RUL, confianza, degradación con gracia
pytest tests/test_safety_agent.py -v  # SafetyComplianceAgent: límites físicos de diseño, bloqueante
pytest tests/test_tools.py -v         # Tools industriales + ToolRegistry
pytest tests/test_guardrails.py -v    # Validación de SQL y de JSON de salida
pytest tests/test_eval.py -v          # Evaluador de fidelidad (métricas mockeadas)
pytest tests/test_offline_eval.py -v  # Harness de evaluación offline: Safety Block Rate, IC de Wilson (sin SLM real)
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
  ni parcialmente. No confundir con `MaintenanceAdvisorAgent` (advisor, adjunta
  `maintenance_alert` sin bloquear nada) ni con `PdMAgent` (diagnóstico
  explícito vía `/v1/pdm/diagnose`, fuera del pipeline de chat) — de los tres,
  solo `SafetyComplianceAgent` bloquea.

## Resiliencia operativa

- **Formato de error uniforme**: toda respuesta de error del gateway, sin
  excepción, tiene la forma `{"error": "<mensaje>", "type": "<NombreDeLa
  Excepcion>"}` — nunca el `{"detail": "..."}` por defecto de FastAPI. Cada
  `except` de `src/api/routes.py` levanta `HTTPException(detail={"error":...,
  "type":...})`, y `http_exception_handler` aplana ese `detail` al body de la
  respuesta en vez de anidarlo una vuelta más.
- **Red de seguridad para excepciones no mapeadas**: `unhandled_exception_handler`
  (`@app.exception_handler(Exception)`) atrapa cualquier error que ningún
  `except` específico haya previsto, lo loguea con su `request_id` y devuelve
  el mismo formato uniforme en vez de la página de traceback por defecto de
  Starlette — verificado con un error forzado que ningún handler del endpoint
  mapea (`tests/test_api.py::test_unhandled_exception_returns_clean_json_500_not_a_traceback_page`).
- **Logging estructurado**: `request_context_middleware` asigna un
  `request_id` por solicitud (hereda `X-Request-ID` si el cliente ya mandó
  uno; si no, genera un UUID), lo devuelve en la respuesta, y loguea inicio y
  fin de cada solicitud como una línea JSON (`JSONLogFormatter`) con
  timestamp, nivel, mensaje y `request_id` — sin depender de una librería de
  logging externa. Verificado corriendo el servidor real (`uvicorn`), no solo
  con `TestClient`.
- **Rate limiting: documentado, no implementado.** Este gateway no impone
  ningún límite de tasa por sí mismo. Para un despliegue real, la recomendación
  es un proxy inverso delante (nginx `limit_req`, Envoy, o el rate limiting
  nativo del API Gateway/Ingress que ya exista en el clúster) en vez de
  agregarlo acá: es responsabilidad de la capa de red, no del servicio, y
  agregar una dependencia nueva (`slowapi`, etc.) solo para esto no se
  justificaba en el alcance de esta semana.

## Production Readiness Checklist

Cada fila se verificó de verdad antes de marcarse -- dos se corrigieron
respecto al plan original porque el número/alcance ahí no coincidía con el
estado real del repositorio (ver la nota de cada una).

| # | Ítem | Estado | Nota |
|---|------|:------:|------|
| 1 | CI/CD automatizado (GitHub Actions) | ✅ | `.github/workflows/ci.yml`, push/PR a `master`/`main`, `pytest -v` real -- **317 tests**, no "310+": recontado en esta corrida, no copiado del día anterior. |
| 2 | Motor GGUF local en CPU, desacoplado | ✅ | `src/engine/LLMServer`, `n_gpu_layers=0` forzable explícitamente, sin acoplar a `src/api`/`src/agents`. |
| 3 | Pipeline de agentes especializados | ✅ | **6, no 5**: `RouterAgent`, `AnalyticsAgent`, `VerifierAgent`, `MaintenanceAdvisorAgent`, `SafetyComplianceAgent`, `PdMAgent`. El plan original preveía 5 (Router/Analytics/Verifier/PdM/Safety); terminaron siendo 6 porque dos sesiones en paralelo construyeron implementaciones independientes y válidas de "el agente de mantenimiento predictivo" (ver historial de commits) y se conservaron ambas en vez de descartar una. |
| 4 | Guardrails de entrada + fidelidad de salida | ✅ | `RouterAgent.check_threat` (entrada) + `VerifierAgent` (fidelidad numérica de salida) + `SafetyComplianceAgent` (límites físicos) + validación Pydantic estricta en cada tool. |
| 5 | Evaluador offline de alucinaciones | ⚠️ Parcial | Lo que corre de verdad y sin judge externo es `run_offline_eval.py` (Safety Block Rate desglosado, 100 prompts reales). El `FaithfulnessMetric`/`AnswerRelevancyMetric` de DeepEval — la medición de alucinación propiamente dicha — sigue sin correr: necesita `OPENAI_API_KEY`, no disponible en este entorno. No se marca ✅ completo para no sobre-representarlo. |
| 6 | Pipeline de fine-tuning QLoRA verificado (CPU dry-run) | ✅ | `scripts/verify_finetuning_pipeline.py` + `tests/test_training.py`: dataset, `QLoRATrainingConfig`, `SFTConfig` construyen y validan sin GPU; `load_base_model_and_tokenizer` falla con un `ImportError` claro y esperado sin GPU/unsloth (verificado también en CI, Linux). Es un dry-run verificado, no un checkpoint entrenado -- ese paso sigue pendiente y requiere GPU real. |
| 7 | Endpoints FastAPI + métricas Prometheus | ✅ | `/v1/chat/completions`, `/v1/pdm/diagnose`, `/v1/models`, `/health`, `/metrics` (Prometheus vía `prometheus-fastapi-instrumentator` + métricas propias de tokens/latencia). |
| 8 | Manejo de errores uniforme + logging estructurado | ✅ | Agregado esta semana: `{"error", "type"}` en toda respuesta de error, handler de excepciones no mapeadas, logging JSON con `request_id` por solicitud (ver "Resiliencia operativa" arriba). |
| 9 | Rate limiting | ⬜ No implementado | Documentado como decisión deliberada (delegado a un proxy inverso), no como un olvido -- ver "Resiliencia operativa" arriba. |
