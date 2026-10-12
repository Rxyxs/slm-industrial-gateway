"""Puente MQTT bidireccional.

Suscribe:  plant/<asset_id>/telemetry        (payload = InferenceRequest JSON)
Publica:   plant/<asset_id>/recommendation   (payload = InferenceResponse JSON, QoS 1)
           plant/<asset_id>/rejected          (payload malformado o no autorizado)

Cada mensaje se procesa en su propia tarea, con un semáforo que acota cuántas
inferencias concurrentes salen del puente: una ráfaga de telemetría (reconexión
de un PLC que vacía su buffer) no puede saturar el motor del borde.
Si el broker se cae, el bucle se reconecta con backoff exponencial acotado.
"""

from __future__ import annotations

import asyncio
import json
import logging

from pydantic import ValidationError

from src.gateway.protocols.field import FieldWriter
from src.gateway.router import InferenceRouter
from src.gateway.schemas import InferenceRequest

log = logging.getLogger("slm_gateway.mqtt")

TELEMETRY_TOPIC = "plant/+/telemetry"


def asset_from_topic(topic: str) -> str | None:
    parts = topic.split("/")
    if len(parts) == 3 and parts[0] == "plant" and parts[2] == "telemetry":
        return parts[1]
    return None


class MqttBridge:
    def __init__(self, router: InferenceRouter, writer: FieldWriter, host: str, port: int = 1883,
                 username: str | None = None, password: str | None = None, tls_context=None,
                 max_inflight: int = 4):
        self.router = router
        self.writer = writer
        self.host, self.port = host, port
        self.username, self.password = username, password
        self.tls_context = tls_context
        self._sem = asyncio.Semaphore(max_inflight)
        self.processed = 0

    async def handle(self, client, topic: str, payload: bytes) -> None:
        asset = asset_from_topic(topic)
        if asset is None:
            return
        try:
            req = InferenceRequest.model_validate_json(payload)
            if req.asset_id != asset:
                raise ValueError(f"asset_id {req.asset_id} does not match topic {asset}")
        except (ValidationError, ValueError) as exc:
            await client.publish(f"plant/{asset}/rejected",
                                 json.dumps({"error": str(exc)[:300]}), qos=1)
            return
        async with self._sem:
            resp = await self.router.infer(req, client_id=f"mqtt:{asset}")
        written, why = await self.writer.apply(req, resp)
        out = resp.model_dump(mode="json") | {"plc_written": written, "plc_write_note": why}
        await client.publish(f"plant/{asset}/recommendation", json.dumps(out), qos=1)
        self.processed += 1

    async def run(self, stop: asyncio.Event | None = None) -> None:
        import aiomqtt

        backoff = 1.0
        stop = stop or asyncio.Event()
        while not stop.is_set():
            try:
                async with aiomqtt.Client(
                    self.host, self.port, username=self.username, password=self.password,
                    tls_context=self.tls_context, identifier="slm-gateway",
                ) as client:
                    await client.subscribe(TELEMETRY_TOPIC, qos=1)
                    log.info("mqtt connected to %s:%s", self.host, self.port)
                    backoff = 1.0
                    tasks: set[asyncio.Task] = set()
                    async for msg in client.messages:
                        t = asyncio.create_task(self.handle(client, str(msg.topic), msg.payload))
                        tasks.add(t)
                        t.add_done_callback(tasks.discard)
                        if stop.is_set():
                            break
                    await asyncio.gather(*tasks, return_exceptions=True)
            except aiomqtt.MqttError as exc:
                log.warning("mqtt disconnected (%s); retry in %.0fs", exc, backoff)
                try:
                    await asyncio.wait_for(stop.wait(), timeout=backoff)
                except asyncio.TimeoutError:
                    pass
                backoff = min(backoff * 2, 30.0)
