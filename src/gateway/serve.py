"""Punto de entrada del gateway: API REST (con mTLS opcional) + puente MQTT en el mismo proceso.

    python -m src.gateway.serve

mTLS: si existen TLS_CERT_FILE, TLS_KEY_FILE y TLS_CA_FILE, uvicorn exige
certificado de cliente firmado por esa CA (`ssl.CERT_REQUIRED`); un cliente sin
certificado válido no completa el handshake y nunca llega a FastAPI.
El puente MQTT arranca si MQTT_HOST está definido.
"""

from __future__ import annotations

import asyncio
import logging
import os
import ssl

import uvicorn

from src.gateway.app import create_app
from src.gateway.config import load_settings

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("slm_gateway")


def _field_writer(settings):
    from src.gateway.protocols.field import FieldWriter, ModbusClient, OpcUaClient

    if os.environ.get("OPCUA_URL"):
        return FieldWriter(OpcUaClient(os.environ["OPCUA_URL"], os.environ.get("OPCUA_SECURITY")),
                           settings.allow_plc_writeback)
    if os.environ.get("MODBUS_HOST"):
        return FieldWriter(ModbusClient(os.environ["MODBUS_HOST"], int(os.environ.get("MODBUS_PORT", "502"))),
                           settings.allow_plc_writeback)
    return FieldWriter(None, settings.allow_plc_writeback)


async def main() -> None:
    settings = load_settings()
    writer = _field_writer(settings)
    if writer.client is not None:
        await writer.client.connect()
    app = create_app(settings, field_writer=writer)

    tls = {}
    cert, key, ca = (os.environ.get(k) for k in ("TLS_CERT_FILE", "TLS_KEY_FILE", "TLS_CA_FILE"))
    if cert and key and ca:
        tls = dict(ssl_certfile=cert, ssl_keyfile=key, ssl_ca_certs=ca, ssl_cert_reqs=ssl.CERT_REQUIRED)
        log.info("mTLS enabled (client certificates required)")

    config = uvicorn.Config(app, host=os.environ.get("HOST", "0.0.0.0"),
                            port=int(os.environ.get("PORT", "8080")), log_config=None, **tls)
    server = uvicorn.Server(config)
    tasks = [asyncio.create_task(server.serve())]

    if os.environ.get("MQTT_HOST"):
        from src.gateway.protocols.mqtt_bridge import MqttBridge

        async def bridge_when_ready():
            while not server.started:
                await asyncio.sleep(0.1)
            bridge = MqttBridge(app.state.router, app.state.writer, os.environ["MQTT_HOST"],
                                int(os.environ.get("MQTT_PORT", "1883")),
                                os.environ.get("MQTT_USERNAME"), os.environ.get("MQTT_PASSWORD"))
            await bridge.run()

        tasks.append(asyncio.create_task(bridge_when_ready()))

    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for t in pending:
        t.cancel()
    if writer.client is not None:
        await writer.client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
