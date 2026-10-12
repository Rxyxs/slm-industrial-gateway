"""Protocolos de campo contra servidores reales levantados en el propio proceso:
un servidor OPC UA (asyncua) y un esclavo Modbus TCP (pymodbus). No hay mocks
del protocolo: el gateway habla por socket igual que lo haría con un PLC."""

import asyncio
import json
import socket

import fakeredis
import pytest

from src.gateway.jobs import RedisJobQueue
from src.gateway.protocols.field import FieldWriter, ModbusClient, OpcUaClient
from src.gateway.protocols.mqtt_bridge import MqttBridge, asset_from_topic
from src.gateway.resilience import RedisCache
from src.gateway.router import InferenceRouter
from src.gateway.schemas import InferenceRequest, InferenceResponse
from src.gateway.tags import TAG_REGISTRY
from tests.gateway.conftest import FakeBackend, action_json


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
async def opcua_server():
    from asyncua import Server, ua

    port = free_port()
    server = Server()
    await server.init()
    server.set_endpoint(f"opc.tcp://127.0.0.1:{port}/plant/")
    await server.register_namespace("urn:plant:flotation")  # ns=2
    obj = await server.nodes.objects.add_object(ua.NodeId("Flotation", 2), "Flotation")
    initial = {"amina_flow": 557.434, "starch_flow": 3019.53, "column_01_air_flow": 249.214,
               "column_01_level": 457.396}
    for tag, value in initial.items():
        node_id = ua.NodeId.from_string(TAG_REGISTRY[tag].opcua_node)
        var = await obj.add_variable(node_id, tag, ua.Variant(value, ua.VariantType.Double))
        await var.set_writable()
    async with server:
        yield f"opc.tcp://127.0.0.1:{port}/plant/"


async def test_opcua_read_and_write(opcua_server):
    c = OpcUaClient(opcua_server)
    await c.connect()
    try:
        before = await c.read(["amina_flow", "column_01_level"])
        assert [s.value for s in before] == [557.434, 457.396]
        await c.write("amina_flow", 545.0)
        (after,) = await c.read(["amina_flow"])
        assert after.value == 545.0 and after.quality == "good"
    finally:
        await c.disconnect()


@pytest.fixture
async def modbus_server():
    from pymodbus.datastore import ModbusDeviceContext, ModbusSequentialDataBlock, ModbusServerContext
    from pymodbus.server import ModbusTcpServer

    port = free_port()
    # pymodbus suma 1 a la dirección del bloque: el bloque arranca en 1 para
    # que el offset 0 del cliente sea el primer registro.
    block = ModbusSequentialDataBlock(1, [3020, 5574, 2492, 457] + [0] * 16)
    ctx = ModbusServerContext(devices=ModbusDeviceContext(hr=block), single=True)
    server = ModbusTcpServer(ctx, address=("127.0.0.1", port))
    task = asyncio.create_task(server.serve_forever())
    await asyncio.sleep(0.3)
    yield port
    await server.shutdown()
    task.cancel()


async def test_modbus_scaled_read_and_write(modbus_server):
    c = ModbusClient("127.0.0.1", modbus_server)
    await c.connect()
    try:
        amina, air = await c.read(["amina_flow", "column_01_air_flow"])
        assert amina.value == 557.4 and air.value == 249.2  # escala x10
        await c.write("amina_flow", 545.06)
        (amina,) = await c.read(["amina_flow"])
        assert amina.value == 545.1  # redondeo a la resolución del registro
        with pytest.raises(ValueError):
            await c.write("amina_flow", 7000.0)  # 70000 no cabe en 16 bits
    finally:
        await c.disconnect()


async def test_closed_loop_end_to_end_over_modbus(modbus_server, settings, audit):
    """Lectura Modbus -> router -> acción validada -> escritura Modbus."""
    c = ModbusClient("127.0.0.1", modbus_server)
    await c.connect()
    try:
        telemetry = await c.read(["amina_flow", "starch_flow"])
        req = InferenceRequest(asset_id="flot-col-01", mode="closed_loop", telemetry=telemetry)
        edge = FakeBackend("edge", default=action_json(
            action="ADJUST_SETPOINT", target_tag="amina_flow", new_value=540.0,
            reason_code="SILICA_HIGH"))
        resp = await InferenceRouter(settings, [edge], audit).infer(req)
        written, why = await FieldWriter(c, allow_writeback=True).apply(req, resp)
        assert written, why
        (amina,) = await c.read(["amina_flow"])
        assert amina.value == 540.0
    finally:
        await c.disconnect()


class FakeMqttClient:
    def __init__(self):
        self.published = []

    async def publish(self, topic, payload, qos=0):
        self.published.append((topic, json.loads(payload), qos))


async def test_mqtt_bridge_handles_valid_and_invalid(settings, audit):
    router = InferenceRouter(settings, [FakeBackend("edge", default=action_json())], audit)
    bridge = MqttBridge(router, FieldWriter(None, False), "unused")
    client = FakeMqttClient()
    good = InferenceRequest(asset_id="flot-col-01",
                            telemetry=[{"tag": "amina_flow", "value": 550}]).model_dump_json()
    await bridge.handle(client, "plant/flot-col-01/telemetry", good.encode())
    await bridge.handle(client, "plant/other/telemetry", good.encode())  # asset != tópico
    await bridge.handle(client, "plant/flot-col-01/telemetry", b"{not json")
    topics = [t for t, _, _ in client.published]
    assert topics == ["plant/flot-col-01/recommendation", "plant/other/rejected",
                      "plant/flot-col-01/rejected"]
    assert client.published[0][1]["action"]["action"] == "HOLD"
    assert client.published[0][1]["plc_written"] is False and client.published[0][2] == 1


def test_asset_from_topic():
    assert asset_from_topic("plant/col-1/telemetry") == "col-1"
    assert asset_from_topic("plant/col-1/recommendation") is None
    assert asset_from_topic("plant/a/b/telemetry") is None


async def test_redis_cache_and_job_queue(settings, audit, plant_request):
    redis = fakeredis.FakeAsyncRedis()
    cache = RedisCache(redis, ttl_s=30)
    await cache.set("k", "v")
    assert await cache.get("k") == "v" and await cache.get("missing") is None

    router = InferenceRouter(settings, [FakeBackend("edge", default=action_json())], audit, cache)
    q = RedisJobQueue(redis, router.infer, workers=2)
    await q.start()
    try:
        ids = [await q.submit(plant_request, "test") for _ in range(5)]
        for _ in range(100):
            states = [await q.get(i) for i in ids]
            if all(s.status == "done" for s in states):
                break
            await asyncio.sleep(0.02)
        assert all(s.status == "done" for s in states)
        assert isinstance(states[0].result, InferenceResponse)
        # Los 5 jobs comparten clave: 1 inferencia real, 4 desde caché en Redis.
        assert sum(s.result.cached for s in states) == 4
    finally:
        await q.stop()
