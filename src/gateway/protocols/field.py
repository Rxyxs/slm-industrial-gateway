"""Conectores de campo: lectura de tags y escritura de setpoints vía OPC UA y Modbus TCP.

La escritura es la única operación del gateway con efecto físico, así que pasa
por `FieldWriter.apply()`, que vuelve a exigir todas las condiciones aunque el
llamador ya las haya revisado:
- modo `closed_loop` pedido explícitamente por el cliente,
- `ALLOW_PLC_WRITEBACK=true` en la configuración del gateway,
- acción ADJUST_SETPOINT validada (sin rechazo, no failsafe ni caché),
- tag escribible con dirección configurada en el protocolo elegido.
Si falta cualquiera, no se escribe y se devuelve el motivo.
"""

from __future__ import annotations

import logging
from typing import Protocol

from src.gateway.schemas import ActionType, InferenceRequest, InferenceResponse, TelemetrySample
from src.gateway.tags import TAG_REGISTRY

log = logging.getLogger("slm_gateway.field")


class FieldClient(Protocol):
    async def read(self, tags: list[str]) -> list[TelemetrySample]: ...

    async def write(self, tag: str, value: float) -> None: ...


class OpcUaClient:
    def __init__(self, url: str, security_string: str | None = None):
        self.url = url
        self.security_string = security_string
        self._client = None

    async def connect(self) -> None:
        from asyncua import Client

        self._client = Client(url=self.url, timeout=2)
        if self.security_string:
            # p. ej. "Basic256Sha256,SignAndEncrypt,cert.der,key.pem"
            await self._client.set_security_string(self.security_string)
        await self._client.connect()

    async def disconnect(self) -> None:
        if self._client is not None:
            await self._client.disconnect()

    async def read(self, tags: list[str]) -> list[TelemetrySample]:
        out = []
        for tag in tags:
            node = self._client.get_node(TAG_REGISTRY[tag].opcua_node)
            dv = await node.read_data_value()
            quality = "good" if (dv.StatusCode is None or dv.StatusCode.is_good()) else "bad"
            out.append(TelemetrySample(tag=tag, value=float(dv.Value.Value), quality=quality))
        return out

    async def write(self, tag: str, value: float) -> None:
        from asyncua import ua

        node = self._client.get_node(TAG_REGISTRY[tag].opcua_node)
        await node.write_value(ua.DataValue(ua.Variant(float(value), ua.VariantType.Double)))


class ModbusClient:
    """Holding registers de 16 bits, convención 4xxxx -> offset (registro - 40001).
    El valor de ingeniería se multiplica por `modbus_scale` y se redondea."""

    def __init__(self, host: str, port: int = 502, device_id: int = 1):
        self.host, self.port, self.device_id = host, port, device_id
        self._client = None

    async def connect(self) -> None:
        from pymodbus.client import AsyncModbusTcpClient

        self._client = AsyncModbusTcpClient(self.host, port=self.port, timeout=2)
        if not await self._client.connect():
            raise ConnectionError(f"modbus {self.host}:{self.port} unreachable")

    async def disconnect(self) -> None:
        if self._client is not None:
            self._client.close()

    @staticmethod
    def _offset(tag: str) -> int:
        reg = TAG_REGISTRY[tag].modbus_register
        if reg is None:
            raise KeyError(f"{tag} has no modbus register")
        return reg - 40001

    async def read(self, tags: list[str]) -> list[TelemetrySample]:
        out = []
        for tag in tags:
            rr = await self._client.read_holding_registers(
                self._offset(tag), count=1, device_id=self.device_id
            )
            if rr.isError():
                out.append(TelemetrySample(tag=tag, value=0.0, quality="bad"))
                continue
            out.append(TelemetrySample(tag=tag, value=rr.registers[0] / TAG_REGISTRY[tag].modbus_scale))
        return out

    async def write(self, tag: str, value: float) -> None:
        raw = round(value * TAG_REGISTRY[tag].modbus_scale)
        if not 0 <= raw <= 0xFFFF:
            raise ValueError(f"{tag}={value} does not fit a 16-bit register at scale "
                             f"{TAG_REGISTRY[tag].modbus_scale}")
        wr = await self._client.write_register(self._offset(tag), raw, device_id=self.device_id)
        if wr.isError():
            raise IOError(f"modbus write {tag} failed: {wr}")


class FieldWriter:
    def __init__(self, client: FieldClient | None, allow_writeback: bool):
        self.client = client
        self.allow_writeback = allow_writeback

    async def apply(self, req: InferenceRequest, resp: InferenceResponse) -> tuple[bool, str]:
        a = resp.action
        if req.mode != "closed_loop":
            return False, "advisory mode"
        if not self.allow_writeback:
            return False, "ALLOW_PLC_WRITEBACK disabled"
        if self.client is None:
            return False, "no field client configured"
        if a.action != ActionType.ADJUST_SETPOINT.value:
            return False, f"action {a.action} has nothing to write"
        if resp.rejected_reason or resp.tier in {"failsafe", "cache"}:
            return False, f"not writable from tier={resp.tier} rejected={resp.rejected_reason!r}"
        if not TAG_REGISTRY[a.target_tag].writable:
            return False, f"{a.target_tag} is read-only"
        await self.client.write(a.target_tag, float(a.new_value))
        log.info("wrote %s=%s for request %s", a.target_tag, a.new_value, resp.request_id)
        return True, "written"
