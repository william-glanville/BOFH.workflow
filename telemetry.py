import socket
import threading
import json
import time
import copy
import constants

from memory_monitor import MemoryMonitor
from abc import ABC, abstractmethod
from typing import Dict

TELEMETRY_MODE_NETWORK = "network"
TELEMETRY_MODE_CONSOLE = "console"

TELEMETRY_MODE = TELEMETRY_MODE_CONSOLE

TYPE_MEMORY = "memory"
TYPE_PROGRESS = "progress"
TYPE_TRAINING = "training"
TYPE_ALERT = "alert"
TYPE_ENVIRONMENT = "env"
TYPE_GPU = "gpu"
TYPE_BANNER = "banner"
TYPE_DIAGNOSTICS = "diagnostics"
TYPE_TELEMETRY = "telemetry"
TYPE_DISPLAY = "display"
TYPE_CONNECTION = "connection"

TAG_UNKNOWN = "unknown"
TAG_VALIDATION = "validation"
TAG_TRAINING = "training"
TAG_GRADNORM = "GradNorm"
TAG_LR = "LR"
TAG_LOSS = "Loss"
TAG_VALLOSS = "ValLoss"
TAG_STATUS = "status"

REC_DISPLAY = {
    "type": TYPE_DISPLAY,
    "data": {
        "tag" : TAG_UNKNOWN,
        "message" : TAG_UNKNOWN,
    }
 }
REC_MEMORY = {
    "type": TYPE_MEMORY,
    "data": {
        "tag":"unknown",
        "timestamp":0,
        "value":0.0
    }
}

REC_GPU_MEMORY = {
    "type": TYPE_MEMORY,
    "data": {
        "timestamp":0,
        "gpu_allocated": 0,
        "gpu_reserved": 0,
        "ram_used": 0
    }
}
REC_PROGRESS = {
    "type":TYPE_PROGRESS, 
    "data" : { 
        "tag":TAG_UNKNOWN, 
        "progress":0, 
        "total": 0 
    } 
}
REC_TRAINING  = {
    "type":TYPE_TRAINING,
    "data": {
        'tag': TAG_UNKNOWN, 
        'step': 0, 
        'value': 0
    }
}

REC_CONNECTION = {
    "type":TYPE_CONNECTION,
    "data": {
        'tag': TAG_STATUS, 
        'state': 'unknown',
        'error': 'none'
    }
}

HOST = "127.0.0.1"
PORT = 6111

def packet_builder( key, data ):
    return { "type" : key, "data" : data }
 
class TelemetryTCPServer(threading.Thread):
    def __init__(self, host=HOST, port=PORT, callback=None):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.callback = callback  # GUI processor
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.online = {
            "type":TYPE_CONNECTION,
            "data": {
                'tag': TAG_STATUS, 
                'state': 'online',
                'error': 'none'
            }
        }
        self.offline = {
            "type":TYPE_CONNECTION,
            "data": {
                'tag': TAG_STATUS, 
                'state': 'offline',
                'error': 'none'
            }
        }
        
    def run(self):        
        self.sock.bind((self.host, self.port))
        self.sock.listen(5)
        while True:
            conn, _ = self.sock.accept()
            threading.Thread(target=self.handle_client, args=(conn,), daemon=True).start()

    def handle_client(self, conn):
        try:
            buffer = ""
            if self.callback:
                self.callback(self.online)
                
            while True:
                data = conn.recv(4096)
                if not data:
                    break
                buffer += data.decode()
                while "\n" in buffer:
                   line, buffer = buffer.split("\n", 1)
                   try:
                       packet = json.loads(line)
                       if self.callback:
                           self.callback(packet)
                   except json.JSONDecodeError as e:
                       print(f"[Telemetry TCP Server Error] {e}")
                       pass
        finally:
            if self.callback:
                self.callback(self.offline)
            conn.close()
            
            
            
class TelemetryInterface(ABC):
    @abstractmethod
    def report_loss(self, step: int, loss: float) -> None: pass

    @abstractmethod
    def report_progress(self, tag: str, step: int, total_steps: int) -> None: pass

    @abstractmethod
    def report_gradnorm(self, step: int, norm: float) -> None: pass

    @abstractmethod
    def report_learningrate(self, step: int, lr: float) -> None: pass

    @abstractmethod
    def report_gpu_memory(self) -> None: pass

    @abstractmethod
    def display(self, source: str, message: str) -> None: pass

    @abstractmethod
    def display_dict(self, source: str, data: Dict) -> None: pass

    @abstractmethod
    def send(self, packet: dict) -> None: pass


class TelemetryProxy(TelemetryInterface):
    def __init__(self, mode: str = TELEMETRY_MODE, host: str = HOST, port: int = PORT):
        self.host = host
        self.port = port
        self.mode = mode
        self.sender: TelemetryInterface = self._resolve_sender(mode)

    def _resolve_sender(self, mode: str) -> TelemetryInterface:
        if mode == TELEMETRY_MODE_NETWORK:
            return SocketTelemetrySender(self.host, self.port)
        else:
            return ConsoleTelemetrySender()

    def set_mode(self, new_mode: str) -> None:
        if new_mode == self.mode:
            print(f"[TelemetryProxy] Already in mode: {new_mode}")
            return
        print(f"[TelemetryProxy] Switching to mode: {new_mode}")
        self.mode = new_mode
        self.sender = self._resolve_sender(new_mode)

    def send(self, packet: dict) -> None:
        self.sender.send( packet )
        
    def report_loss(self, step: int, loss: float) -> None:
        self.sender.report_loss(step, loss)

    def report_progress(self, tag: str, step: int, total_steps: int) -> None:
        self.sender.report_progress(tag, step, total_steps)

    def report_gradnorm(self, step: int, norm: float) -> None:
        self.sender.report_gradnorm(step, norm)

    def report_learningrate(self, step: int, lr: float) -> None:
        self.sender.report_learningrate(step, lr)

    def report_gpu_memory(self) -> None:
        self.sender.report_gpu_memory()

    def display(self, source: str, message: str) -> None:
        self.sender.display(source, message)

    def display_dict(self, source: str, data: Dict) -> None:
        self.sender.display_dict(source, data)
    

    
class SocketTelemetrySender(TelemetryInterface):
    def __init__(self, host=HOST, port=PORT):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))        
        self.memory = MemoryMonitor()

    def send(self, packet: dict) -> None:
        try:
            msg = json.dumps(packet).encode() + b"\n"
            self.sock.sendall(msg)
        except Exception as e:
            print(f"[SocketTelemetrySender Error] {e}")

    def report_gpu_memory(self) -> None:
        gpua, gpur, ram = self.memory.snapshot()
        for tag, value in {
            constants.SERIES_GPUALLOCATED: gpua,
            constants.SERIES_GPURESERVED: gpur,
            constants.SERIES_RAMUSED: ram
        }.items():
            packet = copy.deepcopy(REC_MEMORY)
            packet["data"].update({
                "tag": tag,
                "timestamp": time.time(),
                "value": value
            })
            self.send(packet)

    def report_progress(self, tag: str, step: int, total_steps: int) -> None:
        packet = copy.deepcopy(REC_PROGRESS)
        packet["data"].update({
            "tag": tag,
            "progress": step,
            "total": total_steps
        })
        self.send(packet)

    def report_gradnorm(self, step: int, norm: float) -> None:
        self._send_training(TAG_GRADNORM, step, norm)

    def report_loss(self, step: int, loss: float) -> None:
        self._send_training(TAG_LOSS, step, loss)

    def report_learningrate(self, step: int, lr: float) -> None:
        self._send_training(TAG_LR, step, lr)

    def _send_training(self, tag: str, step: int, value: float) -> None:
        packet = copy.deepcopy(REC_TRAINING)
        packet["data"].update({
            "tag": tag,
            "step": step,
            "value": value
        })
        self.send(packet)

    def display(self, source: str, message: str) -> None:
        packet = copy.deepcopy(REC_DISPLAY)
        packet["data"].update({"tag": source, "message": message})
        self.send(packet)

    def display_dict(self, source: str, data: dict) -> None:
        packet = copy.deepcopy(REC_DISPLAY)
        packet["data"].update({"tag": source, **data})
        self.send(packet)
        
        
class ConsoleTelemetrySender(TelemetryInterface):
    def __init__(self):
        pass  # No connection setup needed

    def send(self,packet:dict) -> None:
        print(f"Raw Packet : {packet}")
        
    def report_loss(self, step: int, loss: float) -> None:
        print(f"[Telemetry] Step {step} | Loss: {loss:.4f}")

    def report_progress(self, tag: str, step: int, total_steps: int) -> None:
        percent = (step / total_steps) * 100
        print(f"[Progress] Step {step}/{total_steps} ({percent:.2f}%)")

    def report_gradnorm(self, step: int, norm: float) -> None:
        print(f"[GradNorm] Step {step} | Norm: {norm:.4f}")

    def report_learningrate(self, step: int, lr: float) -> None:
        print(f"[LearningRate] Step {step} | LR: {lr:.6f}")

    def report_gpu_memory(self) -> None:
        import torch
        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated() / 1024**2
            max_mem = torch.cuda.max_memory_allocated() / 1024**2
            print(f"[Console] GPU Memory: {mem:.2f}MB / Max: {max_mem:.2f}MB")
        else:
            print("[Console] GPU not available.")

    def display(self, source: str, message: str) -> None:
        print(f"[{source}] {message}")

    def display_dict(self, source: str, data: dict) -> None:
        print(f"[{source}] Dictionary Report:")
        for key, value in data.items():
            print(f"  {key}: {value}")

    def __call__(self):
        print("[Telemetry] __call__ triggered — likely heartbeat or silent ping.")        