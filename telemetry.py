import socket
import threading
import json

from memory_monitor import MemoryMonitor


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
REC_GPU_MEMORY = {
    "type": TYPE_MEMORY,
    "data": {
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
        self.online = REC_CONNECTION.copy()
        self.online["data"]["state"] = "online"
        self.offline = REC_CONNECTION.copy()
        self.offline["data"]["state"] = "offline"
        
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
            

class SocketTelemetrySender:
    def __init__(self, host=HOST, port=PORT):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))        
        self.memory = MemoryMonitor()
        
    def send(self, packet: dict):
        try:
            msg = json.dumps(packet).encode() + b"\n"
            self.sock.sendall(msg)
        except Exception as e:
            print(f"[Telemetry Sender Failure] {e}")            
            
    
    def report_gpu_memory( self, tag ):
        gpua, gpur, ram = self.memory.snapshot()
        candidate = REC_GPU_MEMORY.copy()
        candidate["data"]["tag"] = tag
        candidate["data"]["gpu_allocated"] = gpua
        candidate["data"]["gpu_reserved"] = gpur
        candidate["data"]["ram_used"] = ram
        self.send(candidate)
                
    def report_progress(self,tag,current,total):
        candidate = REC_PROGRESS.copy()
        candidate["data"]["tag"] = tag
        candidate["data"]["progress"] = current
        candidate["data"]["total"] = total
        self.send(candidate)

    def report_gradnorm(self, step, norm ):
        candidate = REC_TRAINING.copy()
        candidate["data"]["tag"] = TAG_GRADNORM
        candidate["data"]["step"] = step
        candidate["data"]["value"] = norm
        self.send(candidate)
        
    def report_loss(self, step, loss ):
        candidate = REC_TRAINING.copy()
        candidate["data"]["tag"] = TAG_LOSS
        candidate["data"]["step"] = step
        candidate["data"]["value"] = loss
        self.send(candidate)
        
    def report_learningrate(self, step, rate ):
        candidate = REC_TRAINING.copy()
        candidate["data"]["tag"] = TAG_LR
        candidate["data"]["step"] = step
        candidate["data"]["value"] = rate
        self.send(candidate)
        
    def display( self, tag, message ):
        candidate = {
            "type": TYPE_DISPLAY,
            "data": {
                "tag" : tag,
                "message" : message,
            }
         }
        self.send(candidate)
        
    def display_dict( self, tag, message:dict ):
        candidate = {
            "type": TYPE_DISPLAY,
            "data": {
                "tag" : tag
            }
         }
        candidate["data"].update(message)
        self.send(candidate)        
        
        
class ConsoleTelemetrySender:
    def __init__(self):
        pass  # No connection setup needed

    def display(self, source, message):
        print(f"[{source}] {message}")

    def display_dict(self, source, data: dict):
        print(f"[{source}] Dictionary Report:")
        for key, value in data.items():
            print(f"  {key}: {value}")

    def report_loss(self, step, loss):
        print(f"[Telemetry] Step {step} | Loss: {loss:.4f}")

    def report_progress(self, step, total_steps):
        percent = (step / total_steps) * 100
        print(f"[Progress] Step {step}/{total_steps} ({percent:.2f}%)")

    def report_gradnorm(self, step, norm):
        print(f"[GradNorm] Step {step} | Norm: {norm:.4f}")

    def report_learningrate(self, step, lr):
        print(f"[LearningRate] Step {step} | LR: {lr:.6f}")

    def report_gpu_memory(self, label="Telemetry"):
        import torch
        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated() / 1024**2
            max_mem = torch.cuda.max_memory_allocated() / 1024**2
            print(f"[{label}] GPU Memory: {mem:.2f}MB / Max: {max_mem:.2f}MB")
        else:
            print(f"[{label}] GPU not available.")

    def __call__(self):
        print("[Telemetry] __call__ triggered — likely heartbeat or silent ping.")        