import sys, platform
import constants
import telemetry as tm
from llama_cpp import Llama
from telemetry import SocketTelemetrySender
from memory_monitor import MemoryMonitor

monitor = SocketTelemetrySender()
memory = MemoryMonitor()


def send_telemetry( message ):
    if monitor:
        monitor.send( message )
    else:
        print( message )
        
def banner(text, telemetry=None):
    send_telemetry({
        "type": tm.TYPE_BANNER,
        "data": {"message": text}
    })
    
def is_gpu_enabled(telemetry=None):

    try:
        llm = Llama(model_path=constants.get_model_path(constants.MODEL_SARCASM_CATEGORIZER_LLAMA), n_gpu_layers=-1, verbose=False)

        send_telemetry({
            "type": tm.TYPE_GPU,
            "data": { "tag" : tm.TYPE_GPU, "enabled": True }
        })
        return True
    except Exception as e:
        send_telemetry({
            "type": tm.TYPE_GPU,
            "data": { "tag" : tm.TYPE_GPU, "enabled": False, "error": str(e) }
        })
        return False

def check_memory():            
    if monitor:
        monitor.report_gpu_memory(tm.TYPE_DIAGNOSTICS)


def check_env():
    banner("[BOFH] Environment Check")
    check_memory()
        
    python_version = sys.version
    platform_info = platform.platform()
    gpu_supported = is_gpu_enabled()

    send_telemetry({
        "type": tm.TYPE_DIAGNOSTICS,
        "data": {
            "python_version": python_version,
            "platform": platform_info,
            "gpu_supported": gpu_supported
        }
    })

def scan_bytes(filepath, telemetry=None, targets=(0x8D, 0x8F)):
    with open(filepath, "rb") as f:
        content = f.read()

    matches = [(i, f"0x{b:02X}") for i, b in enumerate(content) if b in targets]

    if matches:
        message = f"[!] Found {len(matches)} matching bytes in {filepath}"
        send_telemetry( message )
        for index, hex_val in matches:
            send_telemetry(f"  → Byte {hex_val} at position {index}")

        send_telemetry({
            "type": "alert",
            "data": { "message": message }
        })
        send_telemetry({
            "type": "byte_scan",
            "data": {"filepath": filepath, "matches": matches}
        })
    else:
        send_telemetry(f"[✓] No target bytes found in {filepath}")
        send_telemetry({
            "type": "byte_scan",
            "data": {"filepath": filepath, "matches": []}
        })

def main():
            
    check_memory()
            
    check_env()
    # Optionally scan files:
    # scan_bytes("somefile.gguf", telemetry=monitor)
    check_memory()
        
if __name__ == "__main__":
    main()  # or pass telemetry from launcher