import sys
import platform

import constants
import telemetry as tm
from llama_cpp import Llama
from telemetry import SocketTelemetrySender#, ConsoleTelemetrySender

class Diagnostics:
    def __init__(self, telemetry_sender=None):
        # Use provided sender or default to a socket-based one
        self.monitor = telemetry_sender or SocketTelemetrySender()
        #self.monitor = telemetry_sender or ConnectionAbortedError()
        
    def send_telemetry(self, message):
        if self.monitor:
            self.monitor.send(message)
        else:
            print(message)

    def banner(self, text):
        self.send_telemetry({
            "type": tm.TYPE_BANNER,
            "data": {"message": text}
        })

    def is_gpu_enabled(self):
        """
        Attempts to instantiate the Llama model with GPU layers enabled.
        Reports success or captures the exception message.
        """
        try:
            Llama(
                model_path=constants.get_model_path(
                    constants.MODEL_SARCASM_CATEGORIZER_LLAMA
                ),
                n_gpu_layers=-1,
                verbose=False
            )
            self.send_telemetry({
                "type": tm.TYPE_GPU,
                "data": {"tag": tm.TYPE_GPU, "enabled": True}
            })
            return True

        except Exception as e:
            self.send_telemetry({
                "type": tm.TYPE_GPU,
                "data": {
                    "tag": tm.TYPE_GPU,
                    "enabled": False,
                    "error": str(e)
                }
            })
            return False

    def check_memory(self):
        # Report GPU memory usage if telemetry sender exists
        if self.monitor:
            self.monitor.report_gpu_memory()

    def check_env(self):
        """
        Emits a banner, checks memory, Python/platform info,
        GPU support, and sends a diagnostics payload.
        """
        self.banner("[BOFH] Environment Check")
        self.check_memory()

        python_version = sys.version
        platform_info = platform.platform()
        gpu_supported = self.is_gpu_enabled()

        self.send_telemetry({
            "type": tm.TYPE_DIAGNOSTICS,
            "data": {
                "python_version": python_version,
                "platform": platform_info,
                "gpu_supported": gpu_supported
            }
        })

    def scan_bytes(self, filepath, targets=(0x8D, 0x8F)):
        """
        Opens a file in binary mode and reports any bytes matching
        the `targets` tuple.
        """
        matches = []
        with open(filepath, "rb") as f:
            content = f.read()
            matches = [
                (i, f"0x{b:02X}")
                for i, b in enumerate(content)
                if b in targets
            ]

        if matches:
            msg = f"[!] Found {len(matches)} matching bytes in {filepath}"
            self.send_telemetry(msg)
            for idx, hex_val in matches:
                self.send_telemetry(f"  → Byte {hex_val} at position {idx}")

            self.send_telemetry({
                "type": "alert",
                "data": {"message": msg}
            })
            self.send_telemetry({
                "type": "byte_scan",
                "data": {"filepath": filepath, "matches": matches}
            })

        else:
            msg = f"[✓] No target bytes found in {filepath}"
            self.send_telemetry(msg)
            self.send_telemetry({
                "type": "byte_scan",
                "data": {"filepath": filepath, "matches": []}
            })

    def run(self):
        """
        Executes the standard diagnostics sequence:
        1. Pre-env memory check
        2. Environment check
        3. (Optional) Byte scans
        4. Post-env memory check
        """
        try:
            self.monitor.report_progress("Diagnostics", 0, 3)
            self.check_memory()
            self.monitor.report_progress("Diagnostics", 1, 3)
            self.check_env()
            self.monitor.report_progress("Diagnostics", 3, 3)
            # Example: self.scan_bytes("somefile.gguf")
            self.check_memory()
            self.monitor.report_progress("Diagnostics", 3, 3)
        except Exception as e:
            print(f"EXCeption {e}")

def main():
    diagnostics = Diagnostics()
    diagnostics.run()


if __name__ == "__main__":
    main()