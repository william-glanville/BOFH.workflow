import torch
import psutil
import os

class MemoryMonitor:
    def __init__(self):
        self.pid = os.getpid()
        self.process = psutil.Process(self.pid)

    def snapshot(self):
        gpu_allocated = torch.cuda.memory_allocated() / 1e9
        gpu_reserved = torch.cuda.memory_reserved() / 1e9
        ram_used = self.process.memory_info().rss / 1e9

        return gpu_allocated,gpu_reserved,ram_used
