# -*- coding: utf-8 -*-
from config import TestConfig
from model_loader import ModelLoader
from evaluator import Evaluator
from metrics import Metrics
from telemetry import Telemetry
from test_runner import TestRunner

cfg = TestConfig()
model_loader = ModelLoader(cfg)
evaluator = Evaluator(model_loader, cfg)
metrics = Metrics()
telemetry = Telemetry()

runner = TestRunner(cfg, evaluator, metrics, telemetry)
runner.run_tests()
