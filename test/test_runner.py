# -*- coding: utf-8 -*-
class TestRunner:
    def __init__(self, config, evaluator, metrics, telemetry=None):
        self.prompts = config.test_prompts
        self.evaluator = evaluator
        self.metrics = metrics
        self.telemetry = telemetry

    def run_tests(self):
        for i, prompt in enumerate(self.prompts):
            if self.telemetry:
                self.telemetry.log("Test", f"Prompt {i+1}: {prompt}")
            response = self.evaluator.generate_response(prompt)
            self.metrics.add(prompt, response)
        self.metrics.print_summary()
