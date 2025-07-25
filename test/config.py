class TestConfig:
    model_path = "adapters/"
    test_prompts = [
        "Describe sarcasm in natural language.",
        "Give an example of neutral tone.",
        "Explain why humor detection is challenging.",
    ]
    max_new_tokens = 100
    temperature = 0.7
    top_p = 0.9
    repetition_penalty = 1.2
    pad_token_id = -100
