"""
Central registry of *display-name → metadata* for every humanizer
model we want to expose in the UI / CLI.

* provider   – "openai", "gemini", or "claude"
* model      – the actual model-id used in the API call
* prompt_id  – "default" (generic LLM) or "finetuned" (our tuned models)
"""

MODEL_REGISTRY = {
    # ---- vanilla OpenAI ------------------------------------------------
    "gpt-4.1":            {"provider": "openai", "model": "gpt-4.1",      "prompt_id": "default"},
    "gpt-4.1-mini":       {"provider": "openai", "model": "gpt-4.1-mini", "prompt_id": "default"},
    "gpt-4o":             {"provider": "openai", "model": "gpt-4o",       "prompt_id": "default"},

    # ---- Claude --------------------------------------------------------
    "claude-sonnet-4":       {"provider": "claude", "model": "claude-sonnet-4-20250514",    "prompt_id": "default"},
    "claude-sonnet-3.7":     {"provider": "claude", "model": "claude-3-7-sonnet-latest",   "prompt_id": "default"},
    "claude-haiku-3.5":      {"provider": "claude", "model": "claude-3-5-haiku-latest",    "prompt_id": "default"},

    # ---- Gemini --------------------------------------------------------
    "gemini-2.0-flash":        {"provider": "gemini", "model": "gemini-2.0-flash",              "prompt_id": "default"},
    "gemini-2.0-flash-lite":   {"provider": "gemini", "model": "gemini-2.0-flash-lite",         "prompt_id": "default"},
    "gemini-2.5-flash":        {"provider": "gemini", "model": "gemini-2.5-flash-preview-05-20", "prompt_id": "default"},
    "gemini-2.5-pro":        {"provider": "gemini", "model": "gemini-2.5-pro", "prompt_id": "default"},

    # ---- our fine-tunes -----------------------------------------------
    "gpt-4o-old-model":          {"provider": "openai_ft", "model": "ft:gpt-4o-mini-2024-07-18:litero-ai:v4-short-simple:9oaYlNl2",                              "prompt_id": "finetuned"},

    # differentiated codenames for fine‑tunes created on 2024‑08‑06 and 2025‑04‑14
    "gpt-4o-hum30raw":           {"provider": "openai_ft", "model": "ft:gpt-4o-2024-08-06:litero-ai:hum30raw:BcCFkyvO",                                         "prompt_id": "finetuned"},
    "gpt-4o-hum40naive":         {"provider": "openai_ft", "model": "ft:gpt-4o-2024-08-06:litero-ai:hum40-naive-auto:Bi0rO31o",                                    "prompt_id": "finetuned"},

    "gpt-4.1-hum30start":        {"provider": "openai_ft", "model": "ft:gpt-4.1-2025-04-14:litero-ai:hum30start:BcBvzILe",                                         "prompt_id": "finetuned"},
    "gpt-4.1-hum40naive":        {"provider": "openai_ft", "model": "ft:gpt-4.1-2025-04-14:litero-ai:hum40-naive-auto:Bi0wCXgi",                                    "prompt_id": "finetuned"},

    "gpt-4.1-mini-hum40naive":   {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:hum40-naive-auto:Bi0qnqGa",                               "prompt_id": "finetuned"},
}