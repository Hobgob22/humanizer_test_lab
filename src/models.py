"""
Central registry of *display-name → metadata* for every humanizer
model we want to expose in the UI / CLI.

* provider          – "openai", "gemini", "claude", "openai_ft", "openai_dynamic"
* model             – the actual model-id used in the API call
* prompt_id         – "default" (generic LLM), "finetuned" (our tuned models), "legacy-finetuned", or "dynamic"
* system_prompt     – (dynamic models only) which system prompt to use
* scores_in_prompt  – (dynamic models only) how to include scores in user prompt
* base_model        – (for grouping in UI)
"""

MODEL_REGISTRY = {
    # ---- vanilla OpenAI ------------------------------------------------
    "gpt-4.1":            {"provider": "openai", "model": "gpt-4.1",      "prompt_id": "default", "base_model": "gpt-4.1"},
    "gpt-4.1-mini":       {"provider": "openai", "model": "gpt-4.1-mini", "prompt_id": "default", "base_model": "gpt-4.1-mini"},
    "gpt-4o":             {"provider": "openai", "model": "gpt-4o",       "prompt_id": "default", "base_model": "gpt-4o"},

    # ---- Claude --------------------------------------------------------
    "claude-sonnet-4":       {"provider": "claude", "model": "claude-sonnet-4-20250514",    "prompt_id": "default", "base_model": "claude"},
    "claude-sonnet-3.7":     {"provider": "claude", "model": "claude-3-7-sonnet-latest",   "prompt_id": "default", "base_model": "claude"},
    "claude-haiku-3.5":      {"provider": "claude", "model": "claude-3-5-haiku-latest",    "prompt_id": "default", "base_model": "claude"},

    # ---- Gemini --------------------------------------------------------
    "gemini-2.0-flash":        {"provider": "gemini", "model": "gemini-2.0-flash",              "prompt_id": "default", "base_model": "gemini"},
    "gemini-2.0-flash-lite":   {"provider": "gemini", "model": "gemini-2.0-flash-lite",         "prompt_id": "default", "base_model": "gemini"},
    "gemini-2.5-flash":        {"provider": "gemini", "model": "gemini-2.5-flash-preview-05-20", "prompt_id": "default", "base_model": "gemini"},
    "gemini-2.5-pro":        {"provider": "gemini", "model": "gemini-2.5-pro", "prompt_id": "default", "base_model": "gemini"},

    # ---- our fine-tunes (LEGACY) -----------------------------------------------
    "gpt-4o-old-model":          {"provider": "openai_ft", "model": "ft:gpt-4o-mini-2024-07-18:litero-ai:v4-short-simple:9oaYlNl2",                              "prompt_id": "legacy-finetuned", "base_model": "gpt-4o-mini"},

    # differentiated codenames for fine-tunes created on 2024-08-06 and 2025-04-14
    "gpt-4o-hum30raw":           {"provider": "openai_ft", "model": "ft:gpt-4o-2024-08-06:litero-ai:hum30raw:BcCFkyvO",                                         "prompt_id": "finetuned", "base_model": "gpt-4o"},
    "gpt-4o-hum40naive":         {"provider": "openai_ft", "model": "ft:gpt-4o-2024-08-06:litero-ai:hum40-naive-auto:Bi0rO31o",                                    "prompt_id": "finetuned", "base_model": "gpt-4o"},

    "gpt-4.1-hum30start":        {"provider": "openai_ft", "model": "ft:gpt-4.1-2025-04-14:litero-ai:hum30start:BcBvzILe",                                         "prompt_id": "finetuned", "base_model": "gpt-4.1"},
    "gpt-4.1-hum40naive":        {"provider": "openai_ft", "model": "ft:gpt-4.1-2025-04-14:litero-ai:hum40-naive-auto:Bi0wCXgi",                                    "prompt_id": "finetuned", "base_model": "gpt-4.1"},

    "gpt-4.1-mini-hum40naive":   {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:hum40-naive-auto:Bi0qnqGa",                               "prompt_id": "finetuned", "base_model": "gpt-4.1-mini"},

    # NEW fine-tunes (2024-08-06 and 2025-04-14) ----------
    "gpt-4.1-hum30raw-fix1":     {"provider": "openai_ft", "model": "ft:gpt-4.1-2025-04-14:litero-ai:hum30raw-fix1:Bm3eO9ge",                                      "prompt_id": "finetuned", "base_model": "gpt-4.1"},
    "gpt-4o-hum30raw-fix1":      {"provider": "openai_ft", "model": "ft:gpt-4o-2024-08-06:litero-ai:hum30raw-fix1:Bm3b7N9S",                                       "prompt_id": "finetuned", "base_model": "gpt-4o"},

    "gpt-4o-hum30raw-fix2":      {"provider": "openai_ft", "model": "ft:gpt-4o-2024-08-06:litero-ai:hum-30raw-fix2:BmIOvuFc",                                      "prompt_id": "finetuned", "base_model": "gpt-4o"},
    "gpt-4.1-hum30raw-fix2":     {"provider": "openai_ft", "model": "ft:gpt-4.1-2025-04-14:litero-ai:hum-30raw-fix2:BmIQK91Y",                                      "prompt_id": "finetuned", "base_model": "gpt-4.1"},

    "gpt-4o-hum30raw-retrain1":     {"provider": "openai_ft", "model": "ft:gpt-4o-2024-08-06:litero-ai:hum30raw-retrain1:BmIs6zjz",                                      "prompt_id": "finetuned", "base_model": "gpt-4o"},
    "gpt-4o-mini-hum30raw-fix1":     {"provider": "openai_ft", "model": "ft:gpt-4o-mini-2024-07-18:litero-ai:hum30raw-fix1:BmIiNRen",                                      "prompt_id": "finetuned", "base_model": "gpt-4o-mini"},

    # ---- NEW DYNAMIC MODELS (from new_models.csv) ----------------------
    # gpt-4o-mini based models
    "min-e3-b8-m10-v10":        {"provider": "openai_ft", "model": "ft:gpt-4o-mini-2024-07-18:litero-ai:min-e3-b8-m10-v10:CahO6KtY", "prompt_id": "dynamic", "system_prompt": "minimal_prompt", "scores_in_prompt": "both_raw", "base_model": "gpt-4o-mini"},
    "raw-e5-b16-m08-v10":       {"provider": "openai_ft", "model": "ft:gpt-4o-mini-2024-07-18:litero-ai:raw-e5-b16-m08-v10:CaRGQPzO", "prompt_id": "dynamic", "system_prompt": "none", "scores_in_prompt": "none", "base_model": "gpt-4o-mini"},
    "cmp-e4-b24-m12-v15":       {"provider": "openai_ft", "model": "ft:gpt-4o-mini-2024-07-18:litero-ai:cmp-e4-b24-m12-v15:CahcnBqU", "prompt_id": "dynamic", "system_prompt": "compact_prompt", "scores_in_prompt": "both_binned", "base_model": "gpt-4o-mini"},
    "min-e6-b16-m05-v20":       {"provider": "openai_ft", "model": "ft:gpt-4o-mini-2024-07-18:litero-ai:min-e6-b16-m05-v20:CahShBED", "prompt_id": "dynamic", "system_prompt": "minimal_prompt", "scores_in_prompt": "ai_score_raw", "base_model": "gpt-4o-mini"},
    "rubx-e8-b32-m03-v15":      {"provider": "openai_dynamic", "model": "gpt-4o-mini", "prompt_id": "dynamic", "system_prompt": "rich_prompt_with_counter_examples", "scores_in_prompt": "both_raw", "base_model": "gpt-4o-mini"},
    "raw-e3-b8-m15-v10-mini":   {"provider": "openai_ft", "model": "ft:gpt-4o-mini-2024-07-18:litero-ai:raw-e3-b8-m15-v10:CaRj65OZ", "prompt_id": "dynamic", "system_prompt": "none", "scores_in_prompt": "ai_score_binned", "base_model": "gpt-4o-mini"},

    # gpt-4.1-mini based models
    "min-e3-b8-m10-v10-d2":     {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:min-e3-b8-m10-v10-d2:CahDAVzH", "prompt_id": "dynamic", "system_prompt": "minimal_prompt", "scores_in_prompt": "both_raw", "base_model": "gpt-4.1-mini"},
    "cmp-e5-b16-m08-v15":       {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:cmp-e5-b16-m08-v15:CaheuKck", "prompt_id": "dynamic", "system_prompt": "compact_prompt", "scores_in_prompt": "both_binned", "base_model": "gpt-4.1-mini"},
    "raw-e4-b24-m06-v20":       {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:raw-e4-b24-m06-v20:CaRJcZNh", "prompt_id": "dynamic", "system_prompt": "none", "scores_in_prompt": "none", "base_model": "gpt-4.1-mini"},
    "rdesc-e6-b20-m10-v10":     {"provider": "openai_dynamic", "model": "gpt-4.1-mini", "prompt_id": "dynamic", "system_prompt": "rich_prompt_with_negative_examples", "scores_in_prompt": "ai_score_raw", "base_model": "gpt-4.1-mini"},
    "min-e8-b32-m04-v20":       {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:min-e8-b32-m04-v20:CahGHp0S", "prompt_id": "dynamic", "system_prompt": "minimal_prompt", "scores_in_prompt": "both_raw", "base_model": "gpt-4.1-mini"},
    "cmp-e3-b12-m12-v12":       {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:cmp-e3-b12-m12-v12:Cahiin3j", "prompt_id": "dynamic", "system_prompt": "compact_prompt", "scores_in_prompt": "ai_score_binned", "base_model": "gpt-4.1-mini"},

    # gpt-4.1 based models
    "min-e3-b6-m08-v15":        {"provider": "openai_ft", "model": "ft:gpt-4.1-2025-04-14:litero-ai:min-e3-b6-m08-v15:CahTVFjJ", "prompt_id": "dynamic", "system_prompt": "minimal_prompt", "scores_in_prompt": "both_raw", "base_model": "gpt-4.1"},
    "raw-e4-b8-m06-v10":        {"provider": "openai_ft", "model": "ft:gpt-4.1-2025-04-14:litero-ai:raw-e4-b8-m06-v10:CaRhZ8la", "prompt_id": "dynamic", "system_prompt": "none", "scores_in_prompt": "none", "base_model": "gpt-4.1"},
    "cmp-e5-b8-m05-v20":        {"provider": "openai_ft", "model": "ft:gpt-4.1-2025-04-14:litero-ai:cmp-e5-b8-m05-v20:Cai16L5a", "prompt_id": "dynamic", "system_prompt": "compact_prompt", "scores_in_prompt": "both_binned", "base_model": "gpt-4.1"},
    "rchi-e6-b12-m04-v15":      {"provider": "openai_dynamic", "model": "gpt-4.1", "prompt_id": "dynamic", "system_prompt": "rich_prompt_with_focus_areas", "scores_in_prompt": "both_raw", "base_model": "gpt-4.1"},
    "min-e8-b16-m03-v20":       {"provider": "openai_ft", "model": "ft:gpt-4.1-2025-04-14:litero-ai:min-e8-b16-m03-v20:Caha7LoY", "prompt_id": "dynamic", "system_prompt": "minimal_prompt", "scores_in_prompt": "both_raw", "base_model": "gpt-4.1"},
    "raw-e3-b10-m10-v10":       {"provider": "openai_ft", "model": "ft:gpt-4.1-2025-04-14:litero-ai:raw-e3-b10-m10-v10:CaSOMjS9", "prompt_id": "dynamic", "system_prompt": "none", "scores_in_prompt": "ai_score_raw", "base_model": "gpt-4.1"},

    # gpt-4.1-nano based models
    "min-e3-b32-m10-v10-nano":  {"provider": "openai_ft", "model": "ft:gpt-4.1-nano-2025-04-14:litero-ai:min-e3-b32-m10-v10:CaicZpmo", "prompt_id": "dynamic", "system_prompt": "minimal_prompt", "scores_in_prompt": "both_binned", "base_model": "gpt-4.1-nano"},
    "raw-e5-b48-m08-v15":       {"provider": "openai_ft", "model": "ft:gpt-4.1-nano-2025-04-14:litero-ai:raw-e5-b48-m08-v15:CaRKgWab", "prompt_id": "dynamic", "system_prompt": "none", "scores_in_prompt": "none", "base_model": "gpt-4.1-nano"},
    "cmp-e6-b64-m06-v20":       {"provider": "openai_dynamic", "model": "gpt-4.1-nano", "prompt_id": "dynamic", "system_prompt": "compact_prompt", "scores_in_prompt": "both_raw", "base_model": "gpt-4.1-nano"},
    "min-e4-b40-m12-v12":       {"provider": "openai_ft", "model": "ft:gpt-4.1-nano-2025-04-14:litero-ai:min-e4-b40-m12-v12:CahPuZKW", "prompt_id": "dynamic", "system_prompt": "minimal_prompt", "scores_in_prompt": "ai_score_binned", "base_model": "gpt-4.1-nano"},
    "rich-e8-b64-m04-v20":      {"provider": "openai_dynamic", "model": "gpt-4.1-nano", "prompt_id": "dynamic", "system_prompt": "rich_prompt_standard", "scores_in_prompt": "both_raw", "base_model": "gpt-4.1-nano"},
    "raw-e3-b24-m15-v10-nano":  {"provider": "openai_ft", "model": "ft:gpt-4.1-nano-2025-04-14:litero-ai:raw-e3-b24-m15-v10:CaRJtSWZ", "prompt_id": "dynamic", "system_prompt": "none", "scores_in_prompt": "none", "base_model": "gpt-4.1-nano"},

    # gpt-4o based models
    "min-e3-b4-m06-v15":        {"provider": "openai_ft", "model": "ft:gpt-4o-2024-08-06:litero-ai:min-e3-b4-m06-v15:Cahds3Pm", "prompt_id": "dynamic", "system_prompt": "minimal_prompt", "scores_in_prompt": "both_raw", "base_model": "gpt-4o"},
    "raw-e4-b6-m05-v10":        {"provider": "openai_ft", "model": "ft:gpt-4o-2024-08-06:litero-ai:raw-e4-b6-m05-v10:CaRkZH9u", "prompt_id": "dynamic", "system_prompt": "none", "scores_in_prompt": "none", "base_model": "gpt-4o"},
    "cmp-e5-b6-m04-v20":        {"provider": "openai_ft", "model": "ft:gpt-4o-2024-08-06:litero-ai:cmp-e5-b6-m04-v20:CaiCmOVf", "prompt_id": "dynamic", "system_prompt": "compact_prompt", "scores_in_prompt": "both_binned", "base_model": "gpt-4o"},
    "rich-e6-b8-m03-v20":       {"provider": "openai_dynamic", "model": "gpt-4o", "prompt_id": "dynamic", "system_prompt": "rich_prompt_standard", "scores_in_prompt": "both_raw", "base_model": "gpt-4o"},
    "min-e3-b5-m10-v10-4o":     {"provider": "openai_ft", "model": "ft:gpt-4o-2024-08-06:litero-ai:min-e3-b5-m10-v10:CahmwAHN", "prompt_id": "dynamic", "system_prompt": "minimal_prompt", "scores_in_prompt": "ai_score_raw", "base_model": "gpt-4o"},
    "raw-e8-b8-m025-v20":       {"provider": "openai_ft", "model": "ft:gpt-4o-2024-08-06:litero-ai:raw-e8-b8-m025-v20:CaS2nSDe", "prompt_id": "dynamic", "system_prompt": "none", "scores_in_prompt": "none", "base_model": "gpt-4o"},

}