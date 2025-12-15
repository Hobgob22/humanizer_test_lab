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
    "kimi-k2":            {"provider": "groq",   "model": "moonshotai/kimi-k2-instruct-0905", "prompt_id": "default", "base_model": "kimi-k2"},
    "gpt-5":              {"provider": "openai", "model": "gpt-5",        "prompt_id": "default", "base_model": "gpt-5"},
    "gpt-5-mini":         {"provider": "openai", "model": "gpt-5-mini",   "prompt_id": "default", "base_model": "gpt-5-mini"},
    "gpt-5-nano":         {"provider": "openai", "model": "gpt-5-nano",   "prompt_id": "default", "base_model": "gpt-5-nano"},
    "gpt-5.1":            {"provider": "openai", "model": "gpt-5.1",      "prompt_id": "default", "base_model": "gpt-5.1"},

    # ---- Claude --------------------------------------------------------
    "claude-sonnet-4":       {"provider": "claude", "model": "claude-sonnet-4-20250514",    "prompt_id": "default", "base_model": "claude"},
    "claude-sonnet-3.7":     {"provider": "claude", "model": "claude-3-7-sonnet-latest",   "prompt_id": "default", "base_model": "claude"},
    "claude-haiku-3.5":      {"provider": "claude", "model": "claude-3-5-haiku-latest",    "prompt_id": "default", "base_model": "claude"},
    "claude-sonnet-4.5":     {"provider": "claude", "model": "claude-sonnet-4-5",          "prompt_id": "default", "base_model": "claude"},
    "claude-haiku-4.5":      {"provider": "claude", "model": "claude-haiku-4-5",           "prompt_id": "default", "base_model": "claude"},

    # ---- Gemini --------------------------------------------------------
    "gemini-2.0-flash":        {"provider": "gemini", "model": "gemini-2.0-flash",              "prompt_id": "default", "base_model": "gemini"},
    "gemini-2.0-flash-lite":   {"provider": "gemini", "model": "gemini-2.0-flash-lite",         "prompt_id": "default", "base_model": "gemini"},
    "gemini-2.5-flash":        {"provider": "gemini", "model": "gemini-2.5-flash",          "prompt_id": "default", "base_model": "gemini"},  # Latest experimental
    "gemini-2.5-pro":          {"provider": "gemini", "model": "gemini-2.5-pro", "prompt_id": "default", "base_model": "gemini"},  # Thinking model
    "gemini-3-pro":            {"provider": "gemini", "model": "gemini-3-pro-preview",               "prompt_id": "default", "base_model": "gemini"},  # Latest December experimental

    # ---- our fine-tunes (OLD GENERATION) -----------------------------------------------
    "gpt-4o-old-model":          {"provider": "openai_ft", "model": "ft:gpt-4o-mini-2024-07-18:litero-ai:v4-short-simple:9oaYlNl2",                              "prompt_id": "legacy-finetuned", "base_model": "gpt-4o-mini"},
    "gpt-4o-hum30raw":           {"provider": "openai_ft", "model": "ft:gpt-4o-2024-08-06:litero-ai:hum30raw:BcCFkyvO",                                         "prompt_id": "finetuned", "base_model": "gpt-4o"},
    "gpt-4o-hum40naive":         {"provider": "openai_ft", "model": "ft:gpt-4o-2024-08-06:litero-ai:hum40-naive-auto:Bi0rO31o",                                    "prompt_id": "finetuned", "base_model": "gpt-4o"},
    "gpt-4.1-hum30start":        {"provider": "openai_ft", "model": "ft:gpt-4.1-2025-04-14:litero-ai:hum30start:BcBvzILe",                                         "prompt_id": "finetuned", "base_model": "gpt-4.1"},
    "gpt-4.1-hum40naive":        {"provider": "openai_ft", "model": "ft:gpt-4.1-2025-04-14:litero-ai:hum40-naive-auto:Bi0wCXgi",                                    "prompt_id": "finetuned", "base_model": "gpt-4.1"},
    "gpt-4.1-mini-hum40naive":   {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:hum40-naive-auto:Bi0qnqGa",                               "prompt_id": "finetuned", "base_model": "gpt-4.1-mini"},
    "gpt-4.1-hum30raw-fix1":     {"provider": "openai_ft", "model": "ft:gpt-4.1-2025-04-14:litero-ai:hum30raw-fix1:Bm3eO9ge",                                      "prompt_id": "finetuned", "base_model": "gpt-4.1"},
    "gpt-4o-hum30raw-fix1":      {"provider": "openai_ft", "model": "ft:gpt-4o-2024-08-06:litero-ai:hum30raw-fix1:Bm3b7N9S",                                       "prompt_id": "finetuned", "base_model": "gpt-4o"},
    "gpt-4o-hum30raw-fix2":      {"provider": "openai_ft", "model": "ft:gpt-4o-2024-08-06:litero-ai:hum-30raw-fix2:BmIOvuFc",                                      "prompt_id": "finetuned", "base_model": "gpt-4o"},
    "gpt-4.1-hum30raw-fix2":     {"provider": "openai_ft", "model": "ft:gpt-4.1-2025-04-14:litero-ai:hum-30raw-fix2:BmIQK91Y",                                      "prompt_id": "finetuned", "base_model": "gpt-4.1"},
    "gpt-4o-hum30raw-retrain1":     {"provider": "openai_ft", "model": "ft:gpt-4o-2024-08-06:litero-ai:hum30raw-retrain1:BmIs6zjz",                                      "prompt_id": "finetuned", "base_model": "gpt-4o"},
    "gpt-4o-mini-hum30raw-fix1":     {"provider": "openai_ft", "model": "ft:gpt-4o-mini-2024-07-18:litero-ai:hum30raw-fix1:BmIiNRen",                                      "prompt_id": "finetuned", "base_model": "gpt-4o-mini"},

    # ==================== NEW DYNAMIC MODELS (from new_models.csv) ====================
    # Each model has 3 versions: checkpoint1, checkpoint2, and final

    # ---- gpt-4o-mini based models (6 models × 3 versions = 18 entries) ----
    # Model 1: min-e3-b8-m10-v10
    "min-e3-b8-m10-v10:ckpt1":  {"provider": "openai_ft", "model": "ft:gpt-4o-mini-2024-07-18:litero-ai:min-e3-b8-m10-v10:CahO58ol:ckpt-step-108", "prompt_id": "dynamic", "system_prompt": "minimal_prompt", "scores_in_prompt": "both_raw", "base_model": "gpt-4o-mini"},
    "min-e3-b8-m10-v10:ckpt2":  {"provider": "openai_ft", "model": "ft:gpt-4o-mini-2024-07-18:litero-ai:min-e3-b8-m10-v10:CahO6W88:ckpt-step-216", "prompt_id": "dynamic", "system_prompt": "minimal_prompt", "scores_in_prompt": "both_raw", "base_model": "gpt-4o-mini"},
    "min-e3-b8-m10-v10":        {"provider": "openai_ft", "model": "ft:gpt-4o-mini-2024-07-18:litero-ai:min-e3-b8-m10-v10:CahO6KtY", "prompt_id": "dynamic", "system_prompt": "minimal_prompt", "scores_in_prompt": "both_raw", "base_model": "gpt-4o-mini"},

    # Model 2: raw-e5-b16-m08-v10
    "raw-e5-b16-m08-v10:ckpt1": {"provider": "openai_ft", "model": "ft:gpt-4o-mini-2024-07-18:litero-ai:raw-e5-b16-m08-v10:CaRGPtkb:ckpt-step-189", "prompt_id": "dynamic", "system_prompt": "none", "scores_in_prompt": "none", "base_model": "gpt-4o-mini"},
    "raw-e5-b16-m08-v10:ckpt2": {"provider": "openai_ft", "model": "ft:gpt-4o-mini-2024-07-18:litero-ai:raw-e5-b16-m08-v10:CaRGQuyf:ckpt-step-252", "prompt_id": "dynamic", "system_prompt": "none", "scores_in_prompt": "none", "base_model": "gpt-4o-mini"},
    "raw-e5-b16-m08-v10":       {"provider": "openai_ft", "model": "ft:gpt-4o-mini-2024-07-18:litero-ai:raw-e5-b16-m08-v10:CaRGQPzO", "prompt_id": "dynamic", "system_prompt": "none", "scores_in_prompt": "none", "base_model": "gpt-4o-mini"},

    # Model 3: cmp-e4-b24-m12-v15
    "cmp-e4-b24-m12-v15:ckpt1": {"provider": "openai_ft", "model": "ft:gpt-4o-mini-2024-07-18:litero-ai:cmp-e4-b24-m12-v15:CahcmZUw:ckpt-step-68", "prompt_id": "dynamic", "system_prompt": "compact_prompt", "scores_in_prompt": "both_binned", "base_model": "gpt-4o-mini"},
    "cmp-e4-b24-m12-v15:ckpt2": {"provider": "openai_ft", "model": "ft:gpt-4o-mini-2024-07-18:litero-ai:cmp-e4-b24-m12-v15:Cahcn1EH:ckpt-step-102", "prompt_id": "dynamic", "system_prompt": "compact_prompt", "scores_in_prompt": "both_binned", "base_model": "gpt-4o-mini"},
    "cmp-e4-b24-m12-v15":       {"provider": "openai_ft", "model": "ft:gpt-4o-mini-2024-07-18:litero-ai:cmp-e4-b24-m12-v15:CahcnBqU", "prompt_id": "dynamic", "system_prompt": "compact_prompt", "scores_in_prompt": "both_binned", "base_model": "gpt-4o-mini"},

    # Model 4: min-e6-b16-m05-v20
    "min-e6-b16-m05-v20:ckpt1": {"provider": "openai_ft", "model": "ft:gpt-4o-mini-2024-07-18:litero-ai:min-e6-b16-m05-v20:CahSgyY6:ckpt-step-192", "prompt_id": "dynamic", "system_prompt": "minimal_prompt", "scores_in_prompt": "ai_score_raw", "base_model": "gpt-4o-mini"},
    "min-e6-b16-m05-v20:ckpt2": {"provider": "openai_ft", "model": "ft:gpt-4o-mini-2024-07-18:litero-ai:min-e6-b16-m05-v20:CahSh7LI:ckpt-step-240", "prompt_id": "dynamic", "system_prompt": "minimal_prompt", "scores_in_prompt": "ai_score_raw", "base_model": "gpt-4o-mini"},
    "min-e6-b16-m05-v20":       {"provider": "openai_ft", "model": "ft:gpt-4o-mini-2024-07-18:litero-ai:min-e6-b16-m05-v20:CahShBED", "prompt_id": "dynamic", "system_prompt": "minimal_prompt", "scores_in_prompt": "ai_score_raw", "base_model": "gpt-4o-mini"},

    # Model 5: rubx-e8-b32-m03-v15
    "rubx-e8-b32-m03-v15:ckpt1": {"provider": "openai_ft", "model": "ft:gpt-4o-mini-2024-07-18:litero-ai:rubx-e8-b32-m03-v15:Cb72VhZJ:ckpt-step-156", "prompt_id": "dynamic", "system_prompt": "rich_prompt_with_counter_examples", "scores_in_prompt": "both_raw", "base_model": "gpt-4o-mini"},
    "rubx-e8-b32-m03-v15:ckpt2": {"provider": "openai_ft", "model": "ft:gpt-4o-mini-2024-07-18:litero-ai:rubx-e8-b32-m03-v15:Cb72Vw9D:ckpt-step-182", "prompt_id": "dynamic", "system_prompt": "rich_prompt_with_counter_examples", "scores_in_prompt": "both_raw", "base_model": "gpt-4o-mini"},
    "rubx-e8-b32-m03-v15":      {"provider": "openai_ft", "model": "ft:gpt-4o-mini-2024-07-18:litero-ai:rubx-e8-b32-m03-v15:Cb72Wl7e", "prompt_id": "dynamic", "system_prompt": "rich_prompt_with_counter_examples", "scores_in_prompt": "both_raw", "base_model": "gpt-4o-mini"},

    # Model 6: raw-e3-b8-m15-v10
    "raw-e3-b8-m15-v10:ckpt1":  {"provider": "openai_ft", "model": "ft:gpt-4o-mini-2024-07-18:litero-ai:raw-e3-b8-m15-v10:CaRj5ACC:ckpt-step-114", "prompt_id": "dynamic", "system_prompt": "none", "scores_in_prompt": "ai_score_binned", "base_model": "gpt-4o-mini"},
    "raw-e3-b8-m15-v10:ckpt2":  {"provider": "openai_ft", "model": "ft:gpt-4o-mini-2024-07-18:litero-ai:raw-e3-b8-m15-v10:CaRj5OHp:ckpt-step-228", "prompt_id": "dynamic", "system_prompt": "none", "scores_in_prompt": "ai_score_binned", "base_model": "gpt-4o-mini"},
    "raw-e3-b8-m15-v10":        {"provider": "openai_ft", "model": "ft:gpt-4o-mini-2024-07-18:litero-ai:raw-e3-b8-m15-v10:CaRj65OZ", "prompt_id": "dynamic", "system_prompt": "none", "scores_in_prompt": "ai_score_binned", "base_model": "gpt-4o-mini"},

    # ---- gpt-4.1-mini based models (6 models × 3 versions = 18 entries) ----
    # Model 7: min-e3-b8-m10-v10-d2
    "min-e3-b8-m10-v10-d2:ckpt1": {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:min-e3-b8-m10-v10-d2:CahD9VbJ:ckpt-step-108", "prompt_id": "dynamic", "system_prompt": "minimal_prompt", "scores_in_prompt": "both_raw", "base_model": "gpt-4.1-mini"},
    "min-e3-b8-m10-v10-d2:ckpt2": {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:min-e3-b8-m10-v10-d2:CahDA1Ig:ckpt-step-216", "prompt_id": "dynamic", "system_prompt": "minimal_prompt", "scores_in_prompt": "both_raw", "base_model": "gpt-4.1-mini"},
    "min-e3-b8-m10-v10-d2":     {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:min-e3-b8-m10-v10-d2:CahDAVzH", "prompt_id": "dynamic", "system_prompt": "minimal_prompt", "scores_in_prompt": "both_raw", "base_model": "gpt-4.1-mini"},

    # Model 8: cmp-e5-b16-m08-v15
    "cmp-e5-b16-m08-v15:ckpt1": {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:cmp-e5-b16-m08-v15:CahetWak:ckpt-step-153", "prompt_id": "dynamic", "system_prompt": "compact_prompt", "scores_in_prompt": "both_binned", "base_model": "gpt-4.1-mini"},
    "cmp-e5-b16-m08-v15:ckpt2": {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:cmp-e5-b16-m08-v15:CahetKtG:ckpt-step-204", "prompt_id": "dynamic", "system_prompt": "compact_prompt", "scores_in_prompt": "both_binned", "base_model": "gpt-4.1-mini"},
    "cmp-e5-b16-m08-v15":       {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:cmp-e5-b16-m08-v15:CaheuKck", "prompt_id": "dynamic", "system_prompt": "compact_prompt", "scores_in_prompt": "both_binned", "base_model": "gpt-4.1-mini"},

    # Model 9: raw-e4-b24-m06-v20
    "raw-e4-b24-m06-v20:ckpt1": {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:raw-e4-b24-m06-v20:CaRJbV6v:ckpt-step-76", "prompt_id": "dynamic", "system_prompt": "none", "scores_in_prompt": "none", "base_model": "gpt-4.1-mini"},
    "raw-e4-b24-m06-v20:ckpt2": {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:raw-e4-b24-m06-v20:CaRJcxPt:ckpt-step-114", "prompt_id": "dynamic", "system_prompt": "none", "scores_in_prompt": "none", "base_model": "gpt-4.1-mini"},
    "raw-e4-b24-m06-v20":       {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:raw-e4-b24-m06-v20:CaRJcZNh", "prompt_id": "dynamic", "system_prompt": "none", "scores_in_prompt": "none", "base_model": "gpt-4.1-mini"},

    # Model 10: rdesc-e6-b20-m10-v10
    "rdesc-e6-b20-m10-v10:ckpt1": {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:rdesc-e6-b20-m10-v10:Cb6zK3tG:ckpt-step-172", "prompt_id": "dynamic", "system_prompt": "rich_prompt_with_negative_examples", "scores_in_prompt": "ai_score_raw", "base_model": "gpt-4.1-mini"},
    "rdesc-e6-b20-m10-v10:ckpt2": {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:rdesc-e6-b20-m10-v10:Cb6zLzME:ckpt-step-215", "prompt_id": "dynamic", "system_prompt": "rich_prompt_with_negative_examples", "scores_in_prompt": "ai_score_raw", "base_model": "gpt-4.1-mini"},
    "rdesc-e6-b20-m10-v10":     {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:rdesc-e6-b20-m10-v10:Cb6zLrfo", "prompt_id": "dynamic", "system_prompt": "rich_prompt_with_negative_examples", "scores_in_prompt": "ai_score_raw", "base_model": "gpt-4.1-mini"},

    # Model 11: min-e8-b32-m04-v20
    "min-e8-b32-m04-v20:ckpt1": {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:min-e8-b32-m04-v20:CahGHoQH:ckpt-step-144", "prompt_id": "dynamic", "system_prompt": "minimal_prompt", "scores_in_prompt": "both_raw", "base_model": "gpt-4.1-mini"},
    "min-e8-b32-m04-v20:ckpt2": {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:min-e8-b32-m04-v20:CahGHE9U:ckpt-step-168", "prompt_id": "dynamic", "system_prompt": "minimal_prompt", "scores_in_prompt": "both_raw", "base_model": "gpt-4.1-mini"},
    "min-e8-b32-m04-v20":       {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:min-e8-b32-m04-v20:CahGHp0S", "prompt_id": "dynamic", "system_prompt": "minimal_prompt", "scores_in_prompt": "both_raw", "base_model": "gpt-4.1-mini"},

    # Model 12: cmp-e3-b12-m12-v12
    "cmp-e3-b12-m12-v12:ckpt1": {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:cmp-e3-b12-m12-v12:CahihiXZ:ckpt-step-70", "prompt_id": "dynamic", "system_prompt": "compact_prompt", "scores_in_prompt": "ai_score_binned", "base_model": "gpt-4.1-mini"},
    "cmp-e3-b12-m12-v12:ckpt2": {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:cmp-e3-b12-m12-v12:CahiiRkT:ckpt-step-140", "prompt_id": "dynamic", "system_prompt": "compact_prompt", "scores_in_prompt": "ai_score_binned", "base_model": "gpt-4.1-mini"},
    "cmp-e3-b12-m12-v12":       {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:cmp-e3-b12-m12-v12:Cahiin3j", "prompt_id": "dynamic", "system_prompt": "compact_prompt", "scores_in_prompt": "ai_score_binned", "base_model": "gpt-4.1-mini"},

    # ---- gpt-4.1 based models (6 models × 3 versions = 18 entries) ----
    # Model 13: min-e3-b6-m08-v15
    "min-e3-b6-m08-v15:ckpt1":  {"provider": "openai_ft", "model": "ft:gpt-4.1-2025-04-14:litero-ai:min-e3-b6-m08-v15:CahTUP8d:ckpt-step-136", "prompt_id": "dynamic", "system_prompt": "minimal_prompt", "scores_in_prompt": "both_raw", "base_model": "gpt-4.1"},
    "min-e3-b6-m08-v15:ckpt2":  {"provider": "openai_ft", "model": "ft:gpt-4.1-2025-04-14:litero-ai:min-e3-b6-m08-v15:CahTVy42:ckpt-step-272", "prompt_id": "dynamic", "system_prompt": "minimal_prompt", "scores_in_prompt": "both_raw", "base_model": "gpt-4.1"},
    "min-e3-b6-m08-v15":        {"provider": "openai_ft", "model": "ft:gpt-4.1-2025-04-14:litero-ai:min-e3-b6-m08-v15:CahTVFjJ", "prompt_id": "dynamic", "system_prompt": "minimal_prompt", "scores_in_prompt": "both_raw", "base_model": "gpt-4.1"},

    # Model 14: raw-e4-b8-m06-v10
    "raw-e4-b8-m06-v10:ckpt1":  {"provider": "openai_ft", "model": "ft:gpt-4.1-2025-04-14:litero-ai:raw-e4-b8-m06-v10:CaRhYycm:ckpt-step-252", "prompt_id": "dynamic", "system_prompt": "none", "scores_in_prompt": "none", "base_model": "gpt-4.1"},
    "raw-e4-b8-m06-v10:ckpt2":  {"provider": "openai_ft", "model": "ft:gpt-4.1-2025-04-14:litero-ai:raw-e4-b8-m06-v10:CaRhZlkW:ckpt-step-378", "prompt_id": "dynamic", "system_prompt": "none", "scores_in_prompt": "none", "base_model": "gpt-4.1"},
    "raw-e4-b8-m06-v10":        {"provider": "openai_ft", "model": "ft:gpt-4.1-2025-04-14:litero-ai:raw-e4-b8-m06-v10:CaRhZ8la", "prompt_id": "dynamic", "system_prompt": "none", "scores_in_prompt": "none", "base_model": "gpt-4.1"},

    # Model 15: cmp-e5-b8-m05-v20
    "cmp-e5-b8-m05-v20:ckpt1":  {"provider": "openai_ft", "model": "ft:gpt-4.1-2025-04-14:litero-ai:cmp-e5-b8-m05-v20:Cai16isJ:ckpt-step-288", "prompt_id": "dynamic", "system_prompt": "compact_prompt", "scores_in_prompt": "both_binned", "base_model": "gpt-4.1"},
    "cmp-e5-b8-m05-v20:ckpt2":  {"provider": "openai_ft", "model": "ft:gpt-4.1-2025-04-14:litero-ai:cmp-e5-b8-m05-v20:Cai16kn4:ckpt-step-384", "prompt_id": "dynamic", "system_prompt": "compact_prompt", "scores_in_prompt": "both_binned", "base_model": "gpt-4.1"},
    "cmp-e5-b8-m05-v20":        {"provider": "openai_ft", "model": "ft:gpt-4.1-2025-04-14:litero-ai:cmp-e5-b8-m05-v20:Cai16L5a", "prompt_id": "dynamic", "system_prompt": "compact_prompt", "scores_in_prompt": "both_binned", "base_model": "gpt-4.1"},

    # Model 16: rchi-e6-b12-m04-v15
    "rchi-e6-b12-m04-v15:ckpt1": {"provider": "openai_ft", "model": "ft:gpt-4.1-2025-04-14:litero-ai:rchi-e6-b12-m04-v15:Cb68OJQz:ckpt-step-272", "prompt_id": "dynamic", "system_prompt": "rich_prompt_with_focus_areas", "scores_in_prompt": "both_raw", "base_model": "gpt-4.1"},
    "rchi-e6-b12-m04-v15:ckpt2": {"provider": "openai_ft", "model": "ft:gpt-4.1-2025-04-14:litero-ai:rchi-e6-b12-m04-v15:Cb68PLxR:ckpt-step-340", "prompt_id": "dynamic", "system_prompt": "rich_prompt_with_focus_areas", "scores_in_prompt": "both_raw", "base_model": "gpt-4.1"},
    "rchi-e6-b12-m04-v15":      {"provider": "openai_ft", "model": "ft:gpt-4.1-2025-04-14:litero-ai:rchi-e6-b12-m04-v15:Cb68PEiE", "prompt_id": "dynamic", "system_prompt": "rich_prompt_with_focus_areas", "scores_in_prompt": "both_raw", "base_model": "gpt-4.1"},

    # Model 17: min-e8-b16-m03-v20
    "min-e8-b16-m03-v20:ckpt1": {"provider": "openai_ft", "model": "ft:gpt-4.1-2025-04-14:litero-ai:min-e8-b16-m03-v20:Caha6hf8:ckpt-step-288", "prompt_id": "dynamic", "system_prompt": "minimal_prompt", "scores_in_prompt": "both_raw", "base_model": "gpt-4.1"},
    "min-e8-b16-m03-v20:ckpt2": {"provider": "openai_ft", "model": "ft:gpt-4.1-2025-04-14:litero-ai:min-e8-b16-m03-v20:Caha7Y0p:ckpt-step-336", "prompt_id": "dynamic", "system_prompt": "minimal_prompt", "scores_in_prompt": "both_raw", "base_model": "gpt-4.1"},
    "min-e8-b16-m03-v20":       {"provider": "openai_ft", "model": "ft:gpt-4.1-2025-04-14:litero-ai:min-e8-b16-m03-v20:Caha7LoY", "prompt_id": "dynamic", "system_prompt": "minimal_prompt", "scores_in_prompt": "both_raw", "base_model": "gpt-4.1"},

    # Model 18: raw-e3-b10-m10-v10 (only has final version, no checkpoints)
    "raw-e3-b10-m10-v10":       {"provider": "openai_ft", "model": "ft:gpt-4.1-2025-04-14:litero-ai:raw-e3-b10-m10-v10:CaSOMjS9", "prompt_id": "dynamic", "system_prompt": "none", "scores_in_prompt": "ai_score_raw", "base_model": "gpt-4.1"},

    # ---- gpt-4.1-nano based models (4 models × 3 versions = 12 entries) ----
    # Model 19: min-e3-b32-m10-v10
    "min-e3-b32-m10-v10:ckpt1": {"provider": "openai_ft", "model": "ft:gpt-4.1-nano-2025-04-14:litero-ai:min-e3-b32-m10-v10:CaicX7xT:ckpt-step-27", "prompt_id": "dynamic", "system_prompt": "minimal_prompt", "scores_in_prompt": "both_binned", "base_model": "gpt-4.1-nano"},
    "min-e3-b32-m10-v10:ckpt2": {"provider": "openai_ft", "model": "ft:gpt-4.1-nano-2025-04-14:litero-ai:min-e3-b32-m10-v10:CaicYHw5:ckpt-step-54", "prompt_id": "dynamic", "system_prompt": "minimal_prompt", "scores_in_prompt": "both_binned", "base_model": "gpt-4.1-nano"},
    "min-e3-b32-m10-v10":       {"provider": "openai_ft", "model": "ft:gpt-4.1-nano-2025-04-14:litero-ai:min-e3-b32-m10-v10:CaicZpmo", "prompt_id": "dynamic", "system_prompt": "minimal_prompt", "scores_in_prompt": "both_binned", "base_model": "gpt-4.1-nano"},

    # Model 20: raw-e5-b48-m08-v15
    "raw-e5-b48-m08-v15:ckpt1": {"provider": "openai_ft", "model": "ft:gpt-4.1-nano-2025-04-14:litero-ai:raw-e5-b48-m08-v15:CaRKfL72:ckpt-step-90", "prompt_id": "dynamic", "system_prompt": "none", "scores_in_prompt": "none", "base_model": "gpt-4.1-nano"},
    "raw-e5-b48-m08-v15:ckpt2": {"provider": "openai_ft", "model": "ft:gpt-4.1-nano-2025-04-14:litero-ai:raw-e5-b48-m08-v15:CaRKg4ph:ckpt-step-120", "prompt_id": "dynamic", "system_prompt": "none", "scores_in_prompt": "none", "base_model": "gpt-4.1-nano"},
    "raw-e5-b48-m08-v15":       {"provider": "openai_ft", "model": "ft:gpt-4.1-nano-2025-04-14:litero-ai:raw-e5-b48-m08-v15:CaRKgWab", "prompt_id": "dynamic", "system_prompt": "none", "scores_in_prompt": "none", "base_model": "gpt-4.1-nano"},

    # Model 22: min-e4-b40-m12-v12
    "min-e4-b40-m12-v12:ckpt1": {"provider": "openai_ft", "model": "ft:gpt-4.1-nano-2025-04-14:litero-ai:min-e4-b40-m12-v12:CahPt2nD:ckpt-step-54", "prompt_id": "dynamic", "system_prompt": "minimal_prompt", "scores_in_prompt": "ai_score_binned", "base_model": "gpt-4.1-nano"},
    "min-e4-b40-m12-v12:ckpt2": {"provider": "openai_ft", "model": "ft:gpt-4.1-nano-2025-04-14:litero-ai:min-e4-b40-m12-v12:CahPueRR:ckpt-step-81", "prompt_id": "dynamic", "system_prompt": "minimal_prompt", "scores_in_prompt": "ai_score_binned", "base_model": "gpt-4.1-nano"},
    "min-e4-b40-m12-v12":       {"provider": "openai_ft", "model": "ft:gpt-4.1-nano-2025-04-14:litero-ai:min-e4-b40-m12-v12:CahPuZKW", "prompt_id": "dynamic", "system_prompt": "minimal_prompt", "scores_in_prompt": "ai_score_binned", "base_model": "gpt-4.1-nano"},

    # Model 24: raw-e3-b24-m15-v10
    "raw-e3-b24-m15-v10:ckpt1": {"provider": "openai_ft", "model": "ft:gpt-4.1-nano-2025-04-14:litero-ai:raw-e3-b24-m15-v10:CaRJsenZ:ckpt-step-42", "prompt_id": "dynamic", "system_prompt": "none", "scores_in_prompt": "none", "base_model": "gpt-4.1-nano"},
    "raw-e3-b24-m15-v10:ckpt2": {"provider": "openai_ft", "model": "ft:gpt-4.1-nano-2025-04-14:litero-ai:raw-e3-b24-m15-v10:CaRJtQb3:ckpt-step-84", "prompt_id": "dynamic", "system_prompt": "none", "scores_in_prompt": "none", "base_model": "gpt-4.1-nano"},
    "raw-e3-b24-m15-v10":       {"provider": "openai_ft", "model": "ft:gpt-4.1-nano-2025-04-14:litero-ai:raw-e3-b24-m15-v10:CaRJtSWZ", "prompt_id": "dynamic", "system_prompt": "none", "scores_in_prompt": "none", "base_model": "gpt-4.1-nano"},

    # ---- gpt-4o based models (6 models × 3 versions = 18 entries) ----
    # Model 25: min-e3-b4-m06-v15
    "min-e3-b4-m06-v15:ckpt1":  {"provider": "openai_ft", "model": "ft:gpt-4o-2024-08-06:litero-ai:min-e3-b4-m06-v15:Cahdrt9p:ckpt-step-203", "prompt_id": "dynamic", "system_prompt": "minimal_prompt", "scores_in_prompt": "both_raw", "base_model": "gpt-4o"},
    "min-e3-b4-m06-v15:ckpt2":  {"provider": "openai_ft", "model": "ft:gpt-4o-2024-08-06:litero-ai:min-e3-b4-m06-v15:Cahds7GM:ckpt-step-406", "prompt_id": "dynamic", "system_prompt": "minimal_prompt", "scores_in_prompt": "both_raw", "base_model": "gpt-4o"},
    "min-e3-b4-m06-v15":        {"provider": "openai_ft", "model": "ft:gpt-4o-2024-08-06:litero-ai:min-e3-b4-m06-v15:Cahds3Pm", "prompt_id": "dynamic", "system_prompt": "minimal_prompt", "scores_in_prompt": "both_raw", "base_model": "gpt-4o"},

    # Model 26: raw-e4-b6-m05-v10
    "raw-e4-b6-m05-v10:ckpt1":  {"provider": "openai_ft", "model": "ft:gpt-4o-2024-08-06:litero-ai:raw-e4-b6-m05-v10:CaRkYfnt:ckpt-step-336", "prompt_id": "dynamic", "system_prompt": "none", "scores_in_prompt": "none", "base_model": "gpt-4o"},
    "raw-e4-b6-m05-v10:ckpt2":  {"provider": "openai_ft", "model": "ft:gpt-4o-2024-08-06:litero-ai:raw-e4-b6-m05-v10:CaRkZrAk:ckpt-step-504", "prompt_id": "dynamic", "system_prompt": "none", "scores_in_prompt": "none", "base_model": "gpt-4o"},
    "raw-e4-b6-m05-v10":        {"provider": "openai_ft", "model": "ft:gpt-4o-2024-08-06:litero-ai:raw-e4-b6-m05-v10:CaRkZH9u", "prompt_id": "dynamic", "system_prompt": "none", "scores_in_prompt": "none", "base_model": "gpt-4o"},

    # Model 27: cmp-e5-b6-m04-v20
    "cmp-e5-b6-m04-v20:ckpt1":  {"provider": "openai_ft", "model": "ft:gpt-4o-2024-08-06:litero-ai:cmp-e5-b6-m04-v20:CaiClBIF:ckpt-step-384", "prompt_id": "dynamic", "system_prompt": "compact_prompt", "scores_in_prompt": "both_binned", "base_model": "gpt-4o"},
    "cmp-e5-b6-m04-v20:ckpt2":  {"provider": "openai_ft", "model": "ft:gpt-4o-2024-08-06:litero-ai:cmp-e5-b6-m04-v20:CaiCm12C:ckpt-step-512", "prompt_id": "dynamic", "system_prompt": "compact_prompt", "scores_in_prompt": "both_binned", "base_model": "gpt-4o"},
    "cmp-e5-b6-m04-v20":        {"provider": "openai_ft", "model": "ft:gpt-4o-2024-08-06:litero-ai:cmp-e5-b6-m04-v20:CaiCmOVf", "prompt_id": "dynamic", "system_prompt": "compact_prompt", "scores_in_prompt": "both_binned", "base_model": "gpt-4o"},

    # Model 28: rich-e6-b8-m03-v20
    "rich-e6-b8-m03-v20:ckpt1": {"provider": "openai_ft", "model": "ft:gpt-4o-2024-08-06:litero-ai:rich-e6-b8-m03-v20:Cb5VHMR5:ckpt-step-384", "prompt_id": "dynamic", "system_prompt": "rich_prompt_standard", "scores_in_prompt": "both_raw", "base_model": "gpt-4o"},
    "rich-e6-b8-m03-v20:ckpt2": {"provider": "openai_ft", "model": "ft:gpt-4o-2024-08-06:litero-ai:rich-e6-b8-m03-v20:Cb5VIF54:ckpt-step-480", "prompt_id": "dynamic", "system_prompt": "rich_prompt_standard", "scores_in_prompt": "both_raw", "base_model": "gpt-4o"},
    "rich-e6-b8-m03-v20":       {"provider": "openai_ft", "model": "ft:gpt-4o-2024-08-06:litero-ai:rich-e6-b8-m03-v20:Cb5VI1BO", "prompt_id": "dynamic", "system_prompt": "rich_prompt_standard", "scores_in_prompt": "both_raw", "base_model": "gpt-4o"},

    # Model 29: min-e3-b5-m10-v10
    "min-e3-b5-m10-v10:ckpt1":  {"provider": "openai_ft", "model": "ft:gpt-4o-2024-08-06:litero-ai:min-e3-b5-m10-v10:CahmvOM9:ckpt-step-172", "prompt_id": "dynamic", "system_prompt": "minimal_prompt", "scores_in_prompt": "ai_score_raw", "base_model": "gpt-4o"},
    "min-e3-b5-m10-v10:ckpt2":  {"provider": "openai_ft", "model": "ft:gpt-4o-2024-08-06:litero-ai:min-e3-b5-m10-v10:CahmwSyI:ckpt-step-344", "prompt_id": "dynamic", "system_prompt": "minimal_prompt", "scores_in_prompt": "ai_score_raw", "base_model": "gpt-4o"},
    "min-e3-b5-m10-v10":        {"provider": "openai_ft", "model": "ft:gpt-4o-2024-08-06:litero-ai:min-e3-b5-m10-v10:CahmwAHN", "prompt_id": "dynamic", "system_prompt": "minimal_prompt", "scores_in_prompt": "ai_score_raw", "base_model": "gpt-4o"},

    # Model 30: raw-e8-b8-m025-v20
    "raw-e8-b8-m025-v20:ckpt1": {"provider": "openai_ft", "model": "ft:gpt-4o-2024-08-06:litero-ai:raw-e8-b8-m025-v20:CaS2mF6q:ckpt-step-672", "prompt_id": "dynamic", "system_prompt": "none", "scores_in_prompt": "none", "base_model": "gpt-4o"},
    "raw-e8-b8-m025-v20:ckpt2": {"provider": "openai_ft", "model": "ft:gpt-4o-2024-08-06:litero-ai:raw-e8-b8-m025-v20:CaS2nu1l:ckpt-step-784", "prompt_id": "dynamic", "system_prompt": "none", "scores_in_prompt": "none", "base_model": "gpt-4o"},
    "raw-e8-b8-m025-v20":       {"provider": "openai_ft", "model": "ft:gpt-4o-2024-08-06:litero-ai:raw-e8-b8-m025-v20:CaS2nSDe", "prompt_id": "dynamic", "system_prompt": "none", "scores_in_prompt": "none", "base_model": "gpt-4o"},

    # ==================== NEW DPO MODELS (Batch 2) ====================
    # ---- gpt-4.1-mini based models ----
    # Model 1: dpo-41m-min-e3-b32-m10-k05-v10
    "dpo-41m-min-e3-b32-m10-k05-v10:ckpt1": {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:dpo-41m-min-e3-b32-m10-k05-v10:CgqrRupO:ckpt-step-27", "prompt_id": "dynamic", "system_prompt": "minimal_style_guardrails", "scores_in_prompt": "both_raw", "base_model": "gpt-4.1-mini"},
    "dpo-41m-min-e3-b32-m10-k05-v10:ckpt2": {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:dpo-41m-min-e3-b32-m10-k05-v10:CgqrSdJo:ckpt-step-54", "prompt_id": "dynamic", "system_prompt": "minimal_style_guardrails", "scores_in_prompt": "both_raw", "base_model": "gpt-4.1-mini"},
    "dpo-41m-min-e3-b32-m10-k05-v10":       {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:dpo-41m-min-e3-b32-m10-k05-v10:CgqrSkIF", "prompt_id": "dynamic", "system_prompt": "minimal_style_guardrails", "scores_in_prompt": "both_raw", "base_model": "gpt-4.1-mini"},

    # Model 2: dpo-41m-raw-e4-b32-m08-k08-v10
    "dpo-41m-raw-e4-b32-m08-k08-v10:ckpt1": {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:dpo-41m-raw-e4-b32-m08-k08-v10:Chx7RaD0:ckpt-step-54", "prompt_id": "dynamic", "system_prompt": "none", "scores_in_prompt": "none", "base_model": "gpt-4.1-mini"},
    "dpo-41m-raw-e4-b32-m08-k08-v10:ckpt2": {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:dpo-41m-raw-e4-b32-m08-k08-v10:Chx7S16B:ckpt-step-81", "prompt_id": "dynamic", "system_prompt": "none", "scores_in_prompt": "none", "base_model": "gpt-4.1-mini"},
    "dpo-41m-raw-e4-b32-m08-k08-v10":       {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:dpo-41m-raw-e4-b32-m08-k08-v10:Chx7SVST", "prompt_id": "dynamic", "system_prompt": "none", "scores_in_prompt": "none", "base_model": "gpt-4.1-mini"},

    # Model 3: dpo-41m-cmp-e3-b32-m10-k08-v15
    "dpo-41m-cmp-e3-b32-m10-k08-v15:ckpt1": {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:dpo-41m-cmp-e3-b32-m10-k08-v15:ChxQgnW0:ckpt-step-26", "prompt_id": "dynamic", "system_prompt": "compact_guidelines_rubric", "scores_in_prompt": "ai_score_raw", "base_model": "gpt-4.1-mini"},
    "dpo-41m-cmp-e3-b32-m10-k08-v15:ckpt2": {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:dpo-41m-cmp-e3-b32-m10-k08-v15:ChxQhduq:ckpt-step-52", "prompt_id": "dynamic", "system_prompt": "compact_guidelines_rubric", "scores_in_prompt": "ai_score_raw", "base_model": "gpt-4.1-mini"},
    "dpo-41m-cmp-e3-b32-m10-k08-v15":       {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:dpo-41m-cmp-e3-b32-m10-k08-v15:ChxQhRw8", "prompt_id": "dynamic", "system_prompt": "compact_guidelines_rubric", "scores_in_prompt": "ai_score_raw", "base_model": "gpt-4.1-mini"},

    # Model 4: dpo-41m-rch-e4-b32-m06-k08-v15
    "dpo-41m-rch-e4-b32-m06-k08-v15:ckpt1": {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:dpo-41m-rch-e4-b32-m06-k08-v15:Chw9oUHw:ckpt-step-52", "prompt_id": "dynamic", "system_prompt": "rich_prompt_standard", "scores_in_prompt": "both_binned", "base_model": "gpt-4.1-mini"},
    "dpo-41m-rch-e4-b32-m06-k08-v15:ckpt2": {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:dpo-41m-rch-e4-b32-m06-k08-v15:Chw9pyRF:ckpt-step-78", "prompt_id": "dynamic", "system_prompt": "rich_prompt_standard", "scores_in_prompt": "both_binned", "base_model": "gpt-4.1-mini"},
    "dpo-41m-rch-e4-b32-m06-k08-v15":       {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:dpo-41m-rch-e4-b32-m06-k08-v15:Chw9qN1z", "prompt_id": "dynamic", "system_prompt": "rich_prompt_standard", "scores_in_prompt": "both_binned", "base_model": "gpt-4.1-mini"},

    # ---- gpt-4.1 based models ----
    # Model 5: dpo-41-min-e3-b16-m08-k05-v10
    "dpo-41-min-e3-b16-m08-k05-v10:ckpt1": {"provider": "openai_ft", "model": "ft:gpt-4.1-2025-04-14:litero-ai:dpo-41-min-e3-b16-m08-k05-v10:CgrtS8V3:ckpt-step-54", "prompt_id": "dynamic", "system_prompt": "minimal_style_guardrails", "scores_in_prompt": "both_raw", "base_model": "gpt-4.1"},
    "dpo-41-min-e3-b16-m08-k05-v10:ckpt2": {"provider": "openai_ft", "model": "ft:gpt-4.1-2025-04-14:litero-ai:dpo-41-min-e3-b16-m08-k05-v10:CgrtSEIa:ckpt-step-108", "prompt_id": "dynamic", "system_prompt": "minimal_style_guardrails", "scores_in_prompt": "both_raw", "base_model": "gpt-4.1"},
    "dpo-41-min-e3-b16-m08-k05-v10":       {"provider": "openai_ft", "model": "ft:gpt-4.1-2025-04-14:litero-ai:dpo-41-min-e3-b16-m08-k05-v10:CgrtSQIo", "prompt_id": "dynamic", "system_prompt": "minimal_style_guardrails", "scores_in_prompt": "both_raw", "base_model": "gpt-4.1"},

    # Model 6: dpo-41-cmp-e4-b16-m06-k08-v15
    "dpo-41-cmp-e4-b16-m06-k08-v15:ckpt1": {"provider": "openai_ft", "model": "ft:gpt-4.1-2025-04-14:litero-ai:dpo-41-cmp-e4-b16-m06-k08-v15:Ci5BliaW:ckpt-step-102", "prompt_id": "dynamic", "system_prompt": "compact_guidelines_rubric", "scores_in_prompt": "ai_score_binned", "base_model": "gpt-4.1"},
    "dpo-41-cmp-e4-b16-m06-k08-v15:ckpt2": {"provider": "openai_ft", "model": "ft:gpt-4.1-2025-04-14:litero-ai:dpo-41-cmp-e4-b16-m06-k08-v15:Ci5BmKuA:ckpt-step-153", "prompt_id": "dynamic", "system_prompt": "compact_guidelines_rubric", "scores_in_prompt": "ai_score_binned", "base_model": "gpt-4.1"},
    "dpo-41-cmp-e4-b16-m06-k08-v15":       {"provider": "openai_ft", "model": "ft:gpt-4.1-2025-04-14:litero-ai:dpo-41-cmp-e4-b16-m06-k08-v15:Ci5BmBWl", "prompt_id": "dynamic", "system_prompt": "compact_guidelines_rubric", "scores_in_prompt": "ai_score_binned", "base_model": "gpt-4.1"},

    # Model 7: dpo-41-rch-e3-b12-m10-k10-v10
    "dpo-41-rch-e3-b12-m10-k10-v10:ckpt1": {"provider": "openai_ft", "model": "ft:gpt-4.1-2025-04-14:litero-ai:dpo-41-rch-e3-b12-m10-k10-v10:ChxNBU3m:ckpt-step-72", "prompt_id": "dynamic", "system_prompt": "rich_prompt_standard", "scores_in_prompt": "ai_score_raw", "base_model": "gpt-4.1"},
    "dpo-41-rch-e3-b12-m10-k10-v10:ckpt2": {"provider": "openai_ft", "model": "ft:gpt-4.1-2025-04-14:litero-ai:dpo-41-rch-e3-b12-m10-k10-v10:ChxNCq4h:ckpt-step-144", "prompt_id": "dynamic", "system_prompt": "rich_prompt_standard", "scores_in_prompt": "ai_score_raw", "base_model": "gpt-4.1"},
    "dpo-41-rch-e3-b12-m10-k10-v10":       {"provider": "openai_ft", "model": "ft:gpt-4.1-2025-04-14:litero-ai:dpo-41-rch-e3-b12-m10-k10-v10:ChxNCnaB", "prompt_id": "dynamic", "system_prompt": "rich_prompt_standard", "scores_in_prompt": "ai_score_raw", "base_model": "gpt-4.1"},

    # Model 8: dpo-41-raw-e5-b16-m05-k08-v20
    "dpo-41-raw-e5-b16-m05-k08-v20:ckpt1": {"provider": "openai_ft", "model": "ft:gpt-4.1-2025-04-14:litero-ai:dpo-41-raw-e5-b16-m05-k08-v20:Chz9w8jc:ckpt-step-144", "prompt_id": "dynamic", "system_prompt": "none", "scores_in_prompt": "none", "base_model": "gpt-4.1"},
    "dpo-41-raw-e5-b16-m05-k08-v20:ckpt2": {"provider": "openai_ft", "model": "ft:gpt-4.1-2025-04-14:litero-ai:dpo-41-raw-e5-b16-m05-k08-v20:Chz9xt1n:ckpt-step-192", "prompt_id": "dynamic", "system_prompt": "none", "scores_in_prompt": "none", "base_model": "gpt-4.1"},
    "dpo-41-raw-e5-b16-m05-k08-v20":       {"provider": "openai_ft", "model": "ft:gpt-4.1-2025-04-14:litero-ai:dpo-41-raw-e5-b16-m05-k08-v20:Chz9xgZc", "prompt_id": "dynamic", "system_prompt": "none", "scores_in_prompt": "none", "base_model": "gpt-4.1"},

    # ---- gpt-4o based models ----
    # Model 9: dpo-4o-min-e3-b16-m08-k05-v10
    "dpo-4o-min-e3-b16-m08-k05-v10:ckpt1": {"provider": "openai_ft", "model": "ft:gpt-4o-2024-08-06:litero-ai:dpo-4o-min-e3-b16-m08-k05-v10:Cgt0Y027:ckpt-step-54", "prompt_id": "dynamic", "system_prompt": "minimal_style_guardrails", "scores_in_prompt": "both_raw", "base_model": "gpt-4o"},
    "dpo-4o-min-e3-b16-m08-k05-v10:ckpt2": {"provider": "openai_ft", "model": "ft:gpt-4o-2024-08-06:litero-ai:dpo-4o-min-e3-b16-m08-k05-v10:Cgt0ZKPO:ckpt-step-108", "prompt_id": "dynamic", "system_prompt": "minimal_style_guardrails", "scores_in_prompt": "both_raw", "base_model": "gpt-4o"},
    "dpo-4o-min-e3-b16-m08-k05-v10":       {"provider": "openai_ft", "model": "ft:gpt-4o-2024-08-06:litero-ai:dpo-4o-min-e3-b16-m08-k05-v10:Cgt0ZHQh", "prompt_id": "dynamic", "system_prompt": "minimal_style_guardrails", "scores_in_prompt": "both_raw", "base_model": "gpt-4o"},

    # ---- gpt-4.1-nano based models ----
    # Model 11: dpo-nano-min-e3-b32-m10-k05-v10
    "dpo-nano-min-e3-b32-m10-k05-v10:ckpt1": {"provider": "openai_ft", "model": "ft:gpt-4.1-nano-2025-04-14:litero-ai:dpo-nano-min-e3-b32-m10-k05-v10:Cgr1jneM:ckpt-step-27", "prompt_id": "dynamic", "system_prompt": "minimal_style_guardrails", "scores_in_prompt": "both_raw", "base_model": "gpt-4.1-nano"},
    "dpo-nano-min-e3-b32-m10-k05-v10:ckpt2": {"provider": "openai_ft", "model": "ft:gpt-4.1-nano-2025-04-14:litero-ai:dpo-nano-min-e3-b32-m10-k05-v10:Cgr1kpQS:ckpt-step-54", "prompt_id": "dynamic", "system_prompt": "minimal_style_guardrails", "scores_in_prompt": "both_raw", "base_model": "gpt-4.1-nano"},
    "dpo-nano-min-e3-b32-m10-k05-v10":       {"provider": "openai_ft", "model": "ft:gpt-4.1-nano-2025-04-14:litero-ai:dpo-nano-min-e3-b32-m10-k05-v10:Cgr1keHI", "prompt_id": "dynamic", "system_prompt": "minimal_style_guardrails", "scores_in_prompt": "both_raw", "base_model": "gpt-4.1-nano"},

    # Model 12: dpo-nano-cmp-e4-b32-m08-k08-v15
    "dpo-nano-cmp-e4-b32-m08-k08-v15:ckpt1": {"provider": "openai_ft", "model": "ft:gpt-4.1-nano-2025-04-14:litero-ai:dpo-nano-cmp-e4-b32-m08-k08-v15:Ci0SZqNW:ckpt-step-52", "prompt_id": "dynamic", "system_prompt": "compact_guidelines_rubric", "scores_in_prompt": "none", "base_model": "gpt-4.1-nano"},
    "dpo-nano-cmp-e4-b32-m08-k08-v15:ckpt2": {"provider": "openai_ft", "model": "ft:gpt-4.1-nano-2025-04-14:litero-ai:dpo-nano-cmp-e4-b32-m08-k08-v15:Ci0SazSy:ckpt-step-78", "prompt_id": "dynamic", "system_prompt": "compact_guidelines_rubric", "scores_in_prompt": "none", "base_model": "gpt-4.1-nano"},
    "dpo-nano-cmp-e4-b32-m08-k08-v15":       {"provider": "openai_ft", "model": "ft:gpt-4.1-nano-2025-04-14:litero-ai:dpo-nano-cmp-e4-b32-m08-k08-v15:Ci0SaiP7", "prompt_id": "dynamic", "system_prompt": "compact_guidelines_rubric", "scores_in_prompt": "none", "base_model": "gpt-4.1-nano"},

    # ---- More gpt-4.1-mini based models ----
    # Model 13: dpo-41m-min-e3-b8-m06-k08-v15
    "dpo-41m-min-e3-b8-m06-k08-v15:ckpt1": {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:dpo-41m-min-e3-b8-m06-k08-v15:CgrsffSJ:ckpt-step-102", "prompt_id": "dynamic", "system_prompt": "minimal_style_guardrails", "scores_in_prompt": "both_raw", "base_model": "gpt-4.1-mini"},
    "dpo-41m-min-e3-b8-m06-k08-v15:ckpt2": {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:dpo-41m-min-e3-b8-m06-k08-v15:CgrsgAH2:ckpt-step-204", "prompt_id": "dynamic", "system_prompt": "minimal_style_guardrails", "scores_in_prompt": "both_raw", "base_model": "gpt-4.1-mini"},
    "dpo-41m-min-e3-b8-m06-k08-v15":       {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:dpo-41m-min-e3-b8-m06-k08-v15:Cgrsh1mU", "prompt_id": "dynamic", "system_prompt": "minimal_style_guardrails", "scores_in_prompt": "both_raw", "base_model": "gpt-4.1-mini"},

    # Model 14: dpo-41m-raw-e3-b10-m04-k10-v20
    "dpo-41m-raw-e3-b10-m04-k10-v20:ckpt1": {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:dpo-41m-raw-e3-b10-m04-k10-v20:ChxhTbWz:ckpt-step-77", "prompt_id": "dynamic", "system_prompt": "none", "scores_in_prompt": "none", "base_model": "gpt-4.1-mini"},
    "dpo-41m-raw-e3-b10-m04-k10-v20:ckpt2": {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:dpo-41m-raw-e3-b10-m04-k10-v20:ChxhVl9G:ckpt-step-154", "prompt_id": "dynamic", "system_prompt": "none", "scores_in_prompt": "none", "base_model": "gpt-4.1-mini"},
    "dpo-41m-raw-e3-b10-m04-k10-v20":       {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:dpo-41m-raw-e3-b10-m04-k10-v20:ChxhVpDC", "prompt_id": "dynamic", "system_prompt": "none", "scores_in_prompt": "none", "base_model": "gpt-4.1-mini"},

    # ---- More gpt-4.1 based models ----
    # Model 15: dpo-41-min-e3-b8-m05-k08-v15
    "dpo-41-min-e3-b8-m05-k08-v15:ckpt1": {"provider": "openai_ft", "model": "ft:gpt-4.1-2025-04-14:litero-ai:dpo-41-min-e3-b8-m05-k08-v15:CgsysQGZ:ckpt-step-102", "prompt_id": "dynamic", "system_prompt": "minimal_style_guardrails", "scores_in_prompt": "both_raw", "base_model": "gpt-4.1"},
    "dpo-41-min-e3-b8-m05-k08-v15:ckpt2": {"provider": "openai_ft", "model": "ft:gpt-4.1-2025-04-14:litero-ai:dpo-41-min-e3-b8-m05-k08-v15:CgsytkYD:ckpt-step-204", "prompt_id": "dynamic", "system_prompt": "minimal_style_guardrails", "scores_in_prompt": "both_raw", "base_model": "gpt-4.1"},
    "dpo-41-min-e3-b8-m05-k08-v15":       {"provider": "openai_ft", "model": "ft:gpt-4.1-2025-04-14:litero-ai:dpo-41-min-e3-b8-m05-k08-v15:CgsytaXZ", "prompt_id": "dynamic", "system_prompt": "minimal_style_guardrails", "scores_in_prompt": "both_raw", "base_model": "gpt-4.1"},

    # Model 16: dpo-41-raw-e4-b10-m04-k10-v20
    "dpo-41-raw-e4-b10-m04-k10-v20:ckpt1": {"provider": "openai_ft", "model": "ft:gpt-4.1-2025-04-14:litero-ai:dpo-41-raw-e4-b10-m04-k10-v20:ChznIwO7:ckpt-step-154", "prompt_id": "dynamic", "system_prompt": "none", "scores_in_prompt": "none", "base_model": "gpt-4.1"},
    "dpo-41-raw-e4-b10-m04-k10-v20:ckpt2": {"provider": "openai_ft", "model": "ft:gpt-4.1-2025-04-14:litero-ai:dpo-41-raw-e4-b10-m04-k10-v20:ChznJ79G:ckpt-step-231", "prompt_id": "dynamic", "system_prompt": "none", "scores_in_prompt": "none", "base_model": "gpt-4.1"},
    "dpo-41-raw-e4-b10-m04-k10-v20":       {"provider": "openai_ft", "model": "ft:gpt-4.1-2025-04-14:litero-ai:dpo-41-raw-e4-b10-m04-k10-v20:ChznKpU7", "prompt_id": "dynamic", "system_prompt": "none", "scores_in_prompt": "none", "base_model": "gpt-4.1"},

    # ---- More gpt-4o based models ----
    # Model 17: dpo-4o-min-e3-b8-m05-k08-v15
    "dpo-4o-min-e3-b8-m05-k08-v15:ckpt1": {"provider": "openai_ft", "model": "ft:gpt-4o-2024-08-06:litero-ai:dpo-4o-min-e3-b8-m05-k08-v15:CguJJHO2:ckpt-step-102", "prompt_id": "dynamic", "system_prompt": "minimal_style_guardrails", "scores_in_prompt": "both_raw", "base_model": "gpt-4o"},
    "dpo-4o-min-e3-b8-m05-k08-v15:ckpt2": {"provider": "openai_ft", "model": "ft:gpt-4o-2024-08-06:litero-ai:dpo-4o-min-e3-b8-m05-k08-v15:CguJKMLE:ckpt-step-204", "prompt_id": "dynamic", "system_prompt": "minimal_style_guardrails", "scores_in_prompt": "both_raw", "base_model": "gpt-4o"},
    "dpo-4o-min-e3-b8-m05-k08-v15":       {"provider": "openai_ft", "model": "ft:gpt-4o-2024-08-06:litero-ai:dpo-4o-min-e3-b8-m05-k08-v15:CguJKBpN", "prompt_id": "dynamic", "system_prompt": "minimal_style_guardrails", "scores_in_prompt": "both_raw", "base_model": "gpt-4o"},

    # Model 18: dpo-4o-raw-e4-b6-m04-k10-v20
    "dpo-4o-raw-e4-b6-m04-k10-v20:ckpt1": {"provider": "openai_ft", "model": "ft:gpt-4o-2024-08-06:litero-ai:dpo-4o-raw-e4-b6-m04-k10-v20:Ci2zIszF:ckpt-step-254", "prompt_id": "dynamic", "system_prompt": "none", "scores_in_prompt": "none", "base_model": "gpt-4o"},
    "dpo-4o-raw-e4-b6-m04-k10-v20:ckpt2": {"provider": "openai_ft", "model": "ft:gpt-4o-2024-08-06:litero-ai:dpo-4o-raw-e4-b6-m04-k10-v20:Ci2zJe4I:ckpt-step-381", "prompt_id": "dynamic", "system_prompt": "none", "scores_in_prompt": "none", "base_model": "gpt-4o"},
    "dpo-4o-raw-e4-b6-m04-k10-v20":       {"provider": "openai_ft", "model": "ft:gpt-4o-2024-08-06:litero-ai:dpo-4o-raw-e4-b6-m04-k10-v20:Ci2zJyg6", "prompt_id": "dynamic", "system_prompt": "none", "scores_in_prompt": "none", "base_model": "gpt-4o"},

    # ---- More gpt-4.1-nano based models ----
    # Model 19: dpo-nano-min-e3-b8-m07-k05-v10
    "dpo-nano-min-e3-b8-m07-k05-v10:ckpt1": {"provider": "openai_ft", "model": "ft:gpt-4.1-nano-2025-04-14:litero-ai:dpo-nano-min-e3-b8-m07-k05-v10:ChwOSzLr:ckpt-step-108", "prompt_id": "dynamic", "system_prompt": "minimal_style_guardrails", "scores_in_prompt": "both_binned", "base_model": "gpt-4.1-nano"},
    "dpo-nano-min-e3-b8-m07-k05-v10:ckpt2": {"provider": "openai_ft", "model": "ft:gpt-4.1-nano-2025-04-14:litero-ai:dpo-nano-min-e3-b8-m07-k05-v10:ChwOTX7v:ckpt-step-216", "prompt_id": "dynamic", "system_prompt": "minimal_style_guardrails", "scores_in_prompt": "both_binned", "base_model": "gpt-4.1-nano"},
    "dpo-nano-min-e3-b8-m07-k05-v10":       {"provider": "openai_ft", "model": "ft:gpt-4.1-nano-2025-04-14:litero-ai:dpo-nano-min-e3-b8-m07-k05-v10:ChwOTCCo", "prompt_id": "dynamic", "system_prompt": "minimal_style_guardrails", "scores_in_prompt": "both_binned", "base_model": "gpt-4.1-nano"},

    # Model 20: dpo-nano-raw-e4-b10-m05-k08-v15
    "dpo-nano-raw-e4-b10-m05-k08-v15:ckpt1": {"provider": "openai_ft", "model": "ft:gpt-4.1-nano-2025-04-14:litero-ai:dpo-nano-raw-e4-b10-m05-k08-v15:ChxToVV6:ckpt-step-162", "prompt_id": "dynamic", "system_prompt": "none", "scores_in_prompt": "none", "base_model": "gpt-4.1-nano"},
    "dpo-nano-raw-e4-b10-m05-k08-v15:ckpt2": {"provider": "openai_ft", "model": "ft:gpt-4.1-nano-2025-04-14:litero-ai:dpo-nano-raw-e4-b10-m05-k08-v15:ChxTq0bm:ckpt-step-243", "prompt_id": "dynamic", "system_prompt": "none", "scores_in_prompt": "none", "base_model": "gpt-4.1-nano"},
    "dpo-nano-raw-e4-b10-m05-k08-v15":       {"provider": "openai_ft", "model": "ft:gpt-4.1-nano-2025-04-14:litero-ai:dpo-nano-raw-e4-b10-m05-k08-v15:ChxTq5T3", "prompt_id": "dynamic", "system_prompt": "none", "scores_in_prompt": "none", "base_model": "gpt-4.1-nano"},

    # ---- More gpt-4.1-mini based models ----
    # Model 21: dpo-41m-min-e3-b32-m08-k03-v10
    "dpo-41m-min-e3-b32-m08-k03-v10:ckpt1": {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:dpo-41m-min-e3-b32-m08-k03-v10:CgrK6Hsp:ckpt-step-27", "prompt_id": "dynamic", "system_prompt": "minimal_style_guardrails", "scores_in_prompt": "both_raw", "base_model": "gpt-4.1-mini"},
    "dpo-41m-min-e3-b32-m08-k03-v10:ckpt2": {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:dpo-41m-min-e3-b32-m08-k03-v10:CgrK7pEN:ckpt-step-54", "prompt_id": "dynamic", "system_prompt": "minimal_style_guardrails", "scores_in_prompt": "both_raw", "base_model": "gpt-4.1-mini"},
    "dpo-41m-min-e3-b32-m08-k03-v10":       {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:dpo-41m-min-e3-b32-m08-k03-v10:CgrK7CD2", "prompt_id": "dynamic", "system_prompt": "minimal_style_guardrails", "scores_in_prompt": "both_raw", "base_model": "gpt-4.1-mini"},

    # ---- More gpt-4.1 based models ----
    # Model 22: dpo-41-min-e3-b16-m08-k03-v10
    "dpo-41-min-e3-b16-m08-k03-v10:ckpt1": {"provider": "openai_ft", "model": "ft:gpt-4.1-2025-04-14:litero-ai:dpo-41-min-e3-b16-m08-k03-v10:Cgs1Ks22:ckpt-step-54", "prompt_id": "dynamic", "system_prompt": "minimal_style_guardrails", "scores_in_prompt": "both_raw", "base_model": "gpt-4.1"},
    "dpo-41-min-e3-b16-m08-k03-v10:ckpt2": {"provider": "openai_ft", "model": "ft:gpt-4.1-2025-04-14:litero-ai:dpo-41-min-e3-b16-m08-k03-v10:Cgs1LtiC:ckpt-step-108", "prompt_id": "dynamic", "system_prompt": "minimal_style_guardrails", "scores_in_prompt": "both_raw", "base_model": "gpt-4.1"},
    "dpo-41-min-e3-b16-m08-k03-v10":       {"provider": "openai_ft", "model": "ft:gpt-4.1-2025-04-14:litero-ai:dpo-41-min-e3-b16-m08-k03-v10:Cgs1LZbO", "prompt_id": "dynamic", "system_prompt": "minimal_style_guardrails", "scores_in_prompt": "both_raw", "base_model": "gpt-4.1"},

    # ---- More gpt-4o based models ----
    # Model 23: dpo-4o-min-e3-b16-m06-k03-v10
    "dpo-4o-min-e3-b16-m06-k03-v10:ckpt1": {"provider": "openai_ft", "model": "ft:gpt-4o-2024-08-06:litero-ai:dpo-4o-min-e3-b16-m06-k03-v10:ChxDLwJQ:ckpt-step-54", "prompt_id": "dynamic", "system_prompt": "minimal_style_guardrails", "scores_in_prompt": "ai_score_raw", "base_model": "gpt-4o"},
    "dpo-4o-min-e3-b16-m06-k03-v10:ckpt2": {"provider": "openai_ft", "model": "ft:gpt-4o-2024-08-06:litero-ai:dpo-4o-min-e3-b16-m06-k03-v10:ChxDMCXy:ckpt-step-108", "prompt_id": "dynamic", "system_prompt": "minimal_style_guardrails", "scores_in_prompt": "ai_score_raw", "base_model": "gpt-4o"},
    "dpo-4o-min-e3-b16-m06-k03-v10":       {"provider": "openai_ft", "model": "ft:gpt-4o-2024-08-06:litero-ai:dpo-4o-min-e3-b16-m06-k03-v10:ChxDN78Q", "prompt_id": "dynamic", "system_prompt": "minimal_style_guardrails", "scores_in_prompt": "ai_score_raw", "base_model": "gpt-4o"},

    # ---- More gpt-4.1-nano based models ----
    # Model 24: dpo-nano-cmp-e4-b32-m08-k03-v10
    "dpo-nano-cmp-e4-b32-m08-k03-v10:ckpt1": {"provider": "openai_ft", "model": "ft:gpt-4.1-nano-2025-04-14:litero-ai:dpo-nano-cmp-e4-b32-m08-k03-v10:Ci0TNVjM:ckpt-step-54", "prompt_id": "dynamic", "system_prompt": "compact_guidelines_rubric", "scores_in_prompt": "both_raw", "base_model": "gpt-4.1-nano"},
    "dpo-nano-cmp-e4-b32-m08-k03-v10:ckpt2": {"provider": "openai_ft", "model": "ft:gpt-4.1-nano-2025-04-14:litero-ai:dpo-nano-cmp-e4-b32-m08-k03-v10:Ci0TNLJi:ckpt-step-81", "prompt_id": "dynamic", "system_prompt": "compact_guidelines_rubric", "scores_in_prompt": "both_raw", "base_model": "gpt-4.1-nano"},
    "dpo-nano-cmp-e4-b32-m08-k03-v10":       {"provider": "openai_ft", "model": "ft:gpt-4.1-nano-2025-04-14:litero-ai:dpo-nano-cmp-e4-b32-m08-k03-v10:Ci0TOjeI", "prompt_id": "dynamic", "system_prompt": "compact_guidelines_rubric", "scores_in_prompt": "both_raw", "base_model": "gpt-4.1-nano"},

    # ---- DPO fine-tunes based on hum40-naive-auto ----
    # Model 25: dpo-h40-e2-b8-m08-b30-v10
    "dpo-h40-e2-b8-m08-b30-v10:ckpt1": {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:dpo-h40-e2-b8-m08-b30-v10:CjPwzEcd:ckpt-step-107", "prompt_id": "default", "base_model": "hum40-naive-auto"},
    "dpo-h40-e2-b8-m08-b30-v10":       {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:dpo-h40-e2-b8-m08-b30-v10:CjPwzMCd", "prompt_id": "default", "base_model": "hum40-naive-auto"},

    # Model 26: dpo-h40-e3-b8-m10-b25-v10
    "dpo-h40-e3-b8-m10-b25-v10:ckpt1": {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:dpo-h40-e3-b8-m10-b25-v10:CkVFxkZy:ckpt-step-107", "prompt_id": "default", "base_model": "hum40-naive-auto"},
    "dpo-h40-e3-b8-m10-b25-v10:ckpt2": {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:dpo-h40-e3-b8-m10-b25-v10:CkVFyBpS:ckpt-step-214", "prompt_id": "default", "base_model": "hum40-naive-auto"},
    "dpo-h40-e3-b8-m10-b25-v10":       {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:dpo-h40-e3-b8-m10-b25-v10:CkVFzeSc", "prompt_id": "default", "base_model": "hum40-naive-auto"},

    # Model 27: dpo-h40-e3-b8-m10-b15-v10
    "dpo-h40-e3-b8-m10-b15-v10:ckpt1": {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:dpo-h40-e3-b8-m10-b15-v10:CjSDbDVc:ckpt-step-107", "prompt_id": "default", "base_model": "hum40-naive-auto"},
    "dpo-h40-e3-b8-m10-b15-v10:ckpt2": {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:dpo-h40-e3-b8-m10-b15-v10:CjSDcxCD:ckpt-step-214", "prompt_id": "default", "base_model": "hum40-naive-auto"},
    "dpo-h40-e3-b8-m10-b15-v10":       {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:dpo-h40-e3-b8-m10-b15-v10:CjSDdZrK", "prompt_id": "default", "base_model": "hum40-naive-auto"},

    # Model 28: dpo-h40-e3-b12-m10-b25-v10
    "dpo-h40-e3-b12-m10-b25-v10:ckpt1": {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:dpo-h40-e3-b12-m10-b25-v10:CkUsJZDX:ckpt-step-71", "prompt_id": "default", "base_model": "hum40-naive-auto"},
    "dpo-h40-e3-b12-m10-b25-v10:ckpt2": {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:dpo-h40-e3-b12-m10-b25-v10:CkUsKXpn:ckpt-step-142", "prompt_id": "default", "base_model": "hum40-naive-auto"},
    "dpo-h40-e3-b12-m10-b25-v10":       {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:dpo-h40-e3-b12-m10-b25-v10:CkUsK7jy", "prompt_id": "default", "base_model": "hum40-naive-auto"},

    # Model 29: dpo-h40-e4-b12-m07-b35-v15
    "dpo-h40-e4-b12-m07-b35-v15:ckpt1": {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:dpo-h40-e4-b12-m07-b35-v15:CkV9YrUG:ckpt-step-134", "prompt_id": "default", "base_model": "hum40-naive-auto"},
    "dpo-h40-e4-b12-m07-b35-v15:ckpt2": {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:dpo-h40-e4-b12-m07-b35-v15:CkV9ZfOs:ckpt-step-201", "prompt_id": "default", "base_model": "hum40-naive-auto"},
    "dpo-h40-e4-b12-m07-b35-v15":       {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:dpo-h40-e4-b12-m07-b35-v15:CkV9ZSO1", "prompt_id": "default", "base_model": "hum40-naive-auto"},

    # Model 30: dpo-h40-e4-b16-m05-b40-v15
    "dpo-h40-e4-b16-m05-b40-v15:ckpt1": {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:dpo-h40-e4-b16-m05-b40-v15:CkVAxv7Y:ckpt-step-102", "prompt_id": "default", "base_model": "hum40-naive-auto"},
    "dpo-h40-e4-b16-m05-b40-v15:ckpt2": {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:dpo-h40-e4-b16-m05-b40-v15:CkVAxla0:ckpt-step-153", "prompt_id": "default", "base_model": "hum40-naive-auto"},
    "dpo-h40-e4-b16-m05-b40-v15":       {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:dpo-h40-e4-b16-m05-b40-v15:CkVAzLeV", "prompt_id": "default", "base_model": "hum40-naive-auto"},

    # Model 31: dpo-h40-e3-b8-m15-b20-v10
    "dpo-h40-e3-b8-m15-b20-v10:ckpt1": {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:dpo-h40-e3-b8-m15-b20-v10:CkViUXDQ:ckpt-step-107", "prompt_id": "default", "base_model": "hum40-naive-auto"},
    "dpo-h40-e3-b8-m15-b20-v10:ckpt2": {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:dpo-h40-e3-b8-m15-b20-v10:CkViW86T:ckpt-step-214", "prompt_id": "default", "base_model": "hum40-naive-auto"},
    "dpo-h40-e3-b8-m15-b20-v10":       {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:dpo-h40-e3-b8-m15-b20-v10:CkViWsWN", "prompt_id": "default", "base_model": "hum40-naive-auto"},

    # Model 32: dpo-h40-e5-b8-m10-b20-v20
    "dpo-h40-e5-b8-m10-b20-v20:ckpt1": {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:dpo-h40-e5-b8-m10-b20-v20:CkWZmXAM:ckpt-step-285", "prompt_id": "default", "base_model": "hum40-naive-auto"},
    "dpo-h40-e5-b8-m10-b20-v20:ckpt2": {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:dpo-h40-e5-b8-m10-b20-v20:CkWZnkbw:ckpt-step-380", "prompt_id": "default", "base_model": "hum40-naive-auto"},
    "dpo-h40-e5-b8-m10-b20-v20":       {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:dpo-h40-e5-b8-m10-b20-v20:CkWZnMoi", "prompt_id": "default", "base_model": "hum40-naive-auto"},

    # Model 33: dpo-h40-e3-b8-m10-bauto-v10
    "dpo-h40-e3-b8-m10-bauto-v10:ckpt1": {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:dpo-h40-e3-b8-m10-bauto-v10:CkW6CjVW:ckpt-step-107", "prompt_id": "default", "base_model": "hum40-naive-auto"},
    "dpo-h40-e3-b8-m10-bauto-v10:ckpt2": {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:dpo-h40-e3-b8-m10-bauto-v10:CkW6DKd8:ckpt-step-214", "prompt_id": "default", "base_model": "hum40-naive-auto"},
    "dpo-h40-e3-b8-m10-bauto-v10":       {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:dpo-h40-e3-b8-m10-bauto-v10:CkW6ENrR", "prompt_id": "default", "base_model": "hum40-naive-auto"},

    # Model 34: dpo-h40-e4-b12-m08-bauto-v15
    "dpo-h40-e4-b12-m08-bauto-v15:ckpt1": {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:dpo-h40-e4-b12-m08-bauto-v15:CkW8IZTh:ckpt-step-134", "prompt_id": "default", "base_model": "hum40-naive-auto"},
    "dpo-h40-e4-b12-m08-bauto-v15:ckpt2": {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:dpo-h40-e4-b12-m08-bauto-v15:CkW8JSWN:ckpt-step-201", "prompt_id": "default", "base_model": "hum40-naive-auto"},
    "dpo-h40-e4-b12-m08-bauto-v15":       {"provider": "openai_ft", "model": "ft:gpt-4.1-mini-2025-04-14:litero-ai:dpo-h40-e4-b12-m08-bauto-v15:CkW8JC1U", "prompt_id": "default", "base_model": "hum40-naive-auto"},

}

def get_model_info(model_id: str) -> dict:
    """
    Get model metadata from registry, or return a default config for custom models.
    
    If the model_id is found in MODEL_REGISTRY, returns that entry.
    Otherwise, assumes it's a custom OpenAI fine-tune and returns a default config.
    """
    if model_id in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_id]
    
    # Default config for custom/unknown models
    return {
        "provider": "openai_ft",  # Default to fine-tune provider
        "model": model_id,        # Use the ID as the model name
        "prompt_id": "default",   # Use default prompts
        "base_model": "custom"
    }
