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

}
