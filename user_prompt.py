USER_PROMP_NONE_SCORES = "" #must be empty presented for reference only


USER_PROMP_SCORES = f"""
This text's AI level is currently classified as **{initial_paragraph_score}**.
'''
{ai_text}
'''
{if both_scores "Rewrite the text more natural to lower AI score to  0.01." else "Rewrite it more natural so the text sounds human-like."}
"""
# initial_paragraph_score - % that paragraph got from GPTZero before humanization
# ai_text - initial non-humanized paragraph
# both_scores - bool variable that defines if we need to add faked score/level in prompt


USER_PROMP_SCORES_BINNED = f"""
This text AI score is  {initial_paragraph_level}.
'''
{ai_text}
'''
{if both_scores "Rewrite it so the level becomes **Low**." else "Rewrite it more natural so the text sounds human-like."}
"""
# initial_paragraph_level - level that paragraph got from GPTZero before humanization (Low, Medium, High) where Low is 0-0.1, Medium is 0.1-0.4, High is 0.4+
# ai_text - initial non-humanized paragraph
# both_scores - bool variable that defines if we need to add faked score/level in prompt