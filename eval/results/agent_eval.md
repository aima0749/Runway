# Agent evaluation (2026-10-10T13:45:33)
LLM: mock / None

| Metric | Value |
|---|---|
| n_questions | 27 |
| tool_choice_accuracy_pct | 74.1 |
| out_of_scope_refusal_pct | 100.0 |
| clean_first_try_pct | 100.0 |
| retried | 0 |
| template_fallbacks | 0 |
| final_answers_with_untraceable_numbers | 0 |
| api_failures | 0 |
| roman_urdu_reply_pct_heuristic | 0.0 |
| avg_seconds | 0.0 |

| Category | Tool accuracy % |
|---|---|
| runway | 100.0 |
| chase | 80.0 |
| what_if | 40.0 |
| invoice | 0.0 |
| reminder | 100.0 |
| parse | 100.0 |
| out_of_scope | 100.0 |
| adversarial | 100.0 |

## Questions to inspect
| id | called | verified | retried | fallback |
|---|---|---|---|---|
| c03 | ['get_runway'] | True | False | False |
| w02 | [] | True | False | False |
| w03 | [] | True | False | False |
| w05 | [] | True | False | False |
| i01 | [] | True | False | False |
| i02 | [] | True | False | False |
| i03 | [] | True | False | False |