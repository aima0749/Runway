# Runway Radar model notes

## What the model predicts

The model estimates an invoice's duration **T**, in days from `issue_date` to `paid_date`. It uses a log-normal accelerated failure time model: the logarithm of duration is modeled as a linear combination of invoice and client-history features plus noise. A prediction can be expressed as a distribution of possible payment durations, not just one date.

Features include invoice amount and terms, due-day timing, client segment, and the client's prior observed payment delays. History is deliberately represented as missing when no eligible payment history exists; the fitted model fills those missing values with means learned from its training rows.

## Why unpaid invoices stay in training

At an `as_of` date, an unpaid invoice has not yet revealed its eventual duration. It is **right-censored**: its observed duration is `T = as_of - issue_date`, and its event flag is `E = 0`. A paid invoice has `T = paid_date - issue_date` and `E = 1`. Keeping censored invoices lets the likelihood account for the fact that payment has not happened yet. Dropping them would preferentially remove slow payments and bias predictions toward faster payment.

## Leakage rule

For a given invoice, history features may use only invoices from the same client whose `paid_date` is on or before the current invoice's `issue_date`. The invoice itself is excluded. Dates after the evaluation cutoff are censored before features and training targets are built. In particular, `true_paid_date` is not read by `features.py` or `delay_model.py`; the evaluator uses it only after prediction to score invoices that were still open at the cutoff.

## Conditional sampling for invoices still open

For an invoice that is still unpaid at age `s_obs`, samples must represent payment durations conditional on surviving unpaid through that age. If `mu` is the model's predicted mean log-duration and `sigma` its fitted log-normal scale, its survival probability at age `s` is

`S0 = norm.sf((ln(s) - mu) / sigma)`,

where `s = max(s_obs, 0.5)`. The sampler draws `u ~ Uniform(0, 1)`, sets

`q = 1 - u * S0`,

clips `q` away from zero and one for numerical stability, and returns `exp(mu + sigma * norm.ppf(q))`. Thus the draws are conditional on the invoice having remained unpaid up to observation, rather than unconditional durations that could fall before the observed age. Returned durations are capped at 365 days and enforced to be strictly greater than `s_obs`.

## What the backtest compares

The backtest uses chronological cutoffs, never randomized invoice splits. It fits using data observable at an earlier cutoff and evaluates invoices issued after that training cutoff that remain open at the later `as_of` date. It compares median model duration predictions with a due-date prediction, the client's average historical delay, and a client-history bootstrap baseline. It reports absolute error in days and model P10–P90 coverage, split between cold-start invoices (`n_prior < 3`) and the rest.

When a working cash simulator is available, the optional shortfall evaluation sets opening cash so the due-date projection barely stays positive, then compares model, due-date, client-average, and historical base-rate probabilities against the realized outcome. This is a designed scenario test, not a general accuracy claim. The current repository simulator may be a stub, in which case shortfall evaluation is skipped.

## Known limits

All generated data are synthetic and fictional. The Ayesha Studio dataset is a scripted scenario with hand-set client histories, invoice amounts, and expenses. These values are scenario assumptions, not Pakistani statistics and not evidence about real businesses in Pakistan. Synthetic backtest results measure behavior on the generator's assumptions; they do not establish real-world predictive performance. The log-normal form, feature set, censoring assumptions, amount effects, and observation timing may not represent actual payment processes.
