# Synthetic generator assumptions

All generated records are fictional and intended for Runway Radar scenarios.
These values are scenario assumptions, not Pakistani statistics or estimates.

`generate_business(seed, n_clients, months, mode, start)` uses NumPy's
`default_rng`, so a fixed argument set produces the same three DataFrames.
The tables use the repository schema columns exactly; dates are ISO strings.

## Clients and payment behavior

- Segments are assigned with equal probability among `corporate`, `sme`, and
  `individual`.
- Segment baseline delays are 17, 8.5, and 3.5 days respectively. Each
  client's baseline is multiplied by a lognormal draw with parameters 0 and
  0.3.
- Each client has lognormal delay noise with sigma uniform from 0.3 to 0.6,
  and a 3% to 15% chance per invoice of an extra 20 to 50 days.
- Twenty percent of clients pay at the end of the payment month. Payments
  whose calculated date falls earlier are moved to that month's final day.
- Invoice size sensitivity is uniform from 0 to 0.25 and scales delay by
  `exp(k * log(amount_pkr / 50000))`.
- Forty percent of clients change behavior at a uniformly selected date in
  the generated period. Easy mode multiplies the baseline by a factor from
  0.8 to 1.25. Hard mode uses an abrupt factor of 0.5 or 2.0.
- Easy mode uses the client's lognormal delay distribution and late tail. Hard
  mode instead draws half of payments near on-time and half around three times
  the client's typical delay, with 20% relative spread.
- Each client-month has a Poisson(1.5) invoice count. Issue dates are uniform
  over the month, amounts are lognormal with parameters 10.8 and 1.0, and terms
  are selected equally from 15, 30, and 45 days.
- Due date is issue date plus terms. The true paid date is due date plus
  generated delay and is clipped to at least one day after issue date.

## Expenses and censoring

- Monthly expense totals are 65% of the generated monthly invoice amount.
  Rent receives 20% of that total, salaries 55%, and suppliers 25%, due on
  days 1, 20, and 25 respectively. All expense rows have recurrence `monthly`.
- `censor_at` drops invoices issued after its cutoff, retains the original
  value as `true_paid_date`, and replaces observable paid dates after the
  cutoff with missing values.
- The module entry point writes easy and hard CSVs plus settings JSON using
  the configured project seed. Opening cash is set to two average months of
  generated expenses and the minimum buffer to one average month.
