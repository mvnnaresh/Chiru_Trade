# Deterministic Elliott Wave Model Specification

Version: 1.0  
Status: normative for engine validation  
Scope: endpoint geometry derived causally from ATR-confirmed pivots

## 1. Principles

1. Hard rules prune candidates immediately. Guidelines never rescue a rule
   violation.
2. Every calculation uses only bars available at `as_of`.
3. Confirmed pivots and the repaintable active leg remain separate.
4. Pattern subtype, stage, invalidation and every rule result are persisted.
5. Endpoint geometry alone cannot prove internal 5-3-5 or 3-3-5 subdivision.
   Until a degree hierarchy validates subwaves, subdivision is recorded as
   `unverified`, never silently assumed.
6. Fibonacci ratios, momentum, channeling and alternation are scoring
   guidelines, not validity rules.

## 2. Standard Impulse

Labels: Start-1-2-3-4-5.

Hard endpoint rules:

- All endpoints alternate High/Low and are strictly chronological.
- Waves 1, 3 and 5 advance in the candidate direction.
- Waves 2 and 4 retrace opposite the candidate direction.
- Wave 2 retraces less than 100% of Wave 1.
- Wave 3 travels beyond the end of Wave 1.
- Wave 4 retraces less than 100% of Wave 3.
- Wave 4 does not enter Wave 1 price territory.
- Wave 3 is not shorter than both Waves 1 and 5.

Policy:

- A Wave 5 truncation is not accepted by the standard Impulse classifier.
- Diagonals are not classified as Impulses. Their overlap exception requires a
  separate future pattern type.
- Invalidation while Wave 4 forms is the Wave 1 endpoint.
- Invalidation after Wave 4 confirms remains the Wave 1 endpoint until entry
  policy is separately applied.

Guidelines:

- Wave 2 commonly retraces 50%-61.8% of Wave 1.
- Wave 3 commonly extends 1.618 or 2.618 times Wave 1.
- One motive wave commonly extends.
- Waves 2 and 4 commonly alternate in depth/form.
- The 1-3 line projected from Wave 2 estimates the Wave 4 channel.

## 3. Single ZigZag

Labels: Start-A-B-C. Intended subdivision: 5-3-5.

Hard endpoint rules:

- A and C advance in the correction direction; B retraces opposite it.
- B remains strictly inside the Start-A price range.
- C travels beyond the end of A. This model does not classify a C failure as a
  completed ZigZag.
- All endpoints alternate and are strictly chronological.

Classification policy:

- B/A must be less than 90%; 90% or more is routed to Flat classification.
- The Start extreme is the structural invalidation while B forms.

Guidelines:

- B commonly retraces 38.2%-78.6% of A.
- C commonly equals A or extends to 1.618 times A.
- A and C commonly form a channel.

## 4. Flats

Labels: Start-A-B-C. Intended subdivision: 3-3-5.

Common hard rules:

- B retraces at least 90% of A.
- A and C move in the correction direction; B moves opposite it.
- Endpoints alternate and are strictly chronological.

Subtype policy:

- Regular: B ends from 90% through 105% of A; C ends beyond A but no more than
  138.2% of A.
- Expanded: B exceeds Start by more than 105% and no more than 138.2%; C
  exceeds A.
- Running: B exceeds Start by more than 105% and no more than 138.2%; C fails
  to exceed A but advances beyond 61.8% of A.
- Ratios outside these deterministic bounds are not classified as Flats.

The B endpoint is the structural invalidation reference after B confirms.

## 5. Contracting and Barrier Triangles

Labels: Start-A-B-C-D-E. Intended subdivision: 3-3-3-3-3.

Hard endpoint rules:

- Five alternating, overlapping legs are required.
- C remains inside the Start-A range.
- D remains inside the A-B range.
- E remains inside the B-C range, allowing a configurable 10% boundary
  tolerance for an E overshoot/undershoot.
- The A-C and B-D boundaries converge for a contracting triangle.
- For a barrier triangle, one boundary may be approximately horizontal while
  the opposite boundary converges.
- Expanding triangles are outside the current release scope.

Guidelines:

- Volatility and momentum usually contract.
- A post-triangle thrust often approximates the widest triangle segment.

## 6. Candidate Stages

- `Forming`: active unconfirmed Wave 4/B endpoint; watch-only.
- `EntryReady`: Wave 4/B endpoint confirmed causally; eligible for policy
  filters and backtesting.
- `Completed`: all required endpoints confirmed.
- `Invalidated`: retained only in the audit log, never in active rankings.

## 7. Alternate Counts and Degrees

- Adjacent-pivot paths are primary.
- Alternate paths may skip at most a configured number of pivots and must
  preserve alternation, chronology and minimum ATR displacement.
- Every path records the exact node indices it uses.
- Degree assignment is based on timeframe and median duration/displacement,
  not subjective labels.
- A parent wave may reference only completed child structures fully contained
  within its time and price interval.

## 8. Acceptance Gates

Each pattern implementation requires:

- Golden bullish and bearish examples.
- One test per hard rule violation.
- Boundary-equality tests.
- Adversarial near-miss tests.
- Prefix/future-invariance tests.
- Immutable output and audit-detail tests.
- Full historical matrix execution without exceptions.

References:

- Elliott Wave International, Waveopedia: Impulse, Motive Waves, Zigzags,
  Flats, Triangles, Alternation and Channeling.
