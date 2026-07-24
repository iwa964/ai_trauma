# Self-report instrument sources

`data/instruments.json` was transcribed **verbatim** from the official source PDFs
listed below (item wording is exact; only typographic glyphs were normalised to
ASCII — see `ascii_normalization` in the JSON). This file records where each
instrument came from and its licensing status.

> **Verify before publishing or redistributing.** The licensing summaries below
> reflect what the source documents state plus the commonly-documented status of
> each instrument. Confirm the current terms of use with the copyright holder
> before distributing the instruments or any derivative outside this project.

## PCL-5 — PTSD Checklist for DSM-5

- **Source file:** `pcl5assessment.pdf` (branded *NovoPsych*; 20-item DSM-5 PCL-5, 2 pages).
- **Instrument citation (from the PDF):** Weathers, F.W., Litz, B.T., Keane, T.M.,
  Palmieri, P.A., Marx, B.P., & Schnurr, P.P. (2013). *The PTSD Checklist for
  DSM-5 (PCL-5).* Scale available from the National Center for PTSD at
  www.ptsd.va.gov.
- **Origin / distributor:** Developed by the U.S. National Center for PTSD
  (Department of Veterans Affairs); this particular form was exported from
  NovoPsych.
- **Licensing status:** The National Center for PTSD provides the PCL-5 free of
  charge, and as a work of the U.S. federal government it is generally treated as
  **public domain**. No purchase or per-use royalty. (The NovoPsych branding is on
  the form layout, not the instrument text.)
- **Response scale:** Not at all (0) · A little bit (1) · Moderately (2) · Quite a
  bit (3) · Extremely (4).
- **DSM-5 cluster mapping applied** (per the published criterion structure):
  B (intrusion) = items 1–5 · C (avoidance) = 6–7 · D (negative alterations in
  cognition and mood) = 8–14 · E (arousal and reactivity) = 15–20.

## PHQ-9 — Patient Health Questionnaire

- **Source file:** `phq9.pdf` (Pfizer form with scoring/interpretation pages).
- **Instrument citation (from the PDF):** Kroenke, K., Spitzer, R.L., Williams, J.B.
  (2001). *The PHQ-9: Validity of a brief depression severity measure.* J Gen
  Intern Med. 16(9), 606–613. Adapted from PRIME-MD TODAY (Spitzer, Williams,
  Kroenke, and colleagues), developed with an educational grant from Pfizer Inc.
- **Licensing status (from the PDF):** "Copyright © 1999 Pfizer Inc. All rights
  reserved. … Use of the PHQ-9 may only be made in accordance with the Terms of
  Use available at www.pfizer.com." Pfizer has made the PHQ family of screeners
  available for use **without a license fee or individual permission**; confirm the
  current Terms of Use before redistribution.
- **Response scale:** Not at all (0) · Several days (1) · More than half the days
  (2) · Nearly every day (3). No clusters (`null`).

## GAD-7 — Generalized Anxiety Disorder scale

- **Source file:** `gad7.pdf` (Pfizer form with scoring page).
- **Instrument citation (from the PDF):** Spitzer, R.L., Kroenke, K., Williams,
  J.B.W., Löwe, B. (2006). *A brief measure for assessing generalized anxiety
  disorder: The GAD-7.* Archives of Internal Medicine, 166(10), 1092–1097.
- **Licensing status (from the PDF):** "Copyright © 1999 Pfizer Inc. All rights
  reserved. Reproduced with permission." Developed as part of PRIME-MD (Spitzer,
  Williams, Kroenke, and colleagues). As with the PHQ-9, the GAD-7 is broadly
  available for use without an individual license; confirm current terms before
  redistribution.
- **Response scale:** Not at all (0) · Several days (1) · More than half the days
  (2) · Nearly every day (3). No clusters (`null`).

## Transcription notes / discrepancies flagged

- **PCL-5 time anchor** in this source form reads **"IN THE LAST MONTH"** (not "in
  the past month"). The verbatim phrase is kept in `pcl5.time_anchor_original`; the
  loader rewrites it to "since that happened".
- **GAD-7** uses **"Over the last two weeks"** (word "two"); **PHQ-9** uses **"Over
  the last 2 weeks"** (digit "2"). Both are recorded verbatim per instrument.
- **PCL-5 item 15** uses British spelling **"behaviour"** — transcribed as-is.
- **PHQ-9 item 8** contains a mid-item question mark ("…could have noticed? Or the
  opposite…") — transcribed as-is.
- **ASCII normalisation:** curly quotes → straight quotes (PCL-5 item 17,
  "superalert"); en-dash → " - " (PHQ-9 items 6 and 8). Words are verbatim.

## Modifications applied on top of the verbatim text

1. **Time anchor** rewritten to `"since that happened"` at load time (originals
   preserved in `time_anchor_original`); byte-identical across all conditions
   including no-injection.
2. **PCL-5 cluster tags** B/C/D/E added per the DSM-5 criterion mapping; PHQ-9 and
   GAD-7 items tagged `null`.
3. **`reverse_scored_items`** left **empty** — none of the three instruments has a
   natively reverse-scored item, so acquiescence is uncontrolled. Reverse-worded
   items / neutral fillers must be authored by the clinician; they are **not**
   auto-generated here.
