# Data Description

The dataset contains 12,000 ICU stays by adult patients, admitted to cardiac,
medical, surgical and trauma ICUs for a wide variety of reasons. Stays shorter
than 48 hours were excluded.

Each record covers the first 48 hours after ICU admission. Observations are timestamped by how long after admission they were taken, and stop at the 48-hour however long the patient actually stayed.

## Files

| File | Contents |
| --- | --- |
| `icu_records/` | 12,000 per-patient files, `<RecordID>.csv` |
| `outcomes.csv` | One row per patient, keyed by `RecordID` |

## Record format

Each record is a CSV with three columns: `Time,Parameter,Value`.

- `Time` — `HH:MM` elapsed since ICU admission, from `00:00` to `47:59`.
- `Parameter` — the variable name.
- `Value` — the measurement. **`-1` indicates a missing value.**

The first rows of every file hold the general descriptors, all stamped `00:00`,
followed by the time-varying observations:

```
Time,Parameter,Value
00:00,RecordID,105306
00:00,Age,85
00:00,Gender,1
00:00,Height,-1
00:00,ICUType,3
00:00,Weight,70.4
00:07,GCS,15
00:07,HR,73
00:37,Urine,175
...
```

Not every variable appears in every record, and sampling is irregular: some
variables are recorded many times per day, others once or never.

### General descriptors

| Variable | Meaning |
| --- | --- |
| `RecordID` | Unique integer identifying the ICU stay |
| `Age` | Years, ranging from 15 to 90. For de-identification, every patient aged 90 or older is recorded as `90`. |
| `Gender` | 0: female, 1: male |
| `Height` | cm |
| `ICUType` | 1: Coronary Care Unit, 2: Cardiac Surgery Recovery Unit, 3: Medical ICU, 4: Surgical ICU |
| `Weight` | kg — appears here at `00:00`, and may also recur later as a time-varying measurement |

### Time-varying variables

| Variable | Meaning | Units |
| --- | --- | --- |
| `Albumin` | — | g/dL |
| `ALP` | Alkaline phosphatase | IU/L |
| `ALT` | Alanine transaminase | IU/L |
| `AST` | Aspartate transaminase | IU/L |
| `Bilirubin` | — | mg/dL |
| `BUN` | Blood urea nitrogen | mg/dL |
| `Cholesterol` | — | mg/dL |
| `Creatinine` | Serum creatinine | mg/dL |
| `DiasABP` | Invasive diastolic arterial blood pressure | mmHg |
| `FiO2` | Fractional inspired O₂ | 0–1 |
| `GCS` | Glasgow Coma Score | 3–15 |
| `Glucose` | Serum glucose | mg/dL |
| `HCO3` | Serum bicarbonate | mmol/L |
| `HCT` | Hematocrit | % |
| `HR` | Heart rate | bpm |
| `K` | Serum potassium | mEq/L |
| `Lactate` | — | mmol/L |
| `MAP` | Invasive mean arterial blood pressure | mmHg |
| `MechVent` | Mechanical ventilation respiration | 0: false, 1: true |
| `Mg` | Serum magnesium | mmol/L |
| `Na` | Serum sodium | mEq/L |
| `NIDiasABP` | Non-invasive diastolic arterial blood pressure | mmHg |
| `NIMAP` | Non-invasive mean arterial blood pressure | mmHg |
| `NISysABP` | Non-invasive systolic arterial blood pressure | mmHg |
| `PaCO2` | Partial pressure of arterial CO₂ | mmHg |
| `PaO2` | Partial pressure of arterial O₂ | mmHg |
| `pH` | Arterial pH | 0–14 |
| `Platelets` | — | cells/nL |
| `RespRate` | Respiration rate | bpm |
| `SaO2` | O₂ saturation in hemoglobin | % |
| `SysABP` | Invasive systolic arterial blood pressure | mmHg |
| `Temp` | Temperature | °C |
| `TroponinI` | Troponin-I | μg/L |
| `TroponinT` | Troponin-T | μg/L |
| `Urine` | Urine output | mL |
| `WBC` | White blood cell count | cells/nL |

The `NI*` variables are non-invasive blood pressure measurements; `DiasABP`,
`MAP` and `SysABP` are the corresponding invasive (arterial line) measurements.

## `outcomes.csv`

| Column | Meaning |
| --- | --- |
| `RecordID` | Unique integer identifying the ICU stay |
| `SAPS-I` | Simplified Acute Physiology Score |
| `SOFA` | Sequential Organ Failure Assessment score |
| `Length_of_stay` | Days between admission to the ICU and the end of hospitalization, including any time spent in hospital after discharge from the ICU |
| `Survival` | Days between admission to the ICU and death; `-1` otherwise |
| `In-hospital_death` | 0: survivor, 1: died before discharge from hospital |
