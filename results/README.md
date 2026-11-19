# Results

| File | What it holds |
|:---|:---|
| [`identity_by_day.csv`](identity_by_day.csv) | Identity on the days with a minute-level reference, in calf-minutes: the validation days 04-22 and 04-23, and the test day 04-24 against the interval notes as written and with two change points corrected. The 04-23 late stratum (23:30 to 24:00) is reported separately |
| [`night_gaps.csv`](night_gaps.csv) | The first labels after each of the six overnight recording stops, and how each was checked |
| [`ablation.csv`](ablation.csv) | What each part of the identity layer contributes, on the validation days |
| [`backbone.csv`](backbone.csv) | DINOv3 backbone size against identification accuracy, on the validation days |
| [`mask_refinement.csv`](mask_refinement.csv) | Fragmented-mask time before and after refinement, in track-hours |

A calf-minute is one calf for one minute. "Trusted" time is calf-time with a reference identity; it leaves out a short margin around each identity change in the reference. Accuracy of labelled time is correct ÷ (correct + wrong); "?" and time without a mask are left out. "Silent" errors are wrong labels the system did not flag. `ablation.csv` says which tracks each comparison ran on and whether its analysis output was kept; the paper's supplement gives each comparison in full. Identity scores were computed on the masks before refinement. Track-hours in `mask_refinement.csv` are the paper's slot-hours: one tracker slot for one hour.
