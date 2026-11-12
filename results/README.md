# Results

| File | What it holds |
|:---|:---|
| [`identity_by_day.csv`](identity_by_day.csv) | Identity on the three days with numeric ground truth, in calf-minutes. The 04-23 late stratum (23:30 to 24:00) is reported separately |
| [`night_gaps.csv`](night_gaps.csv) | The six night gaps and how each was checked |
| [`mask_refinement.csv`](mask_refinement.csv) | Fragmented-mask time before and after refinement, in slot-hours |
| [`backbone.csv`](backbone.csv) | DINOv3 backbone size against re-identification accuracy |

A calf-minute is one calf for one minute. "Trusted" time leaves out a short margin around each identity change in the reference notes. "Silent" errors are wrong labels the system did not flag. Identity scores were computed on the masks before refinement.
