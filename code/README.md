# Code

The pipeline as it ran, in run order. The scripts were written as Databricks notebooks (source format) and read video and write results on a Databricks volume, so change the paths at the top of each script for your own setup. Placeholders such as `<DATABRICKS_USER>` mark personal paths.

Keep this folder flat: scripts import each other by file name. The file names, and the `Tol` label inside the scripts, are the working name of camera `FarmA_R1`. The mask files cite these names, so they stay as they ran.

| Step | What it does | Scripts |
|:---|:---|:---|
| 1. Decode | Decodes each clip to 5 fps frames | `Tol_00e_decode_frames.py` |
| 2. Detector | Builds calf mask labels from the two training days and trains RF-DETR-Seg | `Tol_41_derive_healthy_band.py`, `Tol_42_export_seg_coco.py`, `Tol_43_train_rfdetr_seg.py`, `Tol_44_detector_eval.py` |
| 3. Tracker | SAM 3 with the distilled student, re-seeded by the detector. Writes anonymous segments with one mask per frame | `Tol_47_track_v5.py`, `tol_v5_core.py`, `Tol_03_track_v4.py` (shared base), `Tol_47b_launcher.py` |
| 4. Crops and embeddings | Masked crops every 15 s, embedded with DINOv3 ViT-S/16 | `Tol_32_dense_crops.py`, `Tol_28_build_bands.py` |
| 5. Anchors | A frozen pool of verified crops from the training days, and the map from training calves to the test week | `Tol_50_build_anchor_pool.py`, `Tol_51_anchor_correspondence.py` |
| 6. Identity | Assigns segments to calves in 3-minute windows, with "?" when unsure | `Tol_30_periodic_identity.py`, `prod_relabel_v5.py` |
| 7. Colour filter | Hides a segment whose crops are clearly coloured, since all four calves are black and white | `Tol_54_crop_colour.py` |
| 8. Mask refinement | Repairs fragmented masks with the detector. Identity does not change | `Tol_55_mask_refine.py`, `tol_refine_core.py` |
| 9. Videos | Overlay videos at 50× speed and side-by-side comparisons | `Tol_17_overlay_video.py`, `Tol_18_compress_videos.py`, `Tol_57_side_by_side.py` |

The tracker's student backbone is trained with [FarmSAM](https://github.com/Sonam525/FarmSAM), and its calf weights are on [Hugging Face](https://huggingface.co/Sonam5/FarmSAM). SAM 3 needs access to the gated [`facebook/sam3`](https://huggingface.co/facebook/sam3) repository.

The calf student and the RF-DETR calf detector are at [Calf-Identity-Tracking-Weights](https://huggingface.co/Sonam5/Calf-Identity-Tracking-Weights). Not included: the anchor pool and frame caches.
