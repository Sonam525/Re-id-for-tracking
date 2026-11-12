# Notice

Licence holder: Haiyu Yang.

## Code

Code in this repository is under the MIT licence in [LICENSE](LICENSE).

## Third-party material

Nothing below is copied into this repository. Check each upstream licence yourself.

- **SAM 3.** This work uses SAM 3 by Meta. It is gated and you download it under the SAM License. Trade control limits in that licence apply.
- **Calf student.** The student was distilled from SAM 3. Its weights are published with [FarmSAM](https://github.com/Sonam525/FarmSAM) and at [Calf-Identity-Tracking-Weights](https://huggingface.co/Sonam5/Calf-Identity-Tracking-Weights) under the SAM License. The non-commercial limit in FarmSAM applies only to its Edinburgh student, which is not used here.
- **DINOv3.** Embeddings were computed with DINOv3 ViT-S/16, subject to the DINOv3 License. No weights or embeddings ship here.
- **RF-DETR.** The library is Apache-2.0. The trained calf detector is published at [Calf-Identity-Tracking-Weights](https://huggingface.co/Sonam5/Calf-Identity-Tracking-Weights) under the SAM License, because its training labels are masks from SAM 3.

If you use this work in a publication, please acknowledge SAM 3 and DINOv3 as their licences ask.

## Masks

The masks were made with SAM 3 and a student distilled from it, so we release them under the [SAM License](LICENSES/SAM_LICENSE.txt) to be safe. Anyone who passes them on must do so under those terms and include a copy of the licence. MIT does not apply to them.

## Data shown

The clip and stills in `docs/media/` come from the calf dataset, camera FarmA_R1 (doi:10.57967/hf/10912, CC0), with overlays added. The overlays are masks, so the SAM License applies to them. It does not change the terms of the video.
