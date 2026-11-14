# Models

Weights are too large for git, so they live in the team Google Drive. Download them into this folder.

| File | What | Download |
|---|---|---|
| `TrackNet_best.pt` | TrackNetV3 shuttle detector, fine-tuned on our first-person and stationary clips (45 MB) | [Google Drive](https://drive.google.com/file/d/1_F6jCVLt1p3JbjHTBUTEJGOyacMuX8Fq/view?usp=sharing) |

`tracknet_node` loads `models/TrackNet_best.pt` by default; set its `model_path` parameter to use another file.
