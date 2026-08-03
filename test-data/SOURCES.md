# Test media sources

## Video clips

Both from [Pexels](https://www.pexels.com), under the
[Pexels license](https://www.pexels.com/license/): free for personal and commercial use,
no attribution required, with modification permitted. Credits listed anyway. Downloaded
at 1920x1080 and used unmodified.

| File | Pexels ID | Length | What it exercises |
|---|---|---|---|
| `desk-worker.mp4` | 7581176 | 9.9 s | Someone asleep at a desk. The attention-lost path, and the case that exposed the dwell-timer hysteresis bug |
| `face-closeup.mp4` | 4170238 | 9.1 s | A close, strongly turned head. Large head yaw, so it stresses the head-pose fusion |

## Still images

| File | Source | Licence |
|---|---|---|
| `portrait-*.jpg` | Pexels, IDs in the filename | Pexels license |
| `face_640x480.jpg` | Qualcomm AI Hub, the `face_det_lite` demo asset | BSD-3-Clause |
| `eye_reference.png` | Qualcomm AI Hub, the `eyegaze` demo asset | BSD-3-Clause |

## Dataset

**MPIIGaze** is used for the accuracy section only. It is not redistributed here; the
scripts download it.

> Zhang, Sugano, Fritz and Bulling. *Appearance-Based Gaze Estimation in the Wild.*
> CVPR 2015. https://www.mpi-inf.mpg.de/MPIIGaze

Download: `https://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIGaze.tar.gz` (2.1 GB).
Only `Data/Normalized/` and `Evaluation Subset/` are needed, and both appear early in the
archive, so a partial download can be extracted and used before the rest arrives.
