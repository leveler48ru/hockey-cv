# hockey-cv

Computer-vision toolkit for analyzing amateur hockey game footage from a
single fixed (pan-only) broadcast camera. Everything here was built to
answer practical questions about a full match recording: where is the puck,
who has it, what is the score and game clock, when did the whistle blow,
and how does the play map onto a real 60x30 m rink.

## What it does

- **Rink segmentation & calibration** — YOLOv8-seg models (trained here)
  for rink markings, players, goaltenders, referees and the puck; homography
  estimation from rink anchors (`gcalib*.py`, `gpu_calib*.py`, `solve_h*.py`,
  `ice_quad_calib.py`, `lines_fit.py`, `calib_*.py`).
- **Panorama & mosaics** — full-ice 60x30 m mosaic built by chaining ORB
  matches between neighboring frames (`pano.py`, `mosaic*.py`, `full_run*.py`).
- **Camera pan profile** — pan angle over time recovered from the video
  (`pan_m1.py`, `pan_profile.py`, `pan_measure.py`, `verify_pan.py`).
- **Whole-match tracking** — ByteTrack tracking of puck/players/referees,
  dumped as JSONL (`track_m1.py`, `tracker.yaml`), then converted into world
  meters with speeds, faceoffs and shot detection (`world_m1.py`).
- **Scoreboard OCR** — reading score + game clock from the scoreboard plate
  by seven-segment digit segmentation, no ML (`sb_*.py`).
- **Audio events** — referee whistle and goal siren detection from the audio
  track; small CNN classifiers trained on labeled clips (`whistle.py`,
  `siren.py`, `audio_*.py`).
- **Team assignment** — per-frame jersey color clustering (hue k-means)
  into red/blue teams (`team_color.py`, `finetune_teams.py`).
- **Referee pipeline** — dataset builders, a labeling helper, contact-sheet
  QC (`build_ref_*.py`, `detect_referees*.py`, `referee_labeler/`,
  `check_referee_markup.py`, `scan_referees.py`).
- **Synthetic data** — generator of fake YOLO-seg rink frames to experiment
  with geometry without real footage (`gen_dataset.py`).
- **Event fusion** — motion + audio + scoreboard events merged into a single
  timeline (`events_fuse.py`, `events_motion.py`).
- **Demos & QC figures** — rendered demo clips and check overlays
  (`make_demo_*.py`, `draw_ref*.py`, `ref_check.py`, `check_boxes.py`).
- **Training launches** — `train_v*.sh` run YOLOv8-seg fine-tuning inside
  docker with GPU; `run_v4.sh` shows uploading and launching training on a
  remote GPU box via SSH.

## Layout

```
*.py / *.sh        pipeline scripts (flat, named by purpose)
annotate/          tiny web labeler for rink anchors
geo_calib/         geometric calibration service (IIHF rink model)
referee_labeler/   labeling helper for referee crops
tracker.yaml       ByteTrack config
train_v*.sh        YOLOv8-seg training launches (docker + GPU)
```

## Typical pipeline order (one match)

1. Extract audio and sample frames with `ffmpeg` (`sample_wide*.py`,
   `extract_*.py`).
2. Label rink anchors and estimate calibration (`annotate/`, `gcalib*.py`,
   `gpu_calib*.py`, `solve_h*.py`).
3. Recover the camera pan profile (`pan_m1.py`).
4. Track the whole match (`track_m1.py`).
5. Convert tracks to world meters, detect events (`world_m1.py`).
6. Read the scoreboard for ground-truth score/clock (`sb_scan.py` ->
   `sb_score.py`, `sb_score2.py`).
7. Detect whistles/sirens from audio (`whistle.py`, `siren.py`).
8. Fuse everything into one timeline (`events_fuse.py`).

## Requirements

- Python 3.10+
- `numpy`, `opencv-python`, `scipy`, `matplotlib`, `scikit-learn`,
  `torch`, `ultralytics` (YOLOv8)
- `ffmpeg` for audio/frame extraction
- A CUDA GPU for training (`train_v*.sh` expect docker +
  `ultralytics/ultralytics` image); CPU inference works for most scripts.

Model weights, datasets, source videos and intermediate caches are **not**
included: scripts either train the models (`train_v*.sh`, `audio_train*.py`)
or expect file paths passed as arguments.

## Notes

- Scripts are intentionally independent and hackable: each one is a step of
  an evolving experiment (model generations v1...v8, quick fixes, some dead
  ends kept for reference).
- World coordinate system: IIHF rink 60x30 m, `x` along the rink, origin at
  a goal line corner.
- The assumed input is 1080p footage from a fixed camera that only pans
  horizontally (typical amateur broadcast).

## License

MIT — see [LICENSE](LICENSE).
