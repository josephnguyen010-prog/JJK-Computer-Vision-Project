# Jujutsu Kaisen hand sign recognition

Throw a JJK hand sign at your webcam and it names the character. Cursed energy
gathers at your fingertips, a ring closes, and the domain expands.

MediaPipe finds the hand landmarks; a small MLP trained on my own recordings
classifies the sign; a charge-up state machine decides when to actually fire.
It runs two ways: a Python desktop app, and a browser build (in `web/`) that
runs the same model in plain JavaScript, so no video ever leaves the tab.

**[Try it in the browser →](https://josephnguyen010-prog.github.io/Joseph-Nguyen-Portfolio/jjk/)** (needs a webcam)

## Setup

```
python -m venv .venv
.venv\Scripts\Activate.ps1          # or: source .venv/bin/activate
pip install -r requirements.txt
```

The MediaPipe hand landmarker model downloads itself on first run.

## The vocabulary

Five signs plus idle. Keys are what you press in `record.py`.

| Key | Sign | Shape |
|---|---|---|
| `0` | idle | anything that isn't a sign |
| `1` | Megumi | two closed fists, held apart |
| `2` | Sukuna | open palm thrust at the camera, other hand a raised fist |
| `3` | Yuta | flat palm forward, fingers together, fist low at the chest |
| `4` | Gojo | fingers of one hand crossed through the other (MediaPipe sees one hand) |
| `5` | Malevolent Shrine | hands interlocked closely, upright at the chest |

Names, record keys and the how-to text shown on the web page all live in
[jjk/signs.py](jjk/signs.py).

**Deferred, not dropped.** The face-referenced sign (hand over the eye, finger at
the cheek) can't work on hand landmarks alone: absolute position is normalised
away on purpose, so a palm over your eye and a palm anywhere else are the same
vector. It needs face landmarks added first.

## The workflow, in order

### 1. Find out which signs are viable — `view_signs.py`

**Do this before anything else.** It answers the only question that can sink the
project: can MediaPipe actually see these signs?

```
python view_signs.py            # two-handed signs
python view_signs.py --hands 1  # one-handed
```

Hold each sign steady for about a second and read the verdict. Two things are
being measured:

- **detection rate** — how often both hands are found at all
- **jitter** — how much the landmarks twitch while you hold still

Interlocked fingers make MediaPipe guess at hidden fingertips, and its guess
changes every frame. That jitter is what a sign failing looks like.

Sort your signs into GOOD / MARGINAL / POOR and **start with the GOOD ones
only.** Get the whole pipeline working end to end on four or five easy signs
before going anywhere near Malevolent Shrine. Press `r` to reset between signs.

### 2. Record training data — `record.py`

```
python record.py
```

Press a sign's number key, get a 3-second countdown to set your hands, then it
captures 150 frames. Press `u` to undo a bad burst, `q` to save and quit.
Recording appends, so you can stop and come back.

Three rules, and they matter more than anything in the model:

- **Move while recording.** Drift nearer and further, rotate your wrists, tilt,
  shift side to side. A burst held perfectly still teaches one exact pose.
- **Record idle (key `0`) more than anything else.** Hands relaxed, reaching for
  the keyboard, scratching your face, halfway between two signs. Every frame
  that isn't a sign has to look like idle, or the demo fires at random.
- **Several short bursts, not one long one.** Get up, sit back down, change the
  lighting, record again. Variation *between* bursts is what makes it robust —
  and it's what the evaluation in step 3 measures.

Aim for 300+ samples and 3+ separate bursts per sign; roughly double that for idle.

### 3. Train — `train.py`

```
python train.py
```

Trains in seconds, and prints two accuracy numbers:

```
random split (leaks near-duplicate frames):  99.8%  <- ignore this
held-out bursts (what the demo will feel like): 91.2%  <- trust this
```

Frames recorded back to back are near-identical. A normal random train/test
split scatters copies of the same instant across both sides and reports a number
close to 100% no matter how bad the model is. The second number holds out whole
recording sessions, so it measures what you actually care about: does this work
when you sit down tomorrow. Both are printed because the gap is the point.

`train.py` refuses to run on a dataset with obvious problems (too few samples,
too few bursts, missing or undersized idle class). Pass `--force` to override.

If accuracy is low, read the confusion matrix before touching the model. Two
signs trading off against each other means they look alike to MediaPipe, and
more data for that specific pair beats any hyperparameter.

### 4. Run it — `detect.py`

```
python detect.py
python detect.py --debug    # live probabilities, charge, hand count
```

Hold a sign; energy gathers and pulls toward your palm as the charge builds; at
full charge the domain fires. Press `d` to toggle the debug readout, `q` to quit.

## The charge-up is the debounce

The classifier emits a prediction every frame, so a sign held for two seconds is
sixty activations. Three things prevent that, and all three are visible on screen:

| Mechanism | What it does | Where |
|---|---|---|
| Smoothing | Averages probabilities over recent frames so one bad frame can't fire | `SMOOTHING` |
| Charge time | The sign must hold for ~1s of real time | `CHARGE_SECONDS` |
| Lockout | Nothing else fires until you return to idle | `SignDetector.locked` |

The charge time is a debounce that happens to look like powering up. Tuning
knobs are at the top of `detect.py`.

## Layout

```
view_signs.py      sign viability triage, run this first
record.py          dataset collection
train.py           training + honest evaluation
detect.py          live detection, charge-up, domain expansion
identify.py        photographs each sign next to the class it is read as
rename_signs.py    relabel or swap classes in the dataset, then retrain
export_model.py    exports the model to web/model.json for the browser
build_web.py       assembles web/ + assets/ into one deployable folder
jjk/features.py    landmark normalisation (the heart of it)
jjk/tracker.py     camera + MediaPipe, the only file touching the MP API
jjk/effects.py     particles, glow, domain expansion hit
jjk/signs.py       the sign vocabulary, edit this to add or rename signs
web/               the browser build, a JavaScript port of the above
tests/             feature invariance, layout, portraits, the charge state machine
```

Run the Python tests with `python -m pytest tests/` (no webcam needed). The
browser side has its own checks:

```
node web/verify.mjs          # JS classifier matches scikit-learn's output
node web/detector.test.mjs   # JS charge state machine
```

## Browser build

```
python export_model.py       # after retraining
python build_web.py          # -> dist/, ready to host as static files
```

The model is a scaler and a two-layer MLP, small enough to run as three matrix
multiplies in plain JavaScript, so the page needs no ML runtime.

## Screen layout

By default the window is split down the middle — **camera on the left, character
on the right**:

```
python detect.py                    # split screen (default)
python detect.py --layout overlay   # character beside your hands instead
```

Each panel is the size of the camera frame, so the webcam half is pixel-exact
with no letterboxing, and the whole canvas is scaled down at the end to fit on a
normal screen. The character panel stays empty until a sign is recognised.

## Character portraits

Drop an image into `assets/characters/` named after the sign — `malevolent_shrine.gif`,
`gojo.png` — and it appears beside your hands when that sign is recognised.
Nothing else needs editing.

- **Animated GIF and WebP** play on loop at their own frame timing. **PNG/JPG** work too.
- Transparency is preserved in all formats.
- Scaled automatically to 260px tall; portrait-shaped art looks best.
- The portrait fades in as recognition firms up, and flips to the other side of
  your hands when you're near the edge of frame.
- Signs with no image just don't show one — a half-filled folder is fine.

`python make_placeholder_portraits.py` writes stand-in cards for every sign so
the display works before you have art. Overwrite them at the same filenames.

## Adding or changing signs

Everything reads from `SIGNS` in `jjk/signs.py`. Add an entry, re-record, retrain.
The `occlusion` field is your own triage note from step 1; nothing enforces it.

## Notes

- `jjk/features.py` places **both hands in one shared coordinate frame** rather
  than normalising each hand separately. Several of these signs differ mainly in
  how the hands sit relative to each other, and per-hand normalisation would
  erase exactly that.
- **Hands are slotted by screen position, not by MediaPipe's Left/Right label.**
  On symmetric poses — two fists especially — that label flips between frames,
  and label-keyed slots would swap both halves of the feature vector and read as
  a completely different sign. Screen order is decided by the pixels, so it can't
  disagree with itself. The label still rides along as a soft feature.
- The preview is mirrored so it behaves like a mirror. Recording and detection
  both go through `FrameSource`, so the convention stays consistent — which is
  all the classifier needs.
- If the webcam won't open, close anything else using it, or try
  `--camera 1`.
