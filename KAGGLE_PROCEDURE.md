# Kaggle operating procedure — Tiny DCGAN / cGAN / CycleGAN on full VGGFace2

This document covers the *operational* steps around the notebook — things you do in the
Kaggle UI, not in code. Read this once before your first run.

## 1. Create the notebook

1. Go to kaggle.com → Code → New Notebook.
2. Upload `VGGFace2_GAN_Assignment.ipynb` (File → Import Notebook), or copy/paste cells in.
3. **Settings (right sidebar) → Accelerator → GPU T4 x2** (or P100 if available). Tiny
   DCGAN doesn't need multi-GPU, but T4 x2 is Kaggle's default GPU offering and works fine
   with single-GPU code (the second GPU just sits idle, which is fine for this notebook).
4. **Settings → Persistence → "Files only"** so `/kaggle/working/` (your checkpoints and
   samples) survives between sessions. Without this, you lose your model on every restart.

## 2. Attach VGGFace2

1. **Add Input** (right sidebar) → search "VGGFace2".
2. Kaggle has multiple VGGFace2 mirrors with different sizes — check the file count/size
   before attaching, since "full VGGFace2" (3.31M images, ~36GB) is much larger than the
   common "VGGFace2 Test" split (~170k images). Pick the one that matches your actual
   compute budget; full VGGFace2 across many sessions is realistic, but confirm Kaggle's
   input-disk limit (typically 20GB per dataset attach, sometimes more for popular
   mirrors) can hold it before committing.
3. Once attached, run `!ls /kaggle/input/` in a cell to get the **exact** folder name —
   it varies by which mirror you pick (e.g. `vggface2`, `vggface2-train`, a username
   prefix, etc.).
4. Update `DATA_ROOT` in Section 1 of the notebook to match.

## 3. Attach a painting dataset (for CycleGAN, Task 3)

1. **Add Input** → search "wikiart" or "best artworks of all time" (Kaggle has a few
   painting-image datasets under those names).
2. Same process: `!ls /kaggle/input/` to confirm the path, update `PAINTING_ROOT`.

## 4. GPU quota management — the part that actually matters at this scale

Kaggle gives **30 GPU-hours per week** on the free tier, and a **single session caps at
~9 hours** (12 hours wall-clock includes some idle/setup overhead — treat 9h as your
working budget). Full VGGFace2 is large enough that you will not finish meaningful
training in one session. Plan around this:

- **One epoch over all 3.31M images** at `BATCH_SIZE=128` is ~25,900 steps. At a rough
  ~3-5 steps/sec on a T4 for this tiny model and 64×64 resolution, that's roughly
  1.5-2.5 hours **per epoch** — so you can likely fit 3-5 epochs in one 9h session if
  nothing else is competing for the GPU.
- **Don't try to do all of DCGAN + cGAN + CycleGAN in one sitting.** Budget separate
  sessions: e.g. session 1-2 for DCGAN convergence, session 3 for the attribute
  classifier + pseudo-labeling pass (this is a full dataset inference pass, also slow),
  session 4-5 for cGAN, session 6+ for CycleGAN.
- **Checkpointing is what makes this survivable.** The notebook saves to
  `/kaggle/working/checkpoints/*.pt` every 500 steps and at every epoch boundary, and
  automatically resumes from the latest checkpoint when you re-run the training cell.
  Don't delete `/kaggle/working/` between sessions.
- **Quota resets weekly.** If you burn through 30 hours, you wait for the reset or use a
  secondary account / Colab as a supplement — Anthropic doesn't have visibility into your
  Kaggle quota balance, check it under your Kaggle account settings.

## 5. Monitoring a long run

- Kaggle notebooks **disconnect from the browser but keep running** if you close the tab,
  as long as you started it via **Save & Run All (Commit)** rather than just running
  cells interactively in edit mode. For multi-hour VGGFace2 training, always commit
  rather than relying on an interactive session staying open.
- Check progress via the **Logs** panel on the commit's run page — the notebook prints
  step/loss/timing every 100 steps (`LOG_EVERY`), so you can sanity-check it's making
  progress without re-opening the full notebook.
- Sample image grids save to `/kaggle/working/samples/` every checkpoint — download these
  periodically to track visual progress, since `/kaggle/working` is wiped when you start
  a fresh kernel session that's not a true resume.

## 6. Building the report from notebook outputs

The "Results and Discussion" section of your report should pull directly from:
- `samples/dcgan_epoch*.png` — generation quality over training time (use several epochs'
  worth side by side to show progression)
- `samples/cgan_*` grids split by hair-length condition
- CycleGAN before/after face→painting pairs
- The loss-curve plots if you log losses to a list and plot them (add a simple
  `loss_history.append((step, loss_d.item(), loss_g.item()))` inside the training loop
  if you want this — not included by default to keep the loop lean)
- The gradient-magnitude plot from Section 6 (Task 4 math) as your Figure for the
  minimax-vs-BCE comparison

## 7. Common failure modes specific to this setup

| Symptom | Likely cause | Fix |
|---|---|---|
| `FileNotFoundError` on first cell | `DATA_ROOT` doesn't match attached dataset's actual path | Re-run `!ls /kaggle/input/` and correct the path |
| Index-building step takes forever | Walking 3.3M files for the first time | Expected — it's one-time and cached to `file_index.json`; just let it finish once |
| OOM (CUDA out of memory) | `BATCH_SIZE` too high for T4's 16GB at higher resolutions | Lower `BATCH_SIZE` or keep `IMG_SIZE=64` |
| D loss → 0, G loss climbing | Discriminator overpowering generator (common on diverse, large datasets) | Reduce D capacity (`D_FEAT`), or update G more often than D |
| Session ends mid-epoch, "resume" doesn't pick up cleanly | Forgot to commit (Save & Run All) instead of running interactively | Always use Commit for long runs, not just "Run" in edit mode |
| cGAN samples don't show clear long/short hair difference | Pseudo-labels noisy (classifier undertrained) | Train `TinyAttrClassifier` longer / on more labeled CelebA examples before pseudo-labeling VGGFace2 |
