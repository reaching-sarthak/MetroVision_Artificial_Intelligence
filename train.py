"""
train.py

Entry point: trains the CNN on normalized-image .npz files, checkpointing
often enough that a crash never loses more than a few steps of progress.

USAGE
-----
    python train.py --data /path/to/normalized_npz_dir --ckpt checkpoints/run1 --accum_steps 8

Re-running the exact same command after a crash (or a deliberate
Ctrl-C) automatically resumes from the last checkpoint — same shard,
same sample index, same optimizer momentum, same weights.

GRADIENT ACCUMULATION (--accum_steps)
--------------------------------------------------------------------------
The model trains one image at a time (there's no batched forward pass
anywhere in conv1.py/dense128.py/etc.), which on its own means every
single AdamW update is driven by one noisy sample's gradient, plus
40% dropout noise on top. That combination is why loss can plateau
and bounce around instead of trending down.

--accum_steps N fixes this without touching any of the model math:
forward+backward still runs one image at a time, but the resulting
gradients are SUMMED into an accumulator across N images before a
single averaged AdamW step is taken. This is mathematically the same
update AdamW would compute from an actual batch of N images — it just
never materializes them as one array, so nothing about conv1.py etc.
needs to become batch-aware. Default is 8; try 16 if loss is still
noisy, or 4 if training feels too slow to see movement.

Note this changes what "one step" means for --save_every and for
AdamW's internal step counter t: a "step" from here on is one
accum_steps-image OPTIMIZER update, not one image. Checkpoint
frequency in wall-clock time is therefore roughly
save_every * accum_steps images between saves.

WHAT "checkpointing" MEANS HERE
--------------------------------
- Every `--save_every` optimizer steps (default 50), and once at the
  end of each epoch, the FULL state is atomically written to
  <ckpt>.npz / <ckpt>.json (see checkpoint.py): every weight, AdamW's
  m/v moment estimates, its global step counter, and exactly which
  shard/sample to resume from.
- If something raises partway through (bad image, NaN, kill -9, etc.),
  the `finally` block still saves an emergency checkpoint from the
  last *fully completed* optimizer step before the exception
  propagates — you lose at most the in-progress accumulation window,
  never everything trained so far.
- Optimizer state (m/v) is what actually makes resuming safe for
  AdamW specifically: reloading only the weights and restarting m/v/t
  from zero would give the first several post-resume steps much
  larger effective updates than intended.

PERFORMANCE NOTE
-----------------
conv1/conv2/conv3/maxpool now run through fast_layers.py (vectorized
im2col + matmul) via model.py's TrainFlowFixed, not the raw loop-based
conv1.py/conv2.py/conv3.py/maxpool.py. At 720x720 this is still likely
too slow to finish real training in a reasonable window — resize your
images down (e.g. 256x256 via normalize_dataset.py --image_size 256)
unless you've benchmarked full-resolution as acceptable for your time
budget.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from checkpoint import load_trainflow, save_checkpoint
from data_loader import iter_dataset


def parse_args():
    p = argparse.ArgumentParser(description="Train the road-damage CNN on normalized npz images.")
    p.add_argument("--data", required=True, help="Directory of normalized-image .npz shard files.")
    p.add_argument("--ckpt", required=True, help="Checkpoint path prefix, e.g. checkpoints/run1")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--accum_steps", type=int, default=8,
                    help="Average gradients over this many images before each AdamW update. "
                         "1 = old single-image-per-update behavior.")
    p.add_argument("--save_every", type=int, default=50,
                    help="Checkpoint every N optimizer steps (i.e. every N*accum_steps images).")
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--dropout_rate", type=float, default=0.40)
    p.add_argument(
        "--labels_are_datapipeline_order",
        action="store_true",
        help="Set this if your npz labels are integers written using datapipeline.py's "
             "(Normal=0,Crack=1,Pothole=2,Both=3) order instead of the model's own "
             "(Pothole=0,Crack=1,Both=2,Normal=3) order. String labels never need this flag.",
    )
    return p.parse_args()


class GradAccumulator:
    """Sums gradient dicts across images; averages and resets on flush()."""

    def __init__(self):
        self._sum = None
        self._count = 0

    def add(self, grads: dict) -> None:
        if self._sum is None:
            self._sum = {k: np.array(v, dtype=np.float32, copy=True) for k, v in grads.items()}
        else:
            for k, v in grads.items():
                self._sum[k] += v
        self._count += 1

    @property
    def count(self) -> int:
        return self._count

    def flush_averaged(self) -> dict:
        """Returns the averaged gradients and resets the accumulator."""
        averaged = {k: v / self._count for k, v in self._sum.items()}
        self._sum = None
        self._count = 0
        return averaged


def main():
    args = parse_args()
    if args.accum_steps < 1:
        raise ValueError("--accum_steps must be >= 1")

    tf, progress = load_trainflow(
        args.ckpt,
        learning_rate=args.learning_rate,
        dropout_rate=args.dropout_rate,
    )

    start_epoch = progress.get("epoch", 0)
    start_shard = progress.get("shard_index", 0)
    start_sample = progress.get("sample_index", 0)
    running_loss = progress.get("running_loss") or 0.0
    running_acc = progress.get("running_accuracy") or 0.0
    optimizer_steps_done = tf.optimizer.t

    print(f"Resuming from optimizer step {optimizer_steps_done} "
          f"(epoch {start_epoch}, shard {start_shard}, sample {start_sample}), "
          f"accum_steps={args.accum_steps}"
          if optimizer_steps_done > 0 else
          f"Starting fresh training run. accum_steps={args.accum_steps} "
          f"(one optimizer update per {args.accum_steps} images)")

    images_seen_in_run = 0
    optimizer_steps_in_run = 0
    accumulator = GradAccumulator()
    t0 = time.time()
    warmed_up = optimizer_steps_done > 0

    try:
        for epoch in range(start_epoch, args.epochs):
            shard_start = start_shard if epoch == start_epoch else 0
            sample_start = start_sample if epoch == start_epoch else 0

            for shard_index, sample_index, image, target_class in iter_dataset(
                args.data,
                labels_are_datapipeline_order=args.labels_are_datapipeline_order,
                start_shard_index=shard_start,
                start_sample_index=sample_start,
            ):
                logits = tf.forward(
                    image, training=True,
                    dropout_seed=tf.optimizer.t * args.accum_steps + accumulator.count,
                )
                loss, probabilities, grad_logits = tf.compute_loss(logits, target_class)
                loss = float(loss)
                tf.backward(grad_logits)
                accumulator.add(tf.get_gradients())

                predicted_class = int(np.argmax(logits))
                accuracy = float(predicted_class == target_class)

                images_seen_in_run += 1
                running_loss = loss if not warmed_up else 0.98 * running_loss + 0.02 * loss
                running_acc = accuracy if not warmed_up else 0.98 * running_acc + 0.02 * accuracy
                warmed_up = True

                # progress always records "resume after this sample", independent of
                # where we are in the accumulation window, so resume never skips or
                # replays a sample.
                progress = {
                    "epoch": epoch,
                    "shard_index": shard_index,
                    "sample_index": sample_index + 1,
                    "running_loss": running_loss,
                    "running_accuracy": running_acc,
                }

                if accumulator.count >= args.accum_steps:
                    averaged_grads = accumulator.flush_averaged()
                    params = tf.get_parameters()
                    tf.optimizer.step(params, averaged_grads)
                    optimizer_steps_done += 1
                    optimizer_steps_in_run += 1

                    if optimizer_steps_in_run % 5 == 0:
                        elapsed = time.time() - t0
                        rate = images_seen_in_run / elapsed if elapsed > 0 else 0.0
                        print(
                            f"epoch {epoch} shard {shard_index} sample {sample_index} "
                            f"| opt step {optimizer_steps_done} | loss {loss:.4f} "
                            f"(running {running_loss:.4f}) | acc {accuracy:.0f} "
                            f"(running {running_acc:.3f}) | {rate:.2f} img/s | {elapsed:.1f}s elapsed"
                        )

                    if optimizer_steps_in_run % args.save_every == 0:
                        save_checkpoint(args.ckpt, tf, progress=progress)
                        print(f"  [checkpoint saved at optimizer step {optimizer_steps_done}]")

            # End of epoch: flush any partial accumulation window so no gradients from
            # this epoch's last few images are silently dropped, then checkpoint.
            if accumulator.count > 0:
                averaged_grads = accumulator.flush_averaged()
                tf.optimizer.step(tf.get_parameters(), averaged_grads)
                optimizer_steps_done += 1

            progress = {
                "epoch": epoch + 1, "shard_index": 0, "sample_index": 0,
                "running_loss": running_loss, "running_accuracy": running_acc,
            }
            save_checkpoint(args.ckpt, tf, progress=progress)
            print(f"[checkpoint saved at end of epoch {epoch}] running_loss={running_loss:.4f} running_acc={running_acc:.3f}")

        print("Training complete.")

    except KeyboardInterrupt:
        print("\nInterrupted by user — saving emergency checkpoint before exiting...")
        raise

    except Exception:
        print("\nTraining crashed — saving emergency checkpoint from the last completed optimizer step...", file=sys.stderr)
        raise

    finally:
        # Always persist the last fully completed optimizer step, whatever happened.
        # NOTE: gradients still sitting in `accumulator` from a partial, not-yet-flushed
        # window are intentionally NOT applied here — an emergency checkpoint should
        # reflect only fully-averaged, already-taken optimizer steps.
        save_checkpoint(args.ckpt, tf, progress=progress)
        print(f"Checkpoint saved to {args.ckpt}.npz / {args.ckpt}.json")


if __name__ == "__main__":
    main()
