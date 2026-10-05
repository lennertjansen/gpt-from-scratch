"""Render log124M_10B/log.txt into assets/training_curves.png. Needs seaborn: uv run --with seaborn plot_log.py"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

LOG = "log124M_10B/log.txt"
OUT = "assets/training_curves.png"
GPT2_VAL, GPT2_HELLA, GPT3_HELLA = 3.2924, 0.2955, 0.3367  # OpenAI 124M checkpoint (same evals), GPT-3 paper 125M

series = {"train": [], "val": [], "hella": []}
for line in open(LOG):
    step, kind, value = line.split()
    series[kind].append((int(step), float(value)))

sns.set_theme(style="whitegrid", context="notebook", font_scale=0.95)
c_train, c_val, _ = sns.color_palette("crest", 3)
fig, (ax_loss, ax_hella) = plt.subplots(1, 2, figsize=(12, 4))

ax_loss.plot(*zip(*series["train"]), color=c_train, lw=0.8, alpha=0.6, label="train (per step)")
ax_loss.plot(*zip(*series["val"]), color=c_val, lw=2.2, label="val (every 250 steps)")
ax_loss.axhline(GPT2_VAL, color="0.3", ls=":", lw=1.5, label=f"OpenAI GPT-2 124M checkpoint, val {GPT2_VAL:.2f}")
ax_loss.set_ylim(2.8, 4.5)
ax_loss.set_xlabel("step (1 step = 524,288 tokens)")
ax_loss.set_ylabel("cross-entropy loss")
ax_loss.legend(frameon=False)

ax_hella.plot(*zip(*series["hella"]), color=c_val, lw=2.2, label="this run")
ax_hella.axhline(GPT2_HELLA, color="0.3", ls=":", lw=1.5, label=f"OpenAI GPT-2 124M, {GPT2_HELLA:.2%}")
ax_hella.axhline(GPT3_HELLA, color="0.3", ls="--", lw=1.5, label=f"GPT-3 125M (paper), {GPT3_HELLA:.2%}")
ax_hella.set_xlabel("step")
ax_hella.set_ylabel("HellaSwag accuracy")
ax_hella.legend(frameon=False, loc="lower right")

for ax in (ax_loss, ax_hella):
    sns.despine(ax=ax, left=True, bottom=True)
fig.tight_layout()
fig.savefig(OUT, dpi=130)
print("wrote", OUT)
