"""GenAI layer: turn model output for one subject-session into a plain-English report.

If ANTHROPIC_API_KEY is set, the structured facts are sent to Claude to be written up.
Otherwise a deterministic template is used, so the app always works offline.
The LLM only *narrates* numbers computed here - it is never asked to compute them.
"""
import json
import os

import pandas as pd

MODEL = "claude-sonnet-5-5"


def session_facts(true_labels, pred_labels, confidence, low_conf=0.7, window_sec=2.56 * 0.5):
    """Aggregate window-level predictions into facts (each window = 2.56s, 50% overlap)."""
    df = pd.DataFrame({"true": list(true_labels), "pred": list(pred_labels), "conf": list(confidence)})
    n = len(df)
    share = (df.pred.value_counts(normalize=True) * 100).round(1).to_dict()
    return {
        "n_windows": n,
        "approx_minutes": round(n * window_sec / 60, 1),
        "time_share_pct": share,
        "accuracy_pct": round((df.true == df.pred).mean() * 100, 1),
        "low_confidence_windows": int((df.conf < low_conf).sum()),
        "low_confidence_pct": round((df.conf < low_conf).mean() * 100, 1),
        "mean_confidence": round(df.conf.mean(), 3),
        "most_common_error": (
            df[df.true != df.pred].groupby(["true", "pred"]).size().sort_values(ascending=False)
            .head(1).rename("n").reset_index().to_dict("records") or None),
    }


def _template(f):
    top = sorted(f["time_share_pct"].items(), key=lambda kv: -kv[1])
    parts = ", ".join(f"{k.lower().replace('_', ' ')} ({v}%)" for k, v in top[:3])
    txt = (f"Over about {f['approx_minutes']} minutes of monitoring, the subject's most common activities were "
           f"{parts}. The model's labels matched the recorded activity in {f['accuracy_pct']}% "
           f"of windows; {f['low_confidence_pct']}% of windows were low-confidence and should be "
           f"reviewed manually.")
    e = f["most_common_error"]
    if e:
        txt += (f" The most frequent mistake was labelling {e[0]['true'].lower()} as "
                f"{e[0]['pred'].lower()} ({e[0]['n']} windows), which is expected because those "
                f"postures produce very similar sensor signals.")
    return txt


def write_summary(facts: dict) -> tuple[str, str]:
    """Return (text, source) where source is 'claude' or 'template'."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return _template(facts), "template"
    try:
        import anthropic
        msg = anthropic.Anthropic(api_key=key).messages.create(
            model=MODEL, max_tokens=350,
            system=("You write brief, factual summaries of wearable-sensor activity monitoring "
                    "for a non-technical clinical-operations reader. Use ONLY the numbers given. "
                    "Do not diagnose or give medical advice. 3-4 sentences."),
            messages=[{"role": "user", "content": json.dumps(facts, default=str)}])
        return msg.content[0].text, "claude"
    except Exception as exc:  # network/key/library problems -> degrade gracefully
        return _template(facts) + f"  (LLM unavailable: {type(exc).__name__})", "template"
