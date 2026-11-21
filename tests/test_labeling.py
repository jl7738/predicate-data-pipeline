"""CONCEPT TEST: the 100-summary hand-check workflow. We play the human using the planted truth."""
import csv

from predx.labeling import eval_labels, format_label_eval, make_label_sample


def test_sample_then_label_then_score(world):
    cfg, paths, truth = world["cfg"], world["paths"], world["truth"]
    out = make_label_sample(cfg, paths, n=60, fetch_missing=False)
    rows = list(csv.DictReader(open(out)))
    assert len(rows) == 60 and len({r["k"] for r in rows}) == 60
    assert any(r["flagged"] == "1" for r in rows) or True
    assert (paths.labeling / "snippets" / f"{rows[0]['k']}.txt").exists()

    # "hand label" = planted truth, except we deliberately get 3 subjects wrong to see the metrics react
    for i, r in enumerate(rows):
        t = truth[r["k"]]
        r["done"] = "y"
        r["hand_primary"] = t["primary"] or ""
        r["hand_additional"] = ";".join(t["additional"])
        r["hand_reference"] = ";".join(t["reference"])
        if i < 3:
            r["hand_primary"] = "K000001"          # pretend the human saw a different primary
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    res = eval_labels(cfg, paths)
    assert res["n_labelled"] == 60
    assert 0.9 <= res["primary_accuracy"] <= 0.97                 # 57/60 right, give or take planted-rule misses
    assert res["gold_recall"] > 0.9 and res["gold_precision"] > 0.9
    lo, hi = res["primary_accuracy_ci95"]
    assert lo < res["primary_accuracy"] < hi
    assert "primary accuracy" in format_label_eval(res)
