from raghash.demo import run_demo


def test_demo_runs_end_to_end_and_writes_the_report(tmp_path):
    lines = []
    res = run_demo(tmp_path / "d", seed=42, echo=lines.append)
    for k in ("duplicates", "versions", "contradictions"):
        assert res["scores"][k]["precision"] == 1.0 and res["scores"][k]["recall"] == 1.0
    assert res["actions"] == 75 and res["failures"] == 132 and abs(res["accuracy"] - 0.7832) < 0.001
    assert (tmp_path / "d" / "report.html").read_text(encoding="utf-8").startswith("<!doctype html>")
    assert (tmp_path / "d" / "testset.json").exists() and (tmp_path / "d" / "answers.json").exists()
    assert sum("[" in x and "/5]" in x for x in lines) == 5
