"""Engagement analyzer tests on synthetic classrooms.

synth_person() draws a plausible 13-keypoint upper body (COCO order) for a
seated student with a given head yaw, head-down amount, torso rotation and
hand pose, so the rules can be checked against known ground truth.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hcpose.engagement import DISTRACTED, FOCUSED, UNKNOWN, EngagementAnalyzer, load_settings  # noqa: E402
from hcpose.engagement.features import KeypointIndex, compute  # noqa: E402
from hcpose.presets import UPPER_BODY13_NAMES  # noqa: E402
from hcpose.results import FrameResult, JsonlWriter, Person  # noqa: E402

W, H = 1280, 720


def synth_person(cx, cy, sw, yaw=0.0, down=0.0, body=0.0, hand=None, face=True, jitter=0.0, rng=None):
    """yaw: feature units (-1..1, negative = nose toward image-left). down: 0..1. body: 0..1 torso rotation."""
    rng = rng or np.random.default_rng(0)
    kp = np.zeros((13, 3), np.float32)
    s = sw * (1 - 0.6 * body)
    hx, hy = cx, cy - 0.9 * sw + down * 0.7 * sw
    ear = 0.35 * sw
    kp[0] = [hx + yaw * ear, hy, 0.95]  # nose
    kp[1] = [hx + yaw * 0.25 * sw + 0.12 * sw, hy - 0.1 * sw, 0.9]  # left eye (image right)
    kp[2] = [hx + yaw * 0.25 * sw - 0.12 * sw, hy - 0.1 * sw, 0.9]
    kp[3] = [hx + ear, hy - 0.05 * sw, 0.0 if yaw > 0.8 else 0.8]  # left ear hidden when turned left
    kp[4] = [hx - ear, hy - 0.05 * sw, 0.0 if yaw < -0.8 else 0.8]
    kp[5] = [cx + s / 2, cy, 0.95]  # left shoulder (image right)
    kp[6] = [cx - s / 2, cy, 0.95]
    kp[7] = [cx + 0.55 * sw, cy + 0.5 * sw, 0.8]
    kp[8] = [cx - 0.55 * sw, cy + 0.5 * sw, 0.8]
    kp[9] = [cx + 0.3 * sw, cy + 0.8 * sw, 0.7]
    kp[10] = [cx - 0.3 * sw, cy + 0.8 * sw, 0.7]
    kp[11] = [cx + 0.35 * sw, cy + 1.1 * sw, 0.3]
    kp[12] = [cx - 0.35 * sw, cy + 1.1 * sw, 0.3]
    if hand == "raised":
        kp[9] = [cx + 0.5 * sw, hy - 0.8 * sw, 0.9]
    elif hand == "face":
        kp[9] = [hx + 0.1 * sw, hy + 0.15 * sw, 0.9]
    if not face:
        kp[0:3, 2] = 0.05
    if jitter:
        kp[:, :2] += rng.normal(0, jitter * sw, (13, 2))
    box = [cx - 0.8 * sw, cy - 1.45 * sw, cx + 0.8 * sw, cy + 1.2 * sw]
    return Person(None, 0.9, 0, "person", box, kp)


IDX = KeypointIndex(UPPER_BODY13_NAMES)


def feat(**kw):
    p = synth_person(600, 400, 100, **kw)
    return compute(p.keypoints, p.box, IDX, 0.5)


def test_features_basic():
    assert abs(feat().yaw) < 1e-6
    assert feat(yaw=0.6).yaw == pytest.approx(0.6, abs=1e-5)
    assert feat(yaw=-0.6).yaw == pytest.approx(-0.6, abs=1e-5)
    assert feat(yaw=1.0).yaw > 0.8 and feat(yaw=-1.0).yaw < -0.8  # one-ear fallback keeps the sign
    # one low-confidence ear on a frontal face must not read as a turned head
    # (on the real upper_body_n output one ear is below 0.5 in ~96% of detections)
    p = synth_person(600, 400, 100)
    p.keypoints[3, 2] = 0.1
    assert abs(compute(p.keypoints, p.box, IDX, 0.5).yaw) < 0.05
    p.keypoints[3, 2], p.keypoints[4, 2] = 0.8, 0.1
    assert abs(compute(p.keypoints, p.box, IDX, 0.5).yaw) < 0.05
    assert feat(down=0.8).head_height < feat().head_height - 0.4
    assert feat(body=0.6).torso_ratio < 0.7 * feat().torso_ratio
    assert feat(face=False).face_hidden
    assert feat(hand="raised").hand_raised and not feat().hand_raised
    assert feat(hand="face").hand_at_face and not feat().hand_at_face
    # scale invariance: same pose, student twice as far away
    near = compute(*(lambda p: (p.keypoints, p.box))(synth_person(600, 400, 120, yaw=0.5, down=0.3)), IDX, 0.5)
    far = compute(*(lambda p: (p.keypoints, p.box))(synth_person(600, 400, 60, yaw=0.5, down=0.3)), IDX, 0.5)
    assert near.yaw == pytest.approx(far.yaw, abs=1e-5)
    assert near.head_height == pytest.approx(far.head_height, abs=1e-5)


def seats_grid(n=12, base_yaw=0.0):
    out = []
    for i in range(n):
        r, c = divmod(i, 4)
        sw = 110 - 25 * r  # back rows are smaller
        out.append(dict(cx=200 + 280 * c + 20 * r, cy=560 - 170 * r, sw=sw, yaw=base_yaw))
    return out


def run_class(script, settings=None, seconds=60.0, fps=10, seats=None, analyzer=None, t0=0.0, drop=None):
    """script(i, t) -> dict of overrides for student i at time t (or None = not visible)."""
    seats = seats or seats_grid()
    an = analyzer or EngagementAnalyzer(settings or load_settings(overrides={"classifier": "none"}), UPPER_BODY13_NAMES)
    rng = np.random.default_rng(1)
    hist = []
    for k in range(int(seconds * fps)):
        t = t0 + k / fps
        persons = []
        for i, s in enumerate(seats):
            o = script(i, t)
            if o is None:
                continue
            args = {**s, **o}
            persons.append(synth_person(args.pop("cx"), args.pop("cy"), args.pop("sw"), jitter=0.01, rng=rng, **args))
        res = FrameResult(k, t, W, H, persons)
        an.update(res)
        hist.append((t, res))
    return an, hist


def states_at(hist, t):
    res = min(hist, key=lambda h: abs(h[0] - t))[1]
    return [p.engagement["state"] for p in res.persons], res


def test_distraction_detected_and_cleared():
    turned = {0, 1, 2}

    def script(i, t):
        return {"yaw": 1.0} if i in turned and 20 <= t < 35 else {}

    an, hist = run_class(script, seconds=50)
    st, res = states_at(hist, 18)
    assert st.count(FOCUSED) == 12, st
    st, res = states_at(hist, 30)
    assert [st[i] for i in turned] == [DISTRACTED] * 3 and st.count(DISTRACTED) == 3, st
    assert res.extra["class"]["pct_distracted"] == pytest.approx(25.0)
    assert res.extra["class"]["counted"] == 12
    assert next(p for p in res.persons if p.engagement["state"] == DISTRACTED).engagement["reason"] == "head_turned"
    # onset delay is bounded: smoothing + on_seconds
    first = min(t for t, r in hist if r.persons[0].engagement["state"] == DISTRACTED)
    assert 20 + 2.5 <= first <= 20 + 2.5 + 2.0
    st, _ = states_at(hist, 45)
    assert st.count(FOCUSED) == 12, st


def test_brief_glance_ignored():
    an, hist = run_class(lambda i, t: {"yaw": 1.0} if i == 5 and 20 <= t < 21 else {}, seconds=30)
    assert all(r.persons[5].engagement["state"] == FOCUSED for t, r in hist if t > 3)


def test_cues_strong_alone_weak_together():
    cases = {0: {"down": 0.9}, 1: {"face": False}, 2: {"body": 0.9, "hand": "face"},
             3: {"body": 0.7}, 4: {"hand": "face"}}

    def script(i, t):
        return cases.get(i, {}) if t >= 20 else {}

    an, hist = run_class(script, seconds=35)
    st, res = states_at(hist, 34)
    reasons = {i: res.persons[i].engagement["reason"] for i in (0, 1, 2)}
    assert [st[i] for i in (0, 1, 2)] == [DISTRACTED] * 3, (st, reasons)
    assert reasons[0] == "head_down" and reasons[1] == "face_hidden"
    assert reasons[2] in ("body_turned", "hand_at_face")
    assert st[3] == FOCUSED and st[4] == FOCUSED  # weak cues alone don't flag a student
    assert st.count(FOCUSED) == 9


def test_raised_hand_is_focused():
    an, hist = run_class(lambda i, t: {"yaw": 1.0, "hand": "raised"} if i == 0 and t >= 20 else {}, seconds=35)
    assert states_at(hist, 34)[0][0] == FOCUSED


@pytest.mark.parametrize("position,base_yaw", [("left", -0.5), ("right", 0.5), ("auto", -0.5), ("auto", 0.5)])
def test_corner_camera(position, base_yaw):
    """Corner mount: attentive students all look sideways relative to the camera. That must
    count as focused, and turning *away from the board* must count as distracted."""
    seats = seats_grid(base_yaw=base_yaw)
    away = 1.0 if base_yaw < 0 else -1.0  # turning the other way: deviation 1.5
    an, hist = run_class(lambda i, t: {"yaw": away} if i < 3 and t >= 20 else {},
                         settings=load_settings(overrides={"classifier": "none", "camera_position": position}), seconds=35, seats=seats)
    st, _ = states_at(hist, 2)
    assert st.count(FOCUSED) == 12, st  # right from the start, before any seat is calibrated
    st, _ = states_at(hist, 34)
    assert st[:3] == [DISTRACTED] * 3 and st.count(FOCUSED) == 9, st


def test_calibration_learns_save_load(tmp_path):
    cal = tmp_path / "room.json"
    s = load_settings(overrides={"classifier": "none", "calibration_seconds": 20.0, "calibration": str(cal)})
    an, hist = run_class(lambda i, t: {}, settings=s, seconds=30)
    assert all(seat.calibrated for seat in an.seats) and len(an.seats) == 12
    assert an.save_calibration(str(cal)) == 12
    an2 = EngagementAnalyzer(s, UPPER_BODY13_NAMES)
    assert len(an2.seats) == 12 and all(x.calibrated for x in an2.seats)
    # the reloaded analyzer reuses the seats (ids stable) and decides immediately
    _, hist2 = run_class(lambda i, t: {"yaw": 1.0} if i == 0 and t >= 5 else {}, analyzer=an2, seconds=15)
    ids = {p.engagement["seat"] for p in hist2[-1][1].persons}
    assert ids == {x.id for x in an.seats}
    assert all(p.engagement["calibrated"] for p in hist2[0][1].persons)
    assert states_at(hist2, 14)[0][0] == DISTRACTED
    # recalibrate ignores the file
    an3 = EngagementAnalyzer(load_settings(overrides={"classifier": "none", "calibration": str(cal), "recalibrate": True}), UPPER_BODY13_NAMES)
    assert an3.seats == []


def test_distracted_during_calibration_does_not_poison_baseline():
    """A student who looks away the whole calibration window still gets a sane baseline."""
    s = load_settings(overrides={"classifier": "none", "calibration_seconds": 20.0})
    an, hist = run_class(lambda i, t: {"yaw": 1.0} if i == 0 and t < 25 else {}, settings=s, seconds=45)
    st, _ = states_at(hist, 15)
    assert st[0] == DISTRACTED  # class reference catches it while uncalibrated
    st, _ = states_at(hist, 44)
    assert st[0] == FOCUSED  # and the clipped baseline doesn't flag normal posture later


def test_seats_survive_missing_frames_and_unknown_excluded():
    def script(i, t):
        if i == 4 and 20 <= t < 30:
            return None  # occluded / out of view for 10 s
        return {}

    an, hist = run_class(script, seconds=40)
    st, res = states_at(hist, 25)
    assert len(res.persons) == 11 and res.extra["class"]["counted"] == 11
    seat4 = hist[100][1].persons[4].engagement["seat"]
    st, res = states_at(hist, 35)
    assert res.persons[4].engagement["seat"] == seat4  # same seat after reappearing
    assert len(an.seats) == 12


def test_low_confidence_person_is_unknown():
    def script(i, t):
        return {"face": False, "hidden_all": True} if i == 0 else {}

    seats = seats_grid()
    an = EngagementAnalyzer(load_settings(overrides={"classifier": "none"}), UPPER_BODY13_NAMES)
    for k in range(100):
        t = k / 10
        persons = [synth_person(s["cx"], s["cy"], s["sw"]) for s in seats]
        persons[0].keypoints[:, 2] = 0.1  # nothing usable
        res = FrameResult(k, t, W, H, persons)
        an.update(res)
    assert res.persons[0].engagement["state"] == UNKNOWN
    assert res.extra["class"]["counted"] == 11 and res.extra["class"]["unknown"] == 1


def test_jsonl_and_settings(tmp_path):
    an, hist = run_class(lambda i, t: {}, seconds=3)
    path = tmp_path / "o.jsonl"
    w = JsonlWriter(str(path), {"x": 1})
    w(hist[-1][1])
    w.close()
    line = json.loads(path.read_text().splitlines()[1])
    assert line["class"]["counted"] == 12 and line["persons"][0]["engagement"]["state"] == FOCUSED
    with pytest.raises(ValueError):
        load_settings(overrides={"classifier": "none", "display": "everything"})
    y = tmp_path / "e.yaml"
    y.write_text("display: class\ncamera_position: left\nweights:\n  head_down: 0.0\n")
    s = load_settings(str(y))
    assert s.display == "class" and s.camera_position == "left" and s.weights["head_down"] == 0 and s.weights["head_turned"] == 1
    y.write_text("bogus: 1\n")
    with pytest.raises(ValueError):
        load_settings(str(y))


def test_analyze_command(tmp_path, capsys):
    """`hcpose analyze` replays a --save-json file through the analyzer offline."""
    from hcpose.cli import main

    src = tmp_path / "skel.jsonl"
    w = JsonlWriter(str(src), {"keypoint_names": UPPER_BODY13_NAMES})
    an, hist = run_class(lambda i, t: {"yaw": 1.0} if i < 6 and t >= 60 else {}, seconds=90)
    for _, res in hist:
        for p in res.persons:
            p.engagement = None  # as written by a run without --engagement
        res.extra = {}
        w(res)
    w.close()
    out = tmp_path / "labelled.jsonl"
    cal = tmp_path / "room.json"
    # rules, so the expected percentages don't depend on which learned model is bundled
    assert main(["analyze", str(src), "-o", str(out), "--calibration", str(cal), "--every", "10",
                 "--classifier", "none"]) == 0
    text = capsys.readouterr().out
    rows = [ln.split() for ln in text.splitlines() if ln.strip() and ln.split()[0].replace(".", "").isdigit()]
    pct = [float(r[1].rstrip("%")) for r in rows]
    assert pct[0] == 0.0 and pct[-1] == pytest.approx(50.0, abs=1.0), text
    last = json.loads(out.read_text().splitlines()[-1])
    assert last["class"]["distracted"] == 6
    assert cal.is_file() and json.loads(cal.read_text())["seats"] == []  # 90 s < calibration_seconds


def test_most_of_class_distracted():
    """The headline number must work when most of the class looks away together."""
    an, hist = run_class(lambda i, t: {"yaw": 1.0} if i < 10 and t >= 60 else {}, seconds=90)
    st, res = states_at(hist, 89)
    assert st.count(DISTRACTED) == 10, st
    assert res.extra["class"]["pct_distracted"] == pytest.approx(83.3, abs=0.1)
    assert res.extra["class"]["pct_distracted_smoothed"] == pytest.approx(83.3, abs=0.1)


def test_camera_setting_anchors_a_distracted_start():
    """If most students look away from the very start, auto mode can't know the board
    direction, but a camera_position setting can."""
    script = lambda i, t: {"yaw": 1.0} if i < 9 else {}  # noqa: E731
    _, hist = run_class(script, settings=load_settings(overrides={"classifier": "none", "camera_position": "center"}), seconds=30)
    st, _ = states_at(hist, 29)
    assert st.count(DISTRACTED) == 9 and st.count(FOCUSED) == 3, st


# ---------------------------------------------------------------- learned classifier
def _train_synthetic(tmp_path):
    """A tiny model trained on synthetic students: turned head or face hidden = distracted."""
    pytest.importorskip("sklearn")
    from sklearn.ensemble import HistGradientBoostingClassifier

    from hcpose.engagement.classifier import export_hgb, feature_vector
    from hcpose.engagement.features import Baseline

    rng = np.random.default_rng(0)
    X, y = [], []
    for _ in range(1500):
        dis = rng.random() < 0.3
        yaw = rng.choice([-1.0, 1.0]) * rng.uniform(0.8, 1.2) if dis else rng.normal(0, 0.1)
        p = synth_person(rng.uniform(200, 1000), rng.uniform(300, 600), rng.uniform(50, 120), yaw=yaw,
                         jitter=0.02, rng=rng)
        f = compute(p.keypoints, p.box, IDX, 0.5)
        X.append(feature_vector(p.keypoints, p.box, f, Baseline(0.0, 0.9, 0.38), IDX, 0.5))
        y.append(dis)
    est = HistGradientBoostingClassifier(max_iter=60, random_state=0).fit(np.vstack(X), y)
    path = tmp_path / "clf.json"
    export_hgb(est, str(path), {"name": "synthetic"})
    return est, path, np.vstack(X)


def test_classifier_export_matches_sklearn(tmp_path):
    from hcpose.engagement.classifier import TreeEnsemble

    est, path, X = _train_synthetic(tmp_path)
    rt = TreeEnsemble.load(str(path))
    Xn = X.copy()
    Xn[::7, -6] = np.nan  # missing yaw_dev must follow sklearn's missing-value routing
    assert np.abs(rt.predict_proba(Xn) - est.predict_proba(Xn)[:, 1]).max() < 1e-9


def test_analyzer_with_classifier(tmp_path):
    _, path, _ = _train_synthetic(tmp_path)
    s = load_settings(overrides={"classifier": str(path)})
    an, hist = run_class(lambda i, t: {"yaw": 1.0} if i < 3 and t >= 20 else {}, settings=s, seconds=35)
    assert an.classifier is not None
    st, res = states_at(hist, 34)
    assert st[:3] == [DISTRACTED] * 3 and st.count(FOCUSED) == 9, st
    assert res.persons[0].engagement["reason"] == "head_turned"  # reasons still come from the rule cues
