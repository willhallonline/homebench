import json
import math
import os

import pytest

from homebench.quality import default_suite, load_pack, load_packs
from homebench.quality.packs import PackError

EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "examples")


def _write(tmp_path, name, data):
    p = tmp_path / name
    p.write_text(json.dumps(data) if name.endswith(".json") else data)
    return str(p)


def test_load_json_pack(tmp_path):
    path = _write(tmp_path, "p.json", {
        "name": "t",
        "tasks": [
            {"id": "a", "category": "math", "prompt": "2+2?",
             "grader": {"type": "exact_number", "value": 4}, "reference": "4"},
            {"id": "b", "category": "factual", "prompt": "cap of France?",
             "grader": {"type": "contains_any", "values": ["Paris"]},
             "reference": "Paris"},
        ],
    })
    tasks = load_pack(path)
    assert [t.id for t in tasks] == ["a", "b"]
    assert all(not t.strip_think for t in tasks)
    assert tasks[0].grader("the answer is 4").passed
    assert tasks[1].grader("Paris").passed


def test_bare_list_pack(tmp_path):
    path = _write(tmp_path, "p.json", [
        {"id": "x", "prompt": "hi", "grader": {"type": "contains_any", "values": ["y"]}},
    ])
    tasks = load_pack(path)
    assert len(tasks) == 1 and tasks[0].category == "custom"


def test_open_task_has_no_grader(tmp_path):
    path = _write(tmp_path, "p.json", [
        {"id": "o", "prompt": "explain X", "reference": "because"},
    ])
    t = load_pack(path)[0]
    assert t.grader is None
    assert t.category == "open"  # normalized


def test_all_grader_types(tmp_path):
    path = _write(tmp_path, "p.json", [
        {"id": "n", "prompt": "?", "grader": {"type": "exact_number", "value": 7},
         "reference": "7"},
        {"id": "mc", "prompt": "?", "grader": {"type": "multiple_choice", "value": "B"},
         "reference": "B"},
        {"id": "c", "prompt": "?", "grader": {"type": "contains_any", "values": ["a", "b"]},
         "reference": "a"},
        {"id": "r", "prompt": "?", "grader": {"type": "regex", "pattern": "foo\\d+"},
         "reference": "foo12"},
        {"id": "j", "prompt": "?", "grader": {"type": "valid_json", "keys": ["k"]},
         "reference": "{\"k\": 1}"},
        {"id": "arr", "prompt": "?", "grader": {"type": "valid_json_array", "length": 2},
         "reference": "[1, 2]"},
    ])
    for t in load_pack(path):
        assert t.grader(t.reference).passed, t.id


def test_regex_case_sensitive_flag(tmp_path):
    path = _write(tmp_path, "p.json", [
        {"id": "r", "prompt": "?",
         "grader": {"type": "regex", "pattern": "HELLO", "ignorecase": False}},
    ])
    g = load_pack(path)[0].grader
    assert g("HELLO").passed
    assert not g("hello").passed


def test_unknown_grader_type_errors(tmp_path):
    path = _write(tmp_path, "p.json", [
        {"id": "x", "prompt": "?", "grader": {"type": "nonsense"}},
    ])
    with pytest.raises(PackError):
        load_pack(path)


def test_missing_required_fields(tmp_path):
    p1 = _write(tmp_path, "p1.json", [{"category": "math", "prompt": "?"}])
    with pytest.raises(PackError):
        load_pack(p1)
    p2 = _write(tmp_path, "p2.json", [{"id": "a"}])
    with pytest.raises(PackError):
        load_pack(p2)


def test_duplicate_ids(tmp_path):
    path = _write(tmp_path, "p.json", [
        {"id": "a", "prompt": "?", "grader": {"type": "contains_any", "values": ["x"]}},
        {"id": "a", "prompt": "?", "grader": {"type": "contains_any", "values": ["y"]}},
    ])
    with pytest.raises(PackError):
        load_pack(path)


def test_empty_pack_errors(tmp_path):
    path = _write(tmp_path, "p.json", {"tasks": []})
    with pytest.raises(PackError):
        load_pack(path)


def test_load_packs_dedup_across_files(tmp_path):
    a = _write(tmp_path, "a.json", [
        {"id": "dup", "prompt": "?", "grader": {"type": "contains_any", "values": ["x"]}}])
    b = _write(tmp_path, "b.json", [
        {"id": "dup", "prompt": "?", "grader": {"type": "contains_any", "values": ["y"]}}])
    with pytest.raises(PackError):
        load_packs([a, b])


def test_shipped_example_pack_json():
    path = os.path.join(EXAMPLES, "sample_pack.json")
    tasks = load_pack(path)
    assert len(tasks) == 4
    for t in tasks:
        if t.grader is not None:
            assert t.grader(t.reference).passed, t.id


def test_new_example_packs():
    counts = {
        "sample_pack_years.json": 18,
        "sample_pack_math.json": 20,
        "sample_pack_comprehension.json": 20,
        "sample_pack_calibration.json": 20,
        "sample_pack_instructions.json": 20,
        "sample_pack_multilingual.json": 20,
        "sample_pack_summarization.json": 20,
        "sample_pack_coding.json": 20,
    }
    paths = [os.path.join(EXAMPLES, name) for name in counts]
    for path, expected in zip(paths, counts.values()):
        tasks = load_pack(path)
        assert len(tasks) == expected
        for task in tasks:
            assert task.reference, task.id
            if task.grader is not None:
                assert task.grader(task.reference).passed, task.id
    tasks = load_packs(paths)
    assert len(tasks) == sum(counts.values())
    summary = load_pack(os.path.join(EXAMPLES, "sample_pack_summarization.json"))
    assert all(task.category == "open" and task.grader is None for task in summary)
    coding = load_pack(os.path.join(EXAMPLES, "sample_pack_coding.json"))
    assert all(task.category == "open" and task.grader is None for task in coding)


def test_math_pack_answers():
    expected = [
        38 + 57, 1000 - 487, 24 * 17, 936 / 24, 18 + 6 * 7,
        144 / 12 + 3 ** 3, 3 / 4 + 5 / 8, 0.15 * 240, 64 * 5 / 8,
        (43 + 9) / 4, (23 + 10) / 3, 2 ** 5 + 3 * math.sqrt(144),
        4 + 19 * 5, sum(3 * 2 ** index for index in range(8)),
        math.comb(10, 3), round(7 / 12, 6), math.sqrt(13 ** 2 - 5 ** 2),
        math.gcd(756, 630), 3 * 3 ** 2 - 8 * 3 + 1,
        (3 ** 4) / 2 - (3 * 3 ** 2) / 2 + 4 * 3,
    ]
    tasks = load_pack(os.path.join(EXAMPLES, "sample_pack_math.json"))
    assert len(tasks) == len(expected)
    for task, answer in zip(tasks, expected):
        assert float(task.reference) == pytest.approx(answer), task.id
        assert task.grader(str(answer)).passed, task.id


@pytest.mark.parametrize("filename,task_id,accepted,rejected", [
    ("sample_pack_years.json", "uk_pm_2022",
     ["Boris Johnson\nLiz Truss\nRishi Sunak"], ["Boris Johnson and Liz Truss"]),
    ("sample_pack_years.json", "uk_pm_2024",
     ["Rishi Sunak\nKeir Starmer"], ["Rishi Sunak"]),
    ("sample_pack_years.json", "uk_pm_2026", ["Andy Burnham"], ["Keir Starmer"]),
    ("sample_pack_years.json", "nobel_2026_contribution", ["B"], ["A"]),
    ("sample_pack_comprehension.json", "read_10_shuttle_days",
     ["Weekdays only"], ["Monday", "Friday", "Weekends only"]),
    ("sample_pack_instructions.json", "inst_08_choice_only",
     ["B"], ["B, because 12 is even"]),
    ("sample_pack_instructions.json", "inst_19_json_object",
     ['{"ok":true,"count":3}', '{"count":3,"ok":true}'],
     ['{"ok":false,"count":3}', '{"ok":true,"count":4}']),
    ("sample_pack_multilingual.json", "lang_16_japanese_cat_color",
     ["白", "white"], ["black"]),
])
def test_example_pack_grader_cases(filename, task_id, accepted, rejected):
    tasks = load_pack(os.path.join(EXAMPLES, filename))
    task = next(task for task in tasks if task.id == task_id)
    for response in accepted:
        assert task.grader(response).passed, response
    for response in rejected:
        assert not task.grader(response).passed, response


def test_yaml_pack():
    yaml = pytest.importorskip("yaml")  # noqa: F841
    path = os.path.join(EXAMPLES, "sample_pack.yaml")
    tasks = load_pack(path)
    assert any(t.category == "open" for t in tasks)
    for t in tasks:
        if t.grader is not None:
            assert t.grader(t.reference).passed, t.id
