import io
import json
import sys
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import llm  # noqa: E402
from app.llm import Decision, review_candidates, validate_llm_output  # noqa: E402


def make_cfg(enabled=True, target=0.6, **llm_kwargs):
    llm_cfg = dict(
        enabled=enabled,
        url="http://ollama.test:11434",
        model="qwen2.5:7b-instruct",
        timeout_s=5,
    )
    llm_cfg.update(llm_kwargs)
    return SimpleNamespace(
        llm=SimpleNamespace(**llm_cfg),
        cuts=SimpleNamespace(target_pause_s=target),
    )


CANDIDATES = [
    {"id": "c1", "start": 10.0, "end": 14.0},
    {"id": "c2", "start": 30.0, "end": 31.0},
]

TRANSCRIPT = {
    "segments": [
        {"start": 0.0, "end": 9.5, "text": "Labas, pradėkime."},
        {"start": 14.2, "end": 20.0, "text": "Next slide shows results."},
        {"start": 60.0, "end": 65.0, "text": "Far away text."},
    ]
}

SLIDES = [5.0, 15.0, 90.0]


def default_for(cid, target, reason):
    return Decision(cid, "shorten", target, reason, False)


# --- validate_llm_output ----------------------------------------------------


def test_valid_output_parsed_in_candidate_order():
    raw = json.dumps(
        {
            "decisions": [
                {"id": "c2", "action": "keep", "reason": "emphasis", "needs_review": True},
                {"id": "c1", "action": "shorten", "target_pause_s": 1.0, "reason": "ok"},
            ]
        }
    )
    out = validate_llm_output(raw, CANDIDATES, make_cfg())
    assert out == [
        Decision("c1", "shorten", 1.0, "ok", False),
        Decision("c2", "keep", None, "emphasis", True),
    ]


def test_approve_has_no_target():
    raw = json.dumps([{"id": "c1", "action": "approve", "target_pause_s": 3}])
    out = validate_llm_output(raw, CANDIDATES[:1], make_cfg())
    assert out[0].action == "approve"
    assert out[0].target_pause_s is None


@pytest.mark.parametrize("raw", ["not json", "{", "", b"\xff\xfe", "42", '"text"', '{"x": 1}'])
def test_invalid_json_rejected_with_defaults(raw):
    out = validate_llm_output(raw, CANDIDATES, make_cfg())
    assert out == [
        default_for("c1", 0.6, "llm_invalid_json"),
        default_for("c2", 0.6, "llm_invalid_json"),
    ]


@pytest.mark.parametrize(
    "value,expected",
    [(0.01, 0.2), (-5, 0.2), (100.0, 4.0), (2.5, 2.5), ("1.5", 1.5)],
)
def test_target_pause_clamped(value, expected):
    raw = json.dumps([{"id": "c1", "action": "shorten", "target_pause_s": value}])
    out = validate_llm_output(raw, CANDIDATES[:1], make_cfg())
    assert out[0].target_pause_s == pytest.approx(expected)


@pytest.mark.parametrize("value", [None, "abc", True, [1], float("nan")])
def test_invalid_target_uses_config_default(value):
    raw = json.dumps([{"id": "c1", "action": "shorten", "target_pause_s": value}])
    out = validate_llm_output(raw, CANDIDATES[:1], make_cfg(target=0.8))
    assert out[0].target_pause_s == pytest.approx(0.8)


def test_default_target_clamped_to_short_candidate():
    cands = [{"id": "s", "start": 1.0, "end": 1.4}]
    out = validate_llm_output("[]", cands, make_cfg(target=0.6))
    assert out[0].target_pause_s == pytest.approx(0.4)


def test_unknown_ids_and_invented_timestamps_ignored():
    raw = json.dumps(
        {
            "decisions": [
                {"id": "c99", "action": "approve", "start": 0, "end": 100},
                {
                    "id": "c1",
                    "action": "approve",
                    "start": 0.0,
                    "end": 50.0,
                    "new_start": 1.0,
                    "timestamps": [1, 2],
                    "reason": "silent",
                },
            ]
        }
    )
    out = validate_llm_output(raw, CANDIDATES, make_cfg())
    assert len(out) == 2
    assert [d.candidate_id for d in out] == ["c1", "c2"]
    assert out[0] == Decision("c1", "approve", None, "silent", False)
    assert not hasattr(out[0], "start")
    assert out[1] == default_for("c2", 0.6, "deterministic_default")
    # Candidate intervals are never mutated by the LLM output.
    assert CANDIDATES[0] == {"id": "c1", "start": 10.0, "end": 14.0}


@pytest.mark.parametrize("action", ["delete", "", None, 5, "cut"])
def test_invalid_action_falls_back_to_default(action):
    raw = json.dumps([{"id": "c1", "action": action, "needs_review": True}])
    out = validate_llm_output(raw, CANDIDATES[:1], make_cfg())
    assert out == [default_for("c1", 0.6, "llm_invalid_action")]


def test_missing_candidates_get_deterministic_default():
    raw = json.dumps({"decisions": [{"id": "c1", "action": "keep"}]})
    out = validate_llm_output(raw, CANDIDATES, make_cfg(target=0.5))
    assert out[1] == default_for("c2", 0.5, "deterministic_default")


def test_needs_review_must_be_boolean_true():
    raw = json.dumps([{"id": "c1", "action": "keep", "needs_review": "yes"}])
    out = validate_llm_output(raw, CANDIDATES[:1], make_cfg())
    assert out[0].needs_review is False


def test_numeric_ids_match_and_duplicates_use_first():
    cands = [{"id": 1, "start": 0.0, "end": 3.0}]
    raw = json.dumps(
        [
            {"candidate_id": "1", "action": "keep"},
            {"id": 1, "action": "approve"},
            "garbage",
        ]
    )
    out = validate_llm_output(raw, cands, make_cfg())
    assert out == [Decision(1, "keep", None, "", False)]


def test_attribute_candidates_supported():
    cands = [SimpleNamespace(id="a", start=0.0, end=2.0)]
    raw = json.dumps([{"id": "a", "action": "shorten", "target_pause_s": 9}])
    out = validate_llm_output(raw, cands, make_cfg())
    assert out[0].target_pause_s == pytest.approx(2.0)


# --- review_candidates (HTTP mocked) ---------------------------------------


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def fake_ollama(monkeypatch):
    calls = []
    state = {"replies": [], "error": None}

    def fake_urlopen(request, timeout=None):
        calls.append({"request": request, "timeout": timeout})
        if state["error"] is not None:
            raise state["error"]
        content = state["replies"].pop(0)
        body = {"message": {"role": "assistant", "content": content}}
        return FakeResponse(json.dumps(body).encode("utf-8"))

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    return SimpleNamespace(calls=calls, state=state)


def test_disabled_llm_makes_no_calls(fake_ollama):
    out = review_candidates(CANDIDATES, TRANSCRIPT, SLIDES, make_cfg(enabled=False))
    assert fake_ollama.calls == []
    assert [d.reason for d in out] == ["llm_disabled", "llm_disabled"]
    assert all(d.action == "shorten" and d.target_pause_s == 0.6 for d in out)


def test_empty_candidates(fake_ollama):
    assert review_candidates([], TRANSCRIPT, SLIDES, make_cfg()) == []
    assert fake_ollama.calls == []


def test_request_format_and_context(fake_ollama):
    fake_ollama.state["replies"] = [
        json.dumps(
            {
                "decisions": [
                    {"id": "c1", "action": "approve", "reason": "dead air"},
                    {"id": "c2", "action": "keep", "reason": "pause"},
                ]
            }
        )
    ]
    out = review_candidates(CANDIDATES, TRANSCRIPT, SLIDES, make_cfg())
    assert [d.action for d in out] == ["approve", "keep"]

    assert len(fake_ollama.calls) == 1
    call = fake_ollama.calls[0]
    req = call["request"]
    assert req.full_url == "http://ollama.test:11434/api/chat"
    assert req.get_method() == "POST"
    assert call["timeout"] == 5.0
    body = json.loads(req.data)
    assert body["model"] == "qwen2.5:7b-instruct"
    assert body["format"] == "json"
    assert body["stream"] is False
    assert body["options"]["temperature"] == 0

    user = json.loads(body["messages"][-1]["content"])
    c1, c2 = user["candidates"]
    assert (c1["id"], c1["start"], c1["end"]) == ("c1", 10.0, 14.0)
    assert "Labas" in c1["transcript_context"]
    assert "Next slide" in c1["transcript_context"]
    assert "Far away" not in c1["transcript_context"]
    assert c1["nearby_slide_changes"] == [5.0, 15.0]
    assert c2["nearby_slide_changes"] == [15.0]


def test_default_model_used_when_not_configured(fake_ollama):
    fake_ollama.state["replies"] = ["[]"]
    cfg = SimpleNamespace(
        llm=SimpleNamespace(enabled=True, url="http://x:1"),
        cuts=SimpleNamespace(target_pause_s=0.6),
    )
    review_candidates(CANDIDATES[:1], None, None, cfg)
    body = json.loads(fake_ollama.calls[0]["request"].data)
    assert body["model"] == "qwen2.5:7b-instruct"


def test_batching(fake_ollama):
    cands = [{"id": f"c{i}", "start": i * 10.0, "end": i * 10.0 + 3} for i in range(5)]
    fake_ollama.state["replies"] = [
        json.dumps([{"id": "c0", "action": "keep"}, {"id": "c1", "action": "keep"}]),
        json.dumps([{"id": "c2", "action": "approve"}, {"id": "c3", "action": "keep"}]),
        json.dumps([{"id": "c4", "action": "approve"}]),
    ]
    out = review_candidates(cands, [], [], make_cfg(batch_size=2))
    assert len(fake_ollama.calls) == 3
    assert [d.action for d in out] == ["keep", "keep", "approve", "keep", "approve"]


@pytest.mark.parametrize(
    "error",
    [
        urllib.error.URLError("connection refused"),
        TimeoutError("timed out"),
        ConnectionResetError(),
    ],
)
def test_network_failure_falls_back(fake_ollama, error):
    fake_ollama.state["error"] = error
    out = review_candidates(CANDIDATES, TRANSCRIPT, SLIDES, make_cfg())
    assert out == [
        default_for("c1", 0.6, "llm_unavailable"),
        default_for("c2", 0.6, "llm_unavailable"),
    ]


def test_malformed_ollama_response_falls_back(monkeypatch):
    monkeypatch.setattr(
        llm.urllib.request,
        "urlopen",
        lambda request, timeout=None: FakeResponse(b'{"unexpected": true}'),
    )
    out = review_candidates(CANDIDATES, TRANSCRIPT, SLIDES, make_cfg())
    assert [d.reason for d in out] == ["llm_unavailable"] * 2


def test_failure_in_later_batch_falls_back_for_all(fake_ollama):
    fake_ollama.state["replies"] = [json.dumps([{"id": "c1", "action": "keep"}])]
    out = review_candidates(CANDIDATES, TRANSCRIPT, SLIDES, make_cfg(batch_size=1))
    assert [d.reason for d in out] == ["llm_unavailable"] * 2


def test_invalid_llm_json_via_review(fake_ollama):
    fake_ollama.state["replies"] = ["sorry, I cannot do that"]
    out = review_candidates(CANDIDATES, TRANSCRIPT, SLIDES, make_cfg())
    assert [d.reason for d in out] == ["llm_invalid_json"] * 2
