"""Release plumbing: the release table, the local mirror, names of the published vocoders and prosody predictors,
pauses by name. CPU, no downloads."""

import argparse

import pytest

from fast_tr_drifting.hub import (
    HUB_DIR_ENV,
    LATEST,
    PROSODY_MODELS,
    RELEASES,
    VOCODERS,
    hub_file,
    resolve_prosody,
    resolve_vocoder,
)
from fast_tr_drifting.pauses import PausePolicy
from fast_tr_drifting.synthesize import pause_arg, resolve_pause
from fast_tr_drifting.voices import DEFAULT_VOICE, VOICE_VOCODERS, VOICES, voice_id


def test_release_table():
    assert LATEST == "v3.2" and set(RELEASES) == {"v3.2"}
    v32 = RELEASES["v3.2"]
    assert (v32["model"], v32["vocoder"], v32["prosody"], v32["prosody_durations"], v32["pause"]) == \
        ("drifting_tts_v3.2.pt", "vocos-v2", "drift", "regressor", "punct")
    assert v32["vocoder"] in VOCODERS and v32["prosody"] in PROSODY_MODELS
    assert set(VOCODERS.values()) == {"vocos_v2.pt", "vocos_v2_balanced.pt"}


def test_voices_and_their_vocoders():
    assert DEFAULT_VOICE == "studio" and voice_id("studio") == 722 and voice_id("male") == 389
    assert voice_id("female") == 323 and voice_id(17) == 17 and voice_id("17") == 17
    assert {voice_id(v) for v in VOICES} == set(VOICE_VOCODERS)
    assert VOICE_VOCODERS[722] == "vocos-v2" and VOICE_VOCODERS[389] == VOICE_VOCODERS[323] == "vocos-v2-balanced"
    with pytest.raises(KeyError):
        voice_id("narrator")


def test_local_mirror(tmp_path, monkeypatch):
    for name in ("prosody_drift_v3.2.pt", "vocos_v2.pt"):
        (tmp_path / name).write_bytes(b"x")
    monkeypatch.setenv(HUB_DIR_ENV, str(tmp_path))
    assert hub_file("vocos_v2.pt") == str(tmp_path / "vocos_v2.pt")
    assert resolve_prosody("drift") == str(tmp_path / "prosody_drift_v3.2.pt")
    assert resolve_vocoder("vocos-v2") == str(tmp_path / "vocos_v2.pt")
    assert resolve_vocoder(tmp_path / "vocos_v2.pt") == str(tmp_path / "vocos_v2.pt")
    with pytest.raises(ValueError, match="drift"):
        resolve_prosody("no-such-predictor")
    with pytest.raises(ValueError, match="vocos-v2"):
        resolve_vocoder("bigvgan-v2")


def test_pauses_by_name():
    assert pause_arg("0.3") == 0.3 and pause_arg("punct") == "punct" and pause_arg("punct:1") == "punct:1"
    with pytest.raises(argparse.ArgumentTypeError):
        pause_arg("long")
    assert resolve_pause(0.2, 722) == 0.2
    p = resolve_pause("punct", 389)
    assert isinstance(p, PausePolicy) and p("a.") == PausePolicy.for_voice(389)("a.") and p.jitter == 0
    assert resolve_pause("punct:0.5", 722).jitter == 0.5
    with pytest.raises(ValueError):
        resolve_pause("pause", 722)


def test_pause_policy_by_final_mark():
    p = PausePolicy.for_voice(722)
    assert p("bir soru mu?") == pytest.approx(max(0.0, 0.103 - 0.161))
    assert p("evet.") == pytest.approx(max(0.0, 0.125 - 0.161))
    q = PausePolicy.for_voice(389)
    assert q("tamam.") == pytest.approx(0.848 - 0.176) and q("ve,") == pytest.approx(0.473 - 0.176)
    assert q("bitmedi") == q("ve,")  # cut without a mark: the comma's pause
    assert PausePolicy.for_voice(5).gaps == PausePolicy.for_voice("base").gaps  # other speakers: the corpus
