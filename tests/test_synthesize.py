"""The Synthesizer API and the command line on a tiny random model (CPU, no downloads): defaults, per-voice
vocoders, sentence joins, streaming pieces, the ``say`` command."""

import pytest
import soundfile as sf
import torch

from fast_tr_drifting.audio import HOP_LENGTH, SAMPLE_RATE
from fast_tr_drifting.cli import COMMANDS, main
from fast_tr_drifting.synthesize import DEFAULT_CFG, Synthesizer

from .test_batched import _synth, _tiny_vocos


def test_defaults_are_the_release_settings(tmp_path):
    synth = _synth(tmp_path, prosody=True)
    assert synth.default_temperature == 0.5  # the tiny checkpoint has no calibrated temperature
    synth.model.temperature = 0.3
    synth.default_temperature = 0.3
    a, _ = synth("Merhaba, nasılsınız?", speaker=2, seed=3)
    b, _ = synth("Merhaba, nasılsınız?", speaker=2, seed=3, cfg_scale=DEFAULT_CFG, temperature=0.3)
    assert DEFAULT_CFG == 2.0 and torch.equal(a, b)


def test_sentences_are_joined_by_the_pause(tmp_path):
    synth = _synth(tmp_path, prosody=True)
    text = "Merhaba! Bugün hava çok güzel."
    wav, info = synth(text, speaker=2, seed=1, pause=0.25)
    mels = synth.mels(text, speaker=2, seed=1)
    frames = sum(m.shape[-1] for m in mels)
    assert len(mels) == 2 and wav.numel() == frames * HOP_LENGTH + int(0.25 * SAMPLE_RATE)
    assert info["seconds"] == pytest.approx(wav.numel() / SAMPLE_RATE)
    pieces = list(synth.stream(text, speaker=2, seed=1, pause=0.25, first=4, chunk=12))
    torch.testing.assert_close(torch.cat(pieces), wav, rtol=0, atol=1e-5)  # the vocoder streams exactly
    assert pieces[0].numel() == 4 * HOP_LENGTH


def test_a_seed_changes_the_intonation_not_the_rhythm(tmp_path):
    synth = _synth(tmp_path, prosody=True)  # prosody_durations="regressor": the release's setting
    a = synth.mels("Bu bir deneme cümlesidir.", speaker=2, seed=0)[0]
    b = synth.mels("Bu bir deneme cümlesidir.", speaker=2, seed=1)[0]
    assert a.shape == b.shape and not torch.equal(a, b)


def test_per_voice_vocoders(tmp_path):
    synth = _synth(tmp_path, prosody=False)
    other = _tiny_vocos(seed=1)
    synth.voice_vocoders = {389: (other, None)}
    assert synth.vocoder_for("male") is other and synth.vocoder_for(389) is other
    assert synth.vocoder_for("studio") is synth.vocoder and synth.vocoder_for(2) is synth.vocoder
    v = synth.variant(vocoder=other)
    assert v.vocoder is other and v.voice_vocoders == {} and v.model is synth.model


def test_say_command(tmp_path, capsys):
    synth = _synth(tmp_path, prosody=False)
    torch.save({"vocos": synth.vocoder.model.state_dict(), "mel": "bigvgan", "head_padding": "same"},
               tmp_path / "voc.pt")
    out = tmp_path / "out.wav"
    base = ["say", "Merhaba! Nasılsınız?", "--model", str(tmp_path / "tts.pt"), "--vocoder", str(tmp_path / "voc.pt"),
            "--voice", "2", "--device", "cpu", "--seed", "4"]
    main([*base, "-o", str(out)])
    wav, sr = sf.read(out)
    assert sr == SAMPLE_RATE and len(wav) > 0
    main([*base, "-o", str(tmp_path / "stream.wav"), "--stream", "--chunked"])
    assert "first audio after" in capsys.readouterr().out
    assert len(sf.read(tmp_path / "stream.wav")[0]) == len(wav)  # the same durations: as many samples


def test_cli_lists_every_command(capsys):
    main([])
    text = capsys.readouterr().out
    assert all(name in text for name in COMMANDS) and "say" in COMMANDS
    with pytest.raises(SystemExit):
        main(["no-such-command"])


def test_unknown_speaker_and_release(tmp_path):
    synth = _synth(tmp_path, prosody=False)
    with pytest.raises(ValueError, match="not in this checkpoint"):
        synth("merhaba", speaker=99)
    with pytest.raises(ValueError, match="unknown release"):
        Synthesizer.from_pretrained("v9", "cpu")
