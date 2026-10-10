"""Bit-exact parity with the training repository (``drifting_tts``, github.com/kadirnar/drifting-tts) on tiny random
models: whole sentences, the vocoder's streaming windows, the streaming DiT and the batched path. Skipped unless
``drifting_tts`` and ``vocos`` are importable (``scripts/parity.py`` runs the same comparison on the released
checkpoints, on a GPU)."""

import pytest
import torch

pytest.importorskip("vocos")
pytest.importorskip("drifting_tts.noise", reason="needs drifting_tts at a51553a or later (counter-based noise)")
ref_synthesize = pytest.importorskip("drifting_tts.synthesize")

from fast_tr_drifting.batched import stream_batched  # noqa: E402
from fast_tr_drifting.chunked import Chunking  # noqa: E402
from fast_tr_drifting.synthesize import Synthesizer  # noqa: E402

from .test_batched import MODEL, NET, _models, _tiny_vocos  # noqa: E402

TEXTS = ["Merhaba! Bugün hava çok güzel.", "Bu bir deneme.", "Kısa bir cümle daha? Evet, bir tane daha.",
         "Yapay zekâ modelleri her geçen gün daha hızlı ve daha verimli hâle geliyor."]


def _pair(tmp_path, durations: str = "regressor"):
    """The same tiny checkpoints and vocoder weights in both packages."""
    from drifting_tts.vocoder import Vocoder as RefVocoder
    from vocos.feature_extractors import MelSpectrogramFeatures
    from vocos.heads import ISTFTHead
    from vocos.models import VocosBackbone
    from vocos.pretrained import Vocos as PackageVocos

    model, pred = _models()
    torch.save({"ema": model.state_dict(), "config": {"data": {"root": str(tmp_path)}, "model": MODEL},
                "num_speakers": 3, "stats": {"mean": -5.0, "std": 2.0, "backend": "bigvgan"},
                "duration_scales": {2: 1.1}}, tmp_path / "tts.pt")
    torch.save({"ema": pred.net.state_dict(), "stats": pred.stats.state_dict(), "net_cfg": NET, "cond_dim": 26,
                "temperature": 0.5, "duration_scales": {2: 0.9}}, tmp_path / "prosody.pt")
    ours = _tiny_vocos()
    pkg = PackageVocos(MelSpectrogramFeatures(), VocosBackbone(100, 16, 32, 2), ISTFTHead(16, 1024, 256)).eval()
    pkg.head.istft.padding = "same"
    pkg.load_state_dict({**pkg.state_dict(), **ours.model.state_dict()})
    kw = dict(prosody=str(tmp_path / "prosody.pt"), prosody_durations=durations, pause="punct")
    ref = ref_synthesize.Synthesizer(tmp_path / "tts.pt", "cpu", vocoder="griffin-lim", **kw)
    ref.vocoder = RefVocoder.wrap(pkg, "vocos", "cpu", "bigvgan", "tiny", context=8)
    new = Synthesizer(tmp_path / "tts.pt", "cpu", vocoder=ours, **kw)
    return ref, new


@pytest.mark.parametrize("durations", ["regressor", "sampled"])
def test_whole_sentences_and_streams_are_bit_exact(tmp_path, durations):
    ref, new = _pair(tmp_path, durations)
    kw = dict(speaker=2, cfg_scale=2.0, temperature=0.3)
    for seed, text in enumerate(TEXTS):
        for a, b in zip(ref.mels(text, seed=seed, **kw), new.mels(text, seed=seed, **kw)):
            assert torch.equal(a, b)
        assert torch.equal(ref(text, seed=seed, **kw)[0], new(text, seed=seed, **kw)[0])
        s = dict(seed=seed, first=4, chunk=12, **kw)
        assert torch.equal(torch.cat(list(ref.stream(text, **s))), torch.cat(list(new.stream(text, **s))))


@pytest.mark.parametrize("noise", ["torch", "philox"])
def test_streaming_dit_and_batches_are_bit_exact(tmp_path, noise):
    from drifting_tts.batched import stream_batched as ref_stream_batched
    from drifting_tts.chunked import Chunking as RefChunking

    ref, new = _pair(tmp_path)
    kw = dict(speaker=2, cfg_scale=2.0, temperature=0.3, first=4, pause="punct")
    win = dict(right=6, left=4, chunk=12, crossfade=2, noise=noise)
    for seed, text in enumerate(TEXTS):
        a = list(ref.stream(text, seed=seed, chunked=RefChunking(**win), **kw))
        b = list(new.stream(text, seed=seed, chunked=Chunking(**win), **kw))
        assert [p.numel() for p in a] == [p.numel() for p in b] and torch.equal(torch.cat(a), torch.cat(b))
    if noise == "torch":
        return
    got = []
    for fn, synth, ch in ((ref_stream_batched, ref, RefChunking(**win)), (stream_batched, new, Chunking(**win))):
        out = {i: [] for i in range(len(TEXTS))}
        for rnd in fn(synth, TEXTS, seeds=[5, 6, 7, 8], chunked=ch, **kw):
            for i, piece in rnd:
                out[i].append(piece)
        got.append([torch.cat(out[i]) for i in range(len(TEXTS))])
    assert all(torch.equal(a, b) for a, b in zip(*got))
