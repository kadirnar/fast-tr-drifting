"""The vendored Vocos (fast_tr_drifting.vocoder): checkpoint layout, framing, the inverse STFT and the masked batch.
CPU, tiny models, no downloads."""

import pytest
import torch
import torch.nn.functional as F

from fast_tr_drifting.audio import HOP_LENGTH, N_FFT
from fast_tr_drifting.vocoder import Vocoder, Vocos, load_vocoder, ola_istft, vocode_masked

from .test_batched import _tiny_vocos


def test_from_state_dict_reads_the_sizes_and_skips_the_mel_front_end():
    torch.manual_seed(0)
    src = Vocos(100, 24, 40, 3)
    state = {**src.state_dict(), "feature_extractor.mel_spec.spectrogram.window": torch.ones(4)}
    model = Vocos.from_state_dict(state)
    assert len(model.backbone.convnext) == 3 and model.backbone.embed.out_channels == 24
    assert model.backbone.convnext[0].pwconv1.out_features == 40
    mel = torch.randn(2, 100, 17) - 5
    assert torch.equal(model(mel), src(mel))


def test_vocoder_output_framing_and_range():
    voc = _tiny_vocos()
    wav = voc(torch.randn(3, 100, 21) * 2 - 5)
    assert wav.shape == (3, 21 * HOP_LENGTH) and wav.abs().max() <= 1


def test_ola_istft_inverts_the_bigvgan_framed_stft():
    """Frame i centred on sample i * hop + hop / 2 (reflect-padded uncentred STFT): the inverse gives the signal."""
    torch.manual_seed(0)
    t = 40
    y = torch.randn(2, t * HOP_LENGTH)
    window = torch.hann_window(N_FFT, dtype=torch.float64)
    pad = (N_FFT - HOP_LENGTH) // 2
    yp = F.pad(y.double()[:, None], (pad, pad), mode="reflect")[:, 0]
    spec = torch.stft(yp, N_FFT, HOP_LENGTH, N_FFT, window, center=False, return_complex=True)
    assert spec.shape[-1] == t
    torch.testing.assert_close(ola_istft(spec, window), y.double(), rtol=0, atol=1e-9)


def test_load_vocoder_reads_finetune_checkpoints(tmp_path):
    voc = _tiny_vocos()
    ck = {"vocos": voc.model.state_dict(), "init": "charactr/vocos-mel-24khz", "step": 1, "mel": "bigvgan",
          "head_padding": "same"}
    torch.save(ck, tmp_path / "vocos_ft.pt")
    loaded = load_vocoder(tmp_path / "vocos_ft.pt", "cpu")
    mel = torch.randn(1, 100, 30) - 5
    assert torch.equal(loaded(mel), voc(mel)) and loaded.context == 32
    torch.save({**ck, "head_padding": "center"}, tmp_path / "centred.pt")
    with pytest.raises(ValueError, match="same"):
        load_vocoder(tmp_path / "centred.pt", "cpu")
    with pytest.raises(ValueError, match="unknown vocoder"):
        load_vocoder(tmp_path / "missing.pt", "cpu")


def test_masked_batch_with_one_row_is_the_plain_vocoder():
    voc = _tiny_vocos()
    mel = torch.randn(1, 100, 25) - 5
    torch.testing.assert_close(vocode_masked(voc, mel, torch.tensor([25]))[0], voc(mel)[0], rtol=0, atol=1e-5)


def test_the_vocos_package_gives_the_same_audio():
    """The vendored network against the ``vocos`` package (training extra) with the same weights."""
    pytest.importorskip("vocos")
    from vocos.feature_extractors import MelSpectrogramFeatures
    from vocos.heads import ISTFTHead
    from vocos.models import VocosBackbone
    from vocos.pretrained import Vocos as PackageVocos

    torch.manual_seed(0)
    pkg = PackageVocos(MelSpectrogramFeatures(), VocosBackbone(100, 16, 32, 2), ISTFTHead(16, N_FFT, HOP_LENGTH))
    pkg.head.istft.padding = "same"
    ours = Vocoder(Vocos.from_state_dict(pkg.state_dict()), "cpu")
    mel = torch.randn(2, 100, 33) - 5
    with torch.no_grad():  # the package's ISTFT head, BigVGAN framing: (T - 1) * hop + n_fft - 2 * pad samples
        h = pkg.head.out(pkg.backbone(mel)).transpose(1, 2)
        mag, p = h.chunk(2, dim=1)
        ref = pkg.head.istft(torch.exp(mag).clip(max=1e2) * (torch.cos(p) + 1j * torch.sin(p)))
    torch.testing.assert_close(ours(mel), ref.clamp(-1, 1)[..., : 33 * HOP_LENGTH], rtol=0, atol=1e-5)
