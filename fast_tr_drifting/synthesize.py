"""Synthesise speech: text -> durations and sampled pitch -> one DriftDiT evaluation (1-NFE) -> Vocos.

:class:`Synthesizer` runs one request: whole sentences (``__call__``), or streamed (``stream``: the vocoder on
windows, and with ``chunked`` the DiT too). ``fast=True`` captures the acoustic model, the prosody predictor and the
vocoder windows as CUDA graphs. Many concurrent requests: :func:`fast_tr_drifting.batched.stream_batched`.
"""

from __future__ import annotations

import argparse
import copy
import random
import time
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path

import torch

from .audio import HOP_LENGTH, SAMPLE_RATE, MelStats
from .chunked import Chunking
from .text import normalize, split_sentences, text_to_ids  # noqa: F401 (split_sentences re-exported)
from .vocoder import Vocoder, load_vocoder
from .voices import DEFAULT_VOICE, VOICE_VOCODERS, VOICES, voice_id

DEFAULT_TEMPERATURE = 0.5  # noise temperature of a checkpoint without a calibrated one (the release stores 0.3)
DEFAULT_CFG = 2.0  # the guidance scale alpha the released model was evaluated with
PER_VOICE = "per-voice"  # vocoder=: each voice with the vocoder it sounds best with (voices.VOICE_VOCODERS)

Pause = float | str | Callable[[str, random.Random], float]
_KEEP = object()  # Synthesizer.variant: argument not given


def pause_arg(value: str) -> float | str:
    """``--pause``: seconds or ``punct[:<jitter>]``."""
    try:
        return float(value)
    except ValueError:
        if value.partition(":")[0] != "punct":
            raise argparse.ArgumentTypeError(f"expected seconds or 'punct', got {value!r}") from None
        return value


def resolve_pause(pause: Pause, speaker_id: int, edge: float | None = None) -> float | Callable[[str, random.Random],
                                                                                              float]:
    """``pause``: seconds, a callable (e.g. a :class:`~fast_tr_drifting.pauses.PausePolicy`), or ``"punct"`` /
    ``"punct:<jitter>"``: the voice's measured policy (``PausePolicy.for_voice``, with the generated sentences' edge
    silence ``edge`` if known)."""
    if not isinstance(pause, str):
        return pause
    name, _, jitter = pause.partition(":")
    if name != "punct":
        raise ValueError(f"pause must be seconds, a callable or 'punct[:<jitter>]', got {pause!r}")
    from .pauses import PausePolicy

    return PausePolicy.for_voice(speaker_id, jitter=float(jitter or 0.0), edge=edge)


def silence(pause: float | Callable[[str, random.Random], float], sentence: str, rng: random.Random) -> torch.Tensor:
    """The silence after ``sentence``: ``pause`` seconds, or what a pause policy picks for it."""
    return torch.zeros(int((pause(sentence, rng) if callable(pause) else pause) * SAMPLE_RATE))


def preferred_temperature(model) -> float:
    """The checkpoint's preferred noise temperature (``calibrate-durations --temperature``), else 0.5."""
    t = getattr(model, "temperature", None)
    return DEFAULT_TEMPERATURE if t is None else float(t)


class Synthesizer:
    """Text -> 24 kHz speech with a drifting TTS model, a prosody source, the Vocos vocoder and a pause rule.

    ``vocoder``: ``"vocos-v2"`` / ``"vocos-v2-balanced"`` (downloaded), a checkpoint path, a loaded
    :class:`~fast_tr_drifting.vocoder.Vocoder`, or ``"per-voice"``: ``vocos-v2`` for the studio voice and
    ``vocos-v2-balanced`` for the male and female voices (:data:`~fast_tr_drifting.voices.VOICE_VOCODERS`).
    ``fast`` (CUDA): the acoustic model, the prosody predictor and the streaming vocoder's windows run as CUDA graphs
    (:mod:`fast_tr_drifting.fast`), captured here (about 2 s); the output is the same as without them. ``compile``
    also fuses the DiT with ``torch.compile`` and ``tf32`` uses TF32 matmuls: both change the output slightly.
    ``prosody``: a stochastic prosody predictor (``"drift"``, downloaded, or a ``train-prosody`` checkpoint) that
    samples the token pitch and, with ``prosody_durations="sampled"``, the durations (``"regressor"``: the model's
    durations, as release v3.2); see :meth:`set_prosody`. ``pause``: the silence between sentences (seconds, a
    callable, or ``"punct"``: by the final punctuation, measured per voice). :meth:`from_pretrained` builds a
    published release."""

    def __init__(self, model_path: str | Path, device: str = "cuda", vocoder: str | Path | Vocoder = "vocos-v2",
                 fast: bool = False, compile: bool = False, tf32: bool = False, prosody=None,
                 prosody_temperature: float | None = None, prosody_durations: str = "sampled", pause: Pause = 0.15,
                 prosody_duration_temperature: float | None = None):
        from .models.tts import load_tts

        self.model, self.cfg, stats = load_tts(model_path, device)
        self.stats = MelStats(stats["mean"], stats["std"])
        self.device = device
        self.default_temperature = preferred_temperature(self.model)
        self.acoustic, self.prosody = None, None
        self.fast, self._graph_opts = fast, {"compile": compile, "tf32": tf32}
        self._vocoders: dict[str, tuple[Vocoder, dict | None]] = {}  # name -> (vocoder, CUDA graphs of its windows)
        self._set_vocoder(vocoder)
        resolve_pause(pause, 0)  # validate
        self.pause = pause
        self.set_prosody(prosody, prosody_temperature, prosody_durations, prosody_duration_temperature)

    @classmethod
    def from_pretrained(cls, release: str | None = None, device: str = "cuda", **kw) -> Synthesizer:
        """A published release of ``Vyvo/drifting-tts-tr`` (:data:`fast_tr_drifting.hub.RELEASES`; default: the
        latest, v3.2): its acoustic model, vocoder (``vocos-v2``), prosody source (the ``drift`` predictor, pitch
        only) and pause rule (``"punct"``), downloaded once. Keyword arguments override them
        (``vocoder="per-voice"``, ``prosody=None``, ``model=<path or Hub file>``) or pass other options
        (``fast=True``, ...)."""
        from .hub import LATEST, RELEASES, hub_file

        release = release or LATEST
        if release not in RELEASES:
            raise ValueError(f"unknown release {release!r}; choose one of {', '.join(RELEASES)}")
        opts = {**RELEASES[release], **kw}
        model = str(opts.pop("model"))
        return cls(model if Path(model).is_file() else hub_file(model, f"release {release}"), device, **opts)

    # --- vocoders -------------------------------------------------------------------------------------------------

    def _load_vocoder(self, spec) -> tuple[Vocoder, dict | None]:
        if isinstance(spec, Vocoder):
            return spec, self._capture_vocoder(spec) if self.fast else None
        key = str(spec)
        if key not in self._vocoders:
            voc = load_vocoder(spec, self.device)
            self._vocoders[key] = (voc, self._capture_vocoder(voc) if self.fast else None)
        return self._vocoders[key]

    def _set_vocoder(self, spec) -> None:
        """The default vocoder and the per-voice ones (``"per-voice"``)."""
        self.voice_vocoders: dict[int, tuple[Vocoder, dict | None]] = {}
        if str(spec) == PER_VOICE:
            for spk, name in VOICE_VOCODERS.items():
                self.voice_vocoders[spk] = self._load_vocoder(name)
            spec = VOICE_VOCODERS[voice_id(DEFAULT_VOICE)]
        self.vocoder, self.vocoder_graphs = self._load_vocoder(spec)

    def vocoder_for(self, speaker: str | int) -> Vocoder:
        """The vocoder of a voice (the default one unless ``vocoder="per-voice"`` maps the voice to another)."""
        return self.voice_vocoders.get(self.speaker_id(speaker), (self.vocoder, None))[0]

    def _graphs_for(self, speaker: str | int) -> dict | None:
        return self.voice_vocoders.get(self.speaker_id(speaker), (None, self.vocoder_graphs))[1]

    def _capture_vocoder(self, voc: Vocoder, first: int = 32, chunk: int = 256) -> dict | None:
        """CUDA graphs of the streaming vocoder's two window sizes, or ``None`` (eager windows) when the capture
        fails."""
        from .fast import stream_vocoder

        graphs = {"pool": torch.cuda.graph_pool_handle()}
        mel = torch.full((1, 100, first + chunk + voc.context), self.stats.mean, device=self.device)
        try:
            with torch.no_grad():
                for _ in stream_vocoder(voc, mel, first=first, chunk=chunk, context=voc.context, graphs=graphs):
                    pass
        except RuntimeError as e:
            warnings.warn(f"vocoder {voc.name!r} runs eagerly: CUDA graph capture failed ({e})", stacklevel=3)
            return None
        return graphs

    # --- prosody --------------------------------------------------------------------------------------------------

    def variant(self, vocoder=_KEEP, prosody=_KEEP, pause: Pause = _KEEP, prosody_temperature: float | None = None,
                prosody_durations: str = "sampled", prosody_duration_temperature: float | None = None) -> Synthesizer:
        """A Synthesizer that shares this one's acoustic model (no second copy in memory) with another vocoder,
        prosody source (:meth:`set_prosody`) or default pause; arguments left out are kept. E.g. the male and female
        voices' vocoder: ``tts.variant(vocoder="vocos-v2-balanced")``."""
        other = copy.copy(self)
        other._vocoders = self._vocoders  # loaded vocoders are shared too
        if vocoder is not _KEEP:
            other._set_vocoder(vocoder)
        if prosody is not _KEEP:
            other.set_prosody(prosody, prosody_temperature, prosody_durations, prosody_duration_temperature)
        if pause is not _KEEP:
            resolve_pause(pause, 0)
            other.pause = pause
        return other

    def set_prosody(self, prosody=None, temperature: float | None = None, durations: str = "sampled",
                    duration_temperature: float | None = None) -> None:
        """Where the durations and token pitch come from. ``prosody``: ``None`` (the model's deterministic
        regressors), a :class:`~fast_tr_drifting.models.prosody_net.ProsodyPredictor`, a name of
        :data:`fast_tr_drifting.hub.PROSODY_MODELS` (``"drift"``) or a ``train-prosody`` checkpoint.

        The predictor samples with noise temperature ``temperature`` (``None``: the checkpoint's preferred one, else
        1); the seed of each call drives it, before the DiT's noise. ``durations="regressor"`` keeps the model's
        durations (and per-voice factors) and samples only the token pitch; otherwise its own per-voice duration
        factors replace the model's. ``duration_temperature``: the noise temperature of the sampled letters'
        durations apart from the pitch's (``None``: the checkpoint's preferred one, else the prosody temperature of
        each call). A voice listed in the predictor's ``rhythm`` table samples its durations as that speaker. With
        ``fast``, a one-pass predictor (``drift`` / ``mse``) runs in the acoustic model's CUDA graphs (recaptured
        here); flow matching runs eagerly."""
        if durations not in ("sampled", "regressor"):
            raise ValueError(f"prosody_durations must be 'sampled' or 'regressor', got {durations!r}")
        self.prosody, self.prosody_temperature = None, temperature
        self.prosody_durations = durations
        self.prosody_duration_temperature = duration_temperature
        if prosody is not None:
            from .models.prosody_net import ProsodyPredictor

            self.prosody = prosody if isinstance(prosody, ProsodyPredictor) else \
                ProsodyPredictor.load(prosody, self.device, tts=self.model)
            if temperature is None:
                self.prosody_temperature = 1.0 if self.prosody.temperature is None else self.prosody.temperature
            if duration_temperature is None:
                self.prosody_duration_temperature = self.prosody.duration_temperature
        if self.fast:
            self._capture_acoustic()

    def _capture_acoustic(self) -> None:
        from .fast import GraphedAcoustic, graphable

        self.acoustic = None
        if not graphable(self.prosody):
            warnings.warn("this prosody predictor runs eagerly: only the vocoder windows use CUDA graphs", stacklevel=3)
            return
        self.acoustic = GraphedAcoustic(self.model, prosody=self.prosody, prosody_durations=self.prosody_durations,
                                        duration_row=self._duration_row(), **self._graph_opts)
        self.acoustic.warmup()

    def _duration_row(self) -> bool:
        """Whether the sampled durations need their own row in the prosody predictor's batch: a duration
        temperature of their own or a voice that borrows another speaker's rhythm."""
        return (self.prosody is not None and self.prosody_durations == "sampled"
                and (self.prosody_duration_temperature is not None or bool(self.prosody.rhythm)))

    def _rhythm(self, speaker: str | int) -> tuple[torch.Tensor, float] | None:
        """The speaker whose rhythm this voice's sampled durations borrow (``ProsodyPredictor.rhythm``) and the ratio
        of the voice's regressor factor to its sampler factor (the sentence's edges keep its regressor durations), or
        ``None``."""
        if self.prosody is None or self.prosody_durations != "sampled":
            return None
        spk_id = self.speaker_id(speaker)
        src = self.prosody.rhythm.get(spk_id)
        if src is None or src == spk_id:
            return None
        own = self.model.duration_scales.get(spk_id, self.model.duration_scale)
        return torch.tensor([src], device=self.device), own / self.prosody.duration_scales.get(spk_id, 1.0)

    # --- speakers -------------------------------------------------------------------------------------------------

    def speaker_id(self, speaker: str | int) -> int:
        """A voice name (``studio`` / ``male`` / ``female``) or a training speaker ID."""
        return voice_id(speaker)

    def _speaker(self, speaker: str | int) -> tuple[torch.Tensor, float]:
        """Speaker id tensor and the voice's duration factor."""
        spk_id = self.speaker_id(speaker)
        if spk_id >= self.model.encoder.spk.num_embeddings:
            raise ValueError(f"speaker {speaker!r} ({spk_id}) is not in this checkpoint")
        if self.prosody is not None and self.prosody_durations == "sampled":  # the sampler's own factors (1: none)
            tempo = self.prosody.duration_scales.get(spk_id, 1.0)
        else:
            tempo = self.model.duration_scales.get(spk_id, self.model.duration_scale)
        return torch.tensor([spk_id], device=self.device), tempo

    # --- the acoustic model ---------------------------------------------------------------------------------------

    def _temperatures(self, prosody_temperature: float | None) -> tuple[float | None, float | None]:
        """This call's prosody temperature and the duration row's temperature (``None``: no row of its own)."""
        pt = self.prosody_temperature if prosody_temperature is None else prosody_temperature
        dt = self.prosody_duration_temperature if self.prosody_durations == "sampled" else None
        if dt is None and self._duration_row():  # the same row layout as the CUDA graphs
            dt = pt
        return pt, dt

    def _mel(self, sentence: str, spk: torch.Tensor, g: torch.Generator, cfg_scale: float, temperature: float,
             length_scale: float, prosody_temperature: float | None = None,
             rhythm: tuple[torch.Tensor, float] | None = None) -> torch.Tensor:
        """One normalised sentence -> normalised mel ``[1, n_mels, T]`` (CUDA graphs when ``fast``).
        ``rhythm``: the speaker whose durations are sampled and the edge factor (:meth:`_rhythm`)."""
        ids = torch.tensor([text_to_ids(sentence, normalized=True)], device=self.device)
        pt, dt = self._temperatures(prosody_temperature)
        rs, edge_scale = (None, None) if rhythm is None else (rhythm[0], length_scale * rhythm[1])
        if self.acoustic is not None:
            return self.acoustic(ids, spk, cfg_scale, temperature, length_scale, generator=g,
                                 prosody_temperature=1.0 if pt is None else pt, duration_temperature=dt,
                                 rhythm=rs, edge_scale=edge_scale)
        durations, pitch = self._sample_prosody(ids, spk, g, pt, dt, length_scale, rs, edge_scale)
        mel, _ = self.model.synthesize(ids, torch.tensor([ids.shape[1]], device=self.device), spk,
                                       cfg_scale=cfg_scale, temperature=temperature, length_scale=length_scale,
                                       generator=g, durations=durations, pitch=pitch)
        return mel

    def _sample_prosody(self, ids, spk, g, pt, dt, length_scale, rs, edge_scale):
        """The prosody predictor's durations (``None`` with ``prosody_durations="regressor"``) and token pitch, drawn
        first from the same generator as the DiT's noise; ``(None, None)`` without a predictor."""
        if self.prosody is None:
            return None, None
        ids_len = torch.tensor([ids.shape[1]], device=self.device)
        durations, pitch = self.prosody.predict(self.model, ids, ids_len, spk, pt, length_scale, generator=g,
                                                duration_temperature=dt, duration_speaker=rs, edge_scale=edge_scale)
        return (None if self.prosody_durations == "regressor" else durations), pitch

    def _prepare(self, sentence: str, spk: torch.Tensor, g: torch.Generator, cfg_scale: float, temperature: float,
                 length_scale: float, prosody_temperature: float | None = None,
                 rhythm: tuple[torch.Tensor, float] | None = None) -> tuple[torch.Tensor, ...]:
        """:meth:`_mel` without the DiT, for :mod:`fast_tr_drifting.chunked`: the noise scaled by ``temperature``
        ``[1, n_mels, T]``, the aligned condition ``[1, C, T]``, the CFG scale ``[1]`` and the style codes, with the
        same draws from ``g`` (CUDA graphs for the encoder when ``fast``)."""
        ids = torch.tensor([text_to_ids(sentence, normalized=True)], device=self.device)
        pt, dt = self._temperatures(prosody_temperature)
        rs, edge_scale = (None, None) if rhythm is None else (rhythm[0], length_scale * rhythm[1])
        if self.acoustic is not None:
            return self.acoustic.prepare(ids, spk, cfg_scale, temperature, length_scale, generator=g,
                                         prosody_temperature=1.0 if pt is None else pt, duration_temperature=dt,
                                         rhythm=rs, edge_scale=edge_scale)
        durations, pitch = self._sample_prosody(ids, spk, g, pt, dt, length_scale, rs, edge_scale)
        ids_len = torch.tensor([ids.shape[1]], device=self.device)
        cond, _ = self.model.aligned_condition(ids, ids_len, spk, length_scale, durations=durations, pitch=pitch)
        z = torch.randn(1, self.model.n_mels, cond.shape[-1], device=self.device, generator=g)
        labels = self.model.style_codes(1, self.device, g)
        return z * temperature, cond, torch.full((1,), float(cfg_scale), device=self.device), labels

    def _prepare_philox(self, sentence: str, spk: torch.Tensor, seed: int, k: int, cfg_scale: float,
                        temperature: float, length_scale: float,
                        prosody_temperature: float | None = None) -> tuple:
        """:meth:`_prepare` with the counter-based noise of :mod:`fast_tr_drifting.noise` (sentence ``k`` of the
        request with seed ``seed``), as :func:`fast_tr_drifting.batched.stream_batched` draws it: the DiT's noise is a
        :class:`~fast_tr_drifting.noise.DiTNoise` (drawn window by window)."""
        from .batched import batchable, prepare_batch
        from .chunked import to_device
        from .noise import DiTNoise, Philox, philox_normal, philox_randint

        if not batchable(self):
            raise NotImplementedError("counter-based noise supports the prosody sources of batched synthesis")
        ids = text_to_ids(sentence, normalized=True)
        keys = Philox([seed], [k])
        pt = self.prosody_temperature if prosody_temperature is None else prosody_temperature
        pt = 1.0 if pt is None else pt
        if self.acoustic is None:  # the batched path's own text pass, one row
            z, cond, _, alpha, labels, _ = prepare_batch(self.model, [ids], int(spk[0]), cfg_scale, temperature,
                                                         length_scale, keys, self.prosody, pt, self.prosody_durations)
            return z, cond, alpha, labels
        seeds, sentences = keys.tensors(self.device)
        if self.prosody is not None and self.prosody.kind == "drift":
            net = self.prosody.net
            noise = (philox_normal(seeds, sentences, "prosody_tok", net.noise_tok, len(ids)),
                     philox_normal(seeds, sentences, "prosody_glob", net.noise_glob, 1)[:, :, 0])
        else:  # no prosody noise to draw: prepare draws nothing when given an empty one
            noise = (torch.zeros(1, 0, len(ids), device=self.device), torch.zeros(1, 0, device=self.device))
        _, cond, alpha, _ = self.acoustic.prepare(torch.tensor([ids], device=self.device), spk, cfg_scale,
                                                  temperature, length_scale, prosody_temperature=pt,
                                                  prosody_noise=noise)
        gen = self.model.generator
        labels = philox_randint(seeds, sentences, "style", max(1, gen.noise_coords), gen.noise_classes)
        z = DiTNoise(keys, self.model.n_mels, to_device([cond.shape[-1]], self.device), temperature, cond.shape[-1])
        return z, cond, alpha, labels

    def _window_generator(self):
        """``generate(z, cond, spk, alpha, mask, labels)`` of :class:`~fast_tr_drifting.chunked.WindowedMel` for one
        request: the acoustic model's CUDA graphs when ``fast``, else the eager rollout."""
        if self.acoustic is not None:
            return self.acoustic.generate_window
        model = self.model

        def generate(z, cond, spk, alpha, mask, labels):
            return model.rollout(z, cond, spk, alpha, model.generator.num_steps, mask=mask, noise_labels=labels)

        return generate

    # --- public API -----------------------------------------------------------------------------------------------

    def _defaults(self, cfg_scale: float | None, temperature: float | None) -> tuple[float, float]:
        return (DEFAULT_CFG if cfg_scale is None else float(cfg_scale),
                self.default_temperature if temperature is None else float(temperature))

    @torch.no_grad()
    def mels(self, text: str, speaker: str | int = DEFAULT_VOICE, cfg_scale: float | None = None,
             temperature: float | None = None, length_scale: float = 1.0, seed: int = 0,
             prosody_temperature: float | None = None) -> list[torch.Tensor]:
        """The unnormalised log-mel ``[1, n_mels, T]`` of each sentence, with the same draws as :meth:`__call__`."""
        cfg_scale, temperature = self._defaults(cfg_scale, temperature)
        spk, tempo = self._speaker(speaker)
        rhythm = self._rhythm(speaker)
        g = torch.Generator(device=self.device).manual_seed(seed)
        return [self.stats.denormalize(self._mel(s, spk, g, cfg_scale, temperature, length_scale * tempo,
                                                 prosody_temperature=prosody_temperature, rhythm=rhythm))
                for s in split_sentences(normalize(text))]

    @torch.no_grad()
    def __call__(self, text: str, speaker: str | int = DEFAULT_VOICE, cfg_scale: float | None = None,
                 temperature: float | None = None, length_scale: float = 1.0, seed: int = 0,
                 pause: Pause | None = None, prosody_temperature: float | None = None) -> tuple[torch.Tensor, dict]:
        """``text`` -> waveform (CPU float tensor at 24 kHz) and timings. Sentence by sentence, each in one pass and
        vocoded whole, joined by ``pause`` (default: the Synthesizer's). ``cfg_scale`` defaults to 2,
        ``temperature`` to the checkpoint's preferred one (0.3 for the release), ``prosody_temperature`` to the
        predictor's (0.5); ``length_scale`` > 1 speaks slower."""
        cfg_scale, temperature = self._defaults(cfg_scale, temperature)
        spk, tempo = self._speaker(speaker)
        rhythm = self._rhythm(speaker)
        pause = self._pause(pause, speaker)
        voc = self.vocoder_for(speaker)
        g = torch.Generator(device=self.device).manual_seed(seed)
        wavs, t_acoustic, t_vocoder = [], 0.0, 0.0
        rng = random.Random(seed)  # pause jitter only: the acoustic draws stay those of ``g``
        for sentence in split_sentences(normalize(text)):
            self._sync()
            t0 = time.perf_counter()
            mel = self._mel(sentence, spk, g, cfg_scale, temperature, length_scale * tempo, prosody_temperature,
                            rhythm)
            self._sync()
            t1 = time.perf_counter()
            wav = voc(self.stats.denormalize(mel))[0].cpu()
            self._sync()
            t_acoustic += t1 - t0
            t_vocoder += time.perf_counter() - t1
            wavs += [wav, silence(pause, sentence, rng)]
        wav = torch.cat(wavs[:-1]) if wavs else torch.zeros(0)
        dur = max(wav.numel() / SAMPLE_RATE, 1e-6)
        return wav, {"seconds": dur, "rtf_acoustic": t_acoustic / dur, "rtf_total": (t_acoustic + t_vocoder) / dur}

    @torch.no_grad()
    def stream(self, text: str, speaker: str | int = DEFAULT_VOICE, cfg_scale: float | None = None,
               temperature: float | None = None, length_scale: float = 1.0, seed: int = 0, pause: Pause | None = None,
               first: int = 32, chunk: int = 256, prosody_temperature: float | None = None,
               chunked: Chunking | bool | None = None) -> Iterator[torch.Tensor]:
        """Yield the waveform in pieces as soon as each is ready (CPU float tensors at 24 kHz).

        Each sentence is generated in one pass. The vocoder then streams: the first ``first`` mel frames (0.34 s),
        then ``chunk``-frame windows, each with 32 frames of context (:func:`fast_tr_drifting.fast.stream_vocoder`);
        the pieces join into the whole-sentence audio.

        ``chunked`` (a :class:`~fast_tr_drifting.chunked.Chunking`, or ``True`` for its defaults): the DiT streams
        too. It runs on windows of the sentence (the first one ``first`` + 32 + ``right`` frames long), so the first
        piece no longer waits for the whole sentence's mel; ``chunk`` is then ``chunked.chunk``. The output is
        another sample of the same model, scored the same (docs/HOW_IT_WORKS.md)."""
        cfg_scale, temperature = self._defaults(cfg_scale, temperature)
        if chunked:
            yield from self._stream_chunked(text, speaker, cfg_scale, temperature, length_scale, seed, pause, first,
                                            prosody_temperature, Chunking() if chunked is True else chunked)
            return
        from .fast import stream_vocoder

        spk, tempo = self._speaker(speaker)
        rhythm = self._rhythm(speaker)
        pause = self._pause(pause, speaker)
        voc, graphs = self.vocoder_for(speaker), self._graphs_for(speaker)
        g = torch.Generator(device=self.device).manual_seed(seed)
        rng, previous = random.Random(seed), None
        for sentence in split_sentences(normalize(text)):
            mel = self.stats.denormalize(self._mel(sentence, spk, g, cfg_scale, temperature, length_scale * tempo,
                                                   prosody_temperature=prosody_temperature, rhythm=rhythm))
            if previous is not None:
                yield silence(pause, previous, rng)
            previous = sentence
            for piece in stream_vocoder(voc, mel, first=first, chunk=chunk, context=voc.context, graphs=graphs):
                yield piece.cpu()

    def _stream_chunked(self, text: str, speaker: str | int, cfg_scale: float, temperature: float,
                        length_scale: float, seed: int, pause: Pause | None, first: int,
                        prosody_temperature: float | None, chunking: Chunking) -> Iterator[torch.Tensor]:
        """:meth:`stream` with the DiT on windows (:mod:`fast_tr_drifting.chunked`): each vocoder window runs as soon
        as the DiT has committed its frames."""
        from .batched import stream_windows
        from .chunked import WindowedMel
        from .fast import vocode_window

        spk, tempo = self._speaker(speaker)
        rhythm = self._rhythm(speaker)
        pause = self._pause(pause, speaker)
        g = torch.Generator(device=self.device).manual_seed(seed)
        rng, previous = random.Random(seed), None
        voc, graphs = self.vocoder_for(speaker), self._graphs_for(speaker)
        ctx = voc.context
        sizes = (first + ctx, chunking.chunk + 2 * ctx)
        generate = self._window_generator()
        for k, sentence in enumerate(split_sentences(normalize(text))):
            if chunking.noise == "philox":
                z, cond, alpha, labels = self._prepare_philox(sentence, spk, seed, k, cfg_scale, temperature,
                                                              length_scale * tempo, prosody_temperature)
            else:
                z, cond, alpha, labels = self._prepare(sentence, spk, g, cfg_scale, temperature,
                                                       length_scale * tempo, prosody_temperature, rhythm)
            t = cond.shape[-1]
            wm = WindowedMel(generate, z, cond, spk, alpha, labels, [t], chunking, first + ctx)
            if previous is not None:
                yield silence(pause, previous, rng)
            previous = sentence
            for a, b, start, n in stream_windows(t, first, chunking.chunk, ctx):
                while wm.committed[0] < b:
                    wm.step([0])
                x = self.stats.denormalize(wm.mel[..., a:b])
                yield vocode_window(voc, x, graphs, sizes)[start * HOP_LENGTH: (start + n) * HOP_LENGTH].cpu()

    def _pause(self, pause: Pause | None, speaker: str | int):
        """The pause of a call: ``"punct"`` uses the prosody predictor's measured edge silence of this voice when its
        durations are sampled (``ProsodyPredictor.pause_edges``), else the table's (the regressors')."""
        spk_id = self.speaker_id(speaker)
        edge = None
        if self.prosody is not None and self.prosody_durations == "sampled":
            edge = self.prosody.pause_edges.get(spk_id)
        return resolve_pause(self.pause if pause is None else pause, spk_id, edge)

    def _sync(self) -> None:
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()


# --- command line ---------------------------------------------------------------------------------------------------

def add_vocoder_args(p: argparse.ArgumentParser, default: str | None = None) -> None:
    from .hub import VOCODERS

    p.add_argument("--vocoder", default=default,
                   help=f"{' | '.join(VOCODERS)} | {PER_VOICE} (vocos-v2-balanced for the male and female voices) or "
                        f"a Vocos checkpoint; default: {default or 'the release' + chr(39) + 's'}")


def pipeline_args(args) -> dict:
    """Synthesizer arguments of ``--release`` / ``--model`` / ``--vocoder`` / ``--prosody`` / ``--pause`` /
    ``--prosody-durations`` (the explicit ones win over the release's)."""
    from .hub import LATEST, RELEASES, hub_file

    release = getattr(args, "release", None) or (None if getattr(args, "model", None) else LATEST)
    rel = RELEASES[release] if release else {}
    model = args.model or hub_file(rel["model"], f"release {release}")
    prosody = args.prosody if getattr(args, "prosody", None) is not None else rel.get("prosody")
    pause = args.pause if getattr(args, "pause", None) is not None else rel.get("pause", 0.15)
    durations = getattr(args, "prosody_durations", None) or rel.get("prosody_durations", "sampled")
    return {"model_path": model, "vocoder": args.vocoder or rel.get("vocoder", "vocos-v2"),
            "prosody": None if str(prosody).lower() == "none" else prosody, "prosody_durations": durations,
            "pause": pause}


def add_args(p: argparse.ArgumentParser) -> None:
    """``fast-tr-drifting say``."""
    from .hub import LATEST, RELEASES

    p.add_argument("text", nargs="?", help="Turkish text (or --text-file)")
    p.add_argument("-o", "--out", default="out.wav", help="output .wav (with --text-file: a directory)")
    p.add_argument("--text-file", help="one utterance per line; writes <out>/<n>.wav")
    p.add_argument("--voice", default=DEFAULT_VOICE, help=f"{' | '.join(VOICES)} or a speaker ID")
    p.add_argument("--release", choices=list(RELEASES), default=None, help=f"default: {LATEST}")
    p.add_argument("--model", default=None, help="an acoustic model checkpoint instead of the release's")
    add_vocoder_args(p)
    p.add_argument("--prosody", default=None, help="'drift' (the release's), a train-prosody checkpoint or 'none'")
    p.add_argument("--prosody-durations", choices=["sampled", "regressor"], default=None,
                   help="sampled: the prosody predictor samples the rhythm too (studio voice only; default: the "
                        "release's, regressor)")
    p.add_argument("--prosody-temperature", type=float, default=None, help="default: the predictor's (0.5)")
    p.add_argument("--prosody-duration-temperature", type=float, default=None,
                   help="with sampled durations: the letters' noise temperature (0.3 keeps the words safe)")
    p.add_argument("--temperature", type=float, default=None, help="noise temperature (default: the model's, 0.3)")
    p.add_argument("--cfg", type=float, default=DEFAULT_CFG, help="guidance scale alpha (learned, free at inference)")
    p.add_argument("--rate", type=float, default=1.0, help="speaking rate (> 1 faster)")
    p.add_argument("--seed", type=int, default=0, help="another seed gives another intonation")
    p.add_argument("--pause", type=pause_arg, default=None, help="seconds or 'punct' (default: the release's)")
    p.add_argument("--stream", action="store_true", help="stream: write the pieces as they come, report the latency")
    p.add_argument("--chunked", action="store_true", help="with --stream: the DiT streams too (frame windows)")
    p.add_argument("--fast", action="store_true", help="CUDA graphs (about 2 s to capture; for repeated calls)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")


def run(args) -> None:
    import soundfile as sf

    if not args.text and not args.text_file:
        raise SystemExit("give a text or --text-file")
    t = time.perf_counter()
    synth = Synthesizer(device=args.device, fast=args.fast and args.device.startswith("cuda"),
                        prosody_temperature=args.prosody_temperature,
                        prosody_duration_temperature=args.prosody_duration_temperature, **pipeline_args(args))
    print(f"loaded in {time.perf_counter() - t:.1f} s ({args.device})", flush=True)
    kw = dict(speaker=args.voice, cfg_scale=args.cfg, temperature=args.temperature, length_scale=1.0 / args.rate,
              seed=args.seed)
    if args.text_file:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        lines = [x.strip() for x in Path(args.text_file).read_text().splitlines() if x.strip()]
        for i, line in enumerate(lines):
            wav, info = synth(line, **kw)
            sf.write(out / f"{i:03d}.wav", wav.numpy(), SAMPLE_RATE)
            print(f"{out / f'{i:03d}.wav'}  {info['seconds']:.2f} s  RTF {info['rtf_total']:.4f}")
        return
    if not args.stream:
        wav, info = synth(args.text, **kw)
        sf.write(args.out, wav.numpy(), SAMPLE_RATE)
        print(f"{args.out}  {info['seconds']:.2f} s of audio, RTF {info['rtf_total']:.4f}")
        return
    synth._sync()
    t0, first, n = time.perf_counter(), None, 0
    with sf.SoundFile(args.out, "w", SAMPLE_RATE, 1, "PCM_16") as f:
        for piece in synth.stream(args.text, chunked=Chunking() if args.chunked else None, **kw):
            if first is None:
                first = time.perf_counter() - t0
            f.write(piece.numpy())
            n += piece.numel()
    total = time.perf_counter() - t0
    note = "" if args.fast else " (a first call: real latencies need --fast and a warm-up, benchmarks/bench_ttfa.py)"
    print(f"{args.out}  {n / SAMPLE_RATE:.2f} s of audio; first audio after {1000 * first:.1f} ms, all after "
          f"{1000 * total:.0f} ms{note}")
