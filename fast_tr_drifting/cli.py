"""Command-line entry point: ``fast-tr-drifting <command> [args]``.

Only the chosen command's module is imported, so ``say`` works with the inference dependencies alone; the training
and evaluation commands need the ``train`` / ``eval`` / ``score`` extras.
"""

from __future__ import annotations

import argparse
import importlib
import sys

# command -> (module exposing ``add_args(parser)`` and ``run(args)``, one-line help)
COMMANDS: dict[str, tuple[str, str]] = {
    "say": ("fast_tr_drifting.synthesize", "synthesise speech (streaming with --stream)"),
    "prepare": ("fast_tr_drifting.train.prepare", "build a training set from a Hugging Face parquet dataset"),
    "merge-data": ("fast_tr_drifting.train.merge_data", "merge a prepared voice into a prepared corpus"),
    "score": ("fast_tr_drifting.train.score", "score the training utterances (ASR, DNSMOS, speaker similarity)"),
    "train-mae": ("fast_tr_drifting.train.train_mae", "train the Mel-MAE feature encoder of the drift loss"),
    "train": ("fast_tr_drifting.train.train_tts", "train the acoustic model (text encoder + DriftDiT)"),
    "calibrate-durations": ("fast_tr_drifting.train.calibrate", "per-voice duration factors and the temperature"),
    "prosody-cache": ("fast_tr_drifting.train.prosody_cache", "token targets for the prosody predictor"),
    "train-prosody": ("fast_tr_drifting.train.train_prosody", "train the drift prosody predictor"),
    "finetune-vocoder": ("fast_tr_drifting.train.finetune_vocoder", "fine-tune Vocos on the model's mels"),
    "benchmark": ("fast_tr_drifting.train.benchmark", "Freya-TR-Eval: WER / CER (Whisper large-v3), UTMOSv2"),
    "evaluate": ("fast_tr_drifting.train.evaluate", "held-out evaluation with the judges"),
    "prosody": ("fast_tr_drifting.train.prosody_eval", "prosody metrics against recordings"),
}


def usage() -> str:
    width = max(map(len, COMMANDS))
    rows = "\n".join(f"  {name:{width}s}  {help}" for name, (_, help) in COMMANDS.items())
    return f"usage: fast-tr-drifting <command> [args]\n\ncommands:\n{rows}\n\n'fast-tr-drifting <command> -h' for help"


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(usage())
        return
    if argv[0] not in COMMANDS:
        raise SystemExit(f"unknown command {argv[0]!r}\n\n{usage()}")
    module_name, help = COMMANDS[argv[0]]
    module = importlib.import_module(module_name)
    parser = argparse.ArgumentParser(prog=f"fast-tr-drifting {argv[0]}", description=help)
    module.add_args(parser)
    module.run(parser.parse_args(argv[1:]))


if __name__ == "__main__":
    main()
