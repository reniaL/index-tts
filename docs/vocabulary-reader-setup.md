# Vocabulary reader: stage 1 baseline

Stage 1 prepares version control and preserves the working Windows setup.
Stage 2 is implemented; see [the vocabulary reader guide](vocabulary-reader.md).

- Upstream: https://github.com/index-tts/index-tts
- Base code: `d9e41aac89fd00b3d71497fddb287b7f24613712`
- Development branch: `feature/vocabulary-reader`
- Model: `IndexTeam/IndexTTS-2.5`, revision
  `c39ce5ba981572cb187443877ff559dfb246ce63`
- Runtime baseline: Python 3.11.13, PyTorch 2.8.0+cu128, NumPy 2.2.6,
  Gradio 5.45.0, OpenCV 4.11.0.86, httpx 0.28.1, socksio 1.0.0.
- Windows compatibility changes are committed separately from setup changes.
  Static resolver metadata for optional accelerators does not install them.

## Windows launch

Install dependencies with `uv sync --extra webui`. Download the model as described
in the upstream README, then run `./scripts/start-webui.ps1` in PowerShell. The
launcher uses model version 2.5, local address `127.0.0.1:17860` and `--fp16`.
For this model version, the upstream implementation uses BF16 for that option.
No optional acceleration flags are enabled.

The launcher uses `uv run --no-sync --extra webui` to preserve an installed,
verified environment. Run dependency synchronization explicitly when updating.
`./scripts/start-webui.ps1 -CheckOnly` checks required paths and prints CLI help
without loading the model or starting the server.

An optional, ignored `.index-tts.local.ps1` in the repository root may set
`$indexTtsModelDir`, `$indexTtsVirtualEnv`, and `$indexTtsEnvironmentScript` to
reuse existing local storage. Keep absolute machine paths in that ignored file.
Generated audio and presets live in `outputs/`, which may be a local directory
junction to existing storage. Do not synchronize the same environment from
multiple checkouts concurrently. For a relocated/shared environment, verify
that `indextts.__file__` points into the intended checkout when launched there.

## Public repository hygiene

Commit only selected source files and documentation. Review `git diff --cached`
and the staged filenames before each push. Keep recordings, voice presets,
generated audio, model weights, environments, caches, credentials and local
configuration out of commits. Ignore rules do not remove files already tracked
in upstream history; inspect new commits independently.

Stage 1 validation covers dependency consistency, CUDA availability, import
location and launch behavior. Voice quality and the vocabulary workflow require
later generation and listening checks.
