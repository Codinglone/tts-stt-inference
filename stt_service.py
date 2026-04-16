"""
STT Service using ESPnet OWSM v4 small (370 M parameters).

Model page: https://huggingface.co/espnet/owsm_v4_small_370M
"""

import numpy as np
import soundfile as sf
import torch.utils.checkpoint  # noqa: F401 — needed for newer PyTorch


class STTService:
    def __init__(self):
        self._model = None
        self._load_model()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        from espnet2.bin.s2t_inference import Speech2Text

        # Use GPU if available
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[STT] Using device: {device}")

        print("[STT] Downloading / unpacking OWSM v4 small …")
        print("[STT] Initialising Speech2Text …")
        self._model = Speech2Text.from_pretrained(
            "espnet/owsm_v4_small_370M",
            device=device,
            beam_size=5,
            ctc_weight=0.3,
            maxlenratio=0.0,
            lang_sym="<kin>",
            task_sym="<asr>",
            predict_time=False,
            nbest=1,
        )
        print("[STT] Model ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe a WAV file and return the recognised text.

        Parameters
        ----------
        audio_path : Path to the WAV audio file.

        Returns
        -------
        Transcribed text string, or an empty string if nothing was recognised.
        """
        import time

        speech, _rate = sf.read(audio_path)
        # Convert stereo to mono if needed
        if speech.ndim > 1:
            speech = np.mean(speech, axis=1)
        print(f"[STT] Audio loaded: {len(speech)} samples, rate={_rate}")

        t0 = time.time()
        results = self._model(speech)
        elapsed = time.time() - t0
        print(f"[STT] Inference took {elapsed:.1f}s")

        if results:
            text, *_ = results[0]
            # Strip special tokens like <kin><asr><notimestamps>
            import re
            text = re.sub(r"<[^>]+>", "", text).strip()
            print(f"[STT] Result: {text!r}")
            return text
        return ""
