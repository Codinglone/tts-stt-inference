"""
STT Service using ESPnet OWSM v4 small (370 M parameters).

Model page: https://huggingface.co/espnet/owsm_v4_small_370M
"""

import glob
import os
import tempfile
import yaml

import soundfile as sf


class STTService:
    def __init__(self):
        self._model = None
        self._load_model()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        from espnet_model_zoo.downloader import ModelDownloader
        from espnet2.bin.asr_inference import Speech2Text

        print("[STT] Downloading / unpacking OWSM v4 small …")
        d = ModelDownloader()
        kwargs = d.download_and_unpack("espnet/owsm_v4_small_370M")

        # Locate the config file
        config_path = kwargs.get("asr_train_config") or kwargs.get(
            "s2t_train_config"
        )
        if not config_path:
            pattern = os.path.join(d.cachedir, "**", "config.yaml")
            matches = glob.glob(pattern, recursive=True)
            if matches:
                config_path = matches[0]

        # Strip unsupported 'sym_na' keys that cause Speech2Text to crash
        if config_path:
            print(f"[STT] Cleaning config at {config_path} …")
            with open(config_path) as fh:
                config_data = yaml.safe_load(fh)

            _remove_key(config_data, "sym_na")

            fd, tmp_path = tempfile.mkstemp(suffix=".yaml")
            with os.fdopen(fd, "w") as fh:
                yaml.safe_dump(config_data, fh)

            kwargs["asr_train_config"] = tmp_path

        # Normalise key names from model-zoo metadata
        if "s2t_model_file" in kwargs:
            kwargs["asr_model_file"] = kwargs.pop("s2t_model_file")
        kwargs.pop("s2t_train_config", None)

        # Use GPU if available
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[STT] Using device: {device}")

        print("[STT] Initialising Speech2Text …")
        self._model = Speech2Text(
            **kwargs,
            device=device,
            beam_size=1,
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
        speech, _rate = sf.read(audio_path)
        results = self._model(speech)
        if results:
            text, *_ = results[0]
            return text
        return ""


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _remove_key(obj: object, key: str) -> None:
    """Recursively remove *key* from nested dicts/lists."""
    if isinstance(obj, dict):
        obj.pop(key, None)
        for v in obj.values():
            _remove_key(v, key)
    elif isinstance(obj, list):
        for item in obj:
            _remove_key(item, key)
