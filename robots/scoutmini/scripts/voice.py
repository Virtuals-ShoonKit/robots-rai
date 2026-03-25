"""Scout Mini S2S voice pipeline.

Runs LocalWhisper (ASR) + KokoroTTS alongside the main agent.
Listens on the microphone, publishes transcriptions to /from_human,
subscribes to /to_human, and plays back speech via KokoroTTS.

Launch from the robots/scoutmini/ directory so config.toml is found:
    cd robots/scoutmini && python scripts/voice.py
"""

import signal
import sys
import time

from rai.communication.ros2 import ROS2Context
from rai_s2s.asr.agents.initialization import load_config as load_asr_config
from rai_s2s.asr.models import LocalWhisper, SileroVAD
from rai_s2s.s2s.agents.ros2s2s_agent import ROS2S2SAgent
from rai_s2s.sound_device import SoundDeviceConfig
from rai_s2s.tts.agents.initialization import load_config as load_tts_config
from rai_s2s.tts.models import KokoroTTS

SAMPLE_RATE = 16000


def build_microphone_config(device_name: str) -> SoundDeviceConfig:
    return SoundDeviceConfig(
        stream=True,
        channels=1,
        device_name=device_name,
        block_size=1280,
        consumer_sampling_rate=SAMPLE_RATE,
        dtype="int16",
        device_number=None,
        is_input=True,
        is_output=False,
    )


def build_speaker_config(device_name: str) -> SoundDeviceConfig:
    return SoundDeviceConfig(
        stream=True,
        channels=1,
        device_name=device_name,
        block_size=1280,
        consumer_sampling_rate=24000,
        dtype="int16",
        device_number=None,
        is_input=False,
        is_output=True,
    )


@ROS2Context()
def main():
    asr_cfg = load_asr_config()
    tts_cfg = load_tts_config()

    whisper_model = LocalWhisper(
        model_name=asr_cfg.transcribe.model_name,
        sample_rate=SAMPLE_RATE,
        language=asr_cfg.transcribe.language,
    )
    vad = SileroVAD(SAMPLE_RATE, asr_cfg.voice_activity_detection.threshold)
    tts = KokoroTTS(voice=tts_cfg.text_to_speech.voice or "af_sarah")

    mic_config = build_microphone_config(asr_cfg.microphone.device_name)
    spk_config = build_speaker_config(tts_cfg.speaker.device_name)

    agent = ROS2S2SAgent(
        from_human_topic="/from_human",
        to_human_topic="/to_human",
        microphone_config=mic_config,
        speaker_config=spk_config,
        transcription_model=whisper_model,
        vad=vad,
        tts=tts,
        grace_period=asr_cfg.voice_activity_detection.silence_grace_period,
    )
    agent.run()

    def shutdown(signum, frame):
        agent.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
