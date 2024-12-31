#!/usr/bin/env python3

import wave
import pyaudio
import time
import vosk
import json
import numpy as np
from pydub import AudioSegment
import rclpy
from rclpy.node import Node

class AudioPlayerNode(Node):
    def __init__(self):
        super().__init__('audio_player_node')

        # Declare parameters
        self.declare_parameter('audio_file', '/home/charmander/catkin_ws/src/follow_me/scripts/spongebobstop.wav')
        self.declare_parameter('repeat_interval', 5)
        self.declare_parameter('vosk_model_path', '/home/charmander/catkin_ws/src/follow_me/scripts/vosk-model-small-en-us-0.15')

        # Get parameters
        self.audio_file = self.get_parameter('audio_file').get_parameter_value().string_value
        self.repeat_interval = self.get_parameter('repeat_interval').get_parameter_value().integer_value
        self.model_path = self.get_parameter('vosk_model_path').get_parameter_value().string_value

        # Load Vosk model
        self.get_logger().info("Loading Vosk model...")
        self.model = vosk.Model(self.model_path)

        # Initialize playback and recognition
        self.stop_command_detected = False

        # Start playback and recognition
        self.timer = self.create_timer(self.repeat_interval, self.play_and_listen)

    def play_audio(self, file_name):
        """Play the specified audio file."""
        wave_file = wave.open(file_name, 'rb')
        p = pyaudio.PyAudio()
        stream = p.open(
            format=p.get_format_from_width(wave_file.getsampwidth()),
            channels=wave_file.getnchannels(),
            rate=wave_file.getframerate(),
            output=True
        )

        chunk = 1024
        data = wave_file.readframes(chunk)
        while data:
            stream.write(data)
            data = wave_file.readframes(chunk)

        stream.stop_stream()
        stream.close()
        p.terminate()
        wave_file.close()

    def filter_noise(self, audio_data, rate):
        """Apply a high-pass filter to remove low-frequency noise."""
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        freq_cutoff = 300
        fft_audio = np.fft.rfft(audio_np)
        frequencies = np.fft.rfftfreq(len(audio_np), d=1/rate)
        fft_audio[frequencies < freq_cutoff] = 0
        filtered_audio_np = np.fft.irfft(fft_audio)
        return filtered_audio_np.astype(np.int16).tobytes()

    def listen_for_stop_command(self):
        """Listen for the 'yes' command using Vosk."""
        self.get_logger().info("Listening for 'yes' to stop...")
        recognizer = vosk.KaldiRecognizer(self.model, 16000)
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8192)
        stream.start_stream()

        start_time = time.time()
        while time.time() - start_time < self.repeat_interval:
            data = stream.read(4000, exception_on_overflow=False)
            filtered_data = self.filter_noise(data, 16000)

            if recognizer.AcceptWaveform(filtered_data):
                result = recognizer.Result()
                result_dict = json.loads(result)
                text = result_dict.get("text", "")
                self.get_logger().info(f"Recognized text: {text}")
                if "yes" in text.lower():
                    self.get_logger().info("Detected 'yes', stopping playback.")
                    self.stop_command_detected = True
                    break

        stream.stop_stream()
        stream.close()
        p.terminate()

    def play_and_listen(self):
        """Play audio and listen for the stop command."""
        if self.stop_command_detected:
            self.get_logger().info("Stop command detected. Ending playback loop.")
            self.timer.cancel()
            return

        self.get_logger().info("Playing audio...")
        self.play_audio(self.audio_file)
        self.listen_for_stop_command()


def main(args=None):
    rclpy.init(args=args)
    node = AudioPlayerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
