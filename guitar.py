import sys
import numpy as np
import pygame
import sounddevice as sd

# --- Configuration ---
SAMPLE_RATE = 44100
BLOCK_SIZE = 512  # Low buffer size = low latency (~11ms)
CHANNELS = 1

# Standard Guitar Tuning (6 strings: E2, A2, D3, G3, B3, E4)
GUITAR_STRINGS = [
    82.41,  # 6th: Low E (E2)
    110.00,  # 5th: A2
    146.83,  # 4th: D3
    196.00,  # 3rd: G3
    246.94,  # 2nd: B3
    329.63,  # 1st: High E (E4)
]

# Common Open Chord Fretting offsets: (String 6 down to 1)
# -1 means muted, 0 means open string, N means fret N
CHORDS = {
    pygame.K_e: ("E Maj", [0, 2, 2, 1, 0, 0]),
    pygame.K_a: ("A Maj", [-1, 0, 2, 2, 2, 0]),
    pygame.K_d: ("D Maj", [-1, -1, 0, 2, 3, 2]),
    pygame.K_g: ("G Maj", [3, 2, 0, 0, 0, 3]),
    pygame.K_c: ("C Maj", [-1, 3, 2, 0, 1, 0]),
}


def fret_to_freq(base_freq, fret):
    """Calculates frequency for a given fret (equal temperament: f * 2^(fret/12))"""
    if fret < 0:
        return 0
    return base_freq * (2.0 ** (fret / 12.0))


# --- Sound Synthesizers ---


class KarplusStrongVoice:
    """Simulates a plucked guitar string using Karplus-Strong algorithm"""

    def __init__(self, freq, decay=0.994):
        self.freq = max(20.0, freq)
        self.buffer_len = int(SAMPLE_RATE / self.freq)
        # Seed with initial burst of white noise (pluck)
        self.buffer = (np.random.rand(self.buffer_len) * 2 - 1).astype(
            np.float32
        )
        self.decay = decay
        self.idx = 0
        self.active = True

    def render(self, num_samples):
        out = np.zeros(num_samples, dtype=np.float32)
        if not self.active:
            return out

        for i in range(num_samples):
            # Read current
            current = self.buffer[self.idx]
            next_idx = (self.idx + 1) % self.buffer_len
            # Low-pass filter + decay
            new_val = 0.5 * (current + self.buffer[next_idx]) * self.decay
            self.buffer[self.idx] = new_val

            out[i] = current
            self.idx = next_idx

        # Stop processing voice when energy drops near zero
        if np.max(np.abs(self.buffer)) < 1e-4:
            self.active = False

        return out


class SynthVoice:
    """Basic subtractive/FM or sine-synth voice with ADSR"""

    def __init__(self, freq, wave_type="sine"):
        self.freq = freq
        self.wave_type = wave_type
        self.phase = 0.0
        self.active = True
        self.released = False
        self.env = 0.0

    def render(self, num_samples):
        if not self.active:
            return np.zeros(num_samples, dtype=np.float32)

        t = (
            np.arange(num_samples) * (2 * np.pi * self.freq / SAMPLE_RATE)
            + self.phase
        )
        self.phase = (
            self.phase + num_samples * (2 * np.pi * self.freq / SAMPLE_RATE)
        ) % (2 * np.pi)

        # Wave generation
        if self.wave_type == "sine":
            signal = np.sin(t)
        elif self.wave_type == "saw":
            signal = 2.0 * (t / (2 * np.pi) - np.floor(0.5 + t / (2 * np.pi)))
        else:
            signal = np.sin(t)

        # Simple decay envelope for synth demo
        decay = np.linspace(1.0, 0.98, num_samples, dtype=np.float32)
        signal *= 0.5
        return signal.astype(np.float32)


# --- Voice Manager & Audio Engine ---


class AudioManager:

    def __init__(self):
        self.voices = []
        self.mode = "guitar"  # 'guitar' or 'synth'

    def play_note(self, freq):
        if freq <= 0:
            return
        if self.mode == "guitar":
            self.voices.append(KarplusStrongVoice(freq))
        else:
            self.voices.append(SynthVoice(freq, wave_type="sine"))

    def strum_chord(self, frets, speed_samples=200):
        """Strums 6 strings with a slight human delay between string hits"""
        for i, fret in enumerate(frets):
            if fret >= 0:
                freq = fret_to_freq(GUITAR_STRINGS[i], fret)
                self.play_note(freq)

    def audio_callback(self, outdata, frames, time_info, status):
        mixed = np.zeros(frames, dtype=np.float32)

        # Render all active notes
        alive_voices = []
        for voice in self.voices:
            if voice.active:
                mixed += voice.render(frames)
                alive_voices.append(voice)
        self.voices = alive_voices

        # Soft clipping limiter to prevent harsh digital distortion
        np.clip(mixed, -0.95, 0.95, out=mixed)
        outdata[:, 0] = mixed


# --- Main Application Loop ---


def main():
    pygame.init()
    screen = pygame.display.set_mode((700, 360))
    pygame.display.set_caption("Virtual Guitar & Synth")
    font = pygame.font.SysFont("monospace", 18)
    big_font = pygame.font.SysFont("monospace", 28, bold=True)

    audio = AudioManager()

    # Open the low-latency audio stream
    stream = sd.OutputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        channels=CHANNELS,
        callback=audio.audio_callback,
    )
    stream.start()

    # Key mappings for 6 individual strings (bottom to top: 6th string to 1st)
    STRING_KEYS = {
        pygame.K_1: 0,  # Low E
        pygame.K_2: 1,  # A
        pygame.K_3: 2,  # D
        pygame.K_4: 3,  # G
        pygame.K_5: 4,  # B
        pygame.K_6: 5,  # High E
    }

    current_chord = "None (Open)"
    active_fret_offset = [0, 0, 0, 0, 0, 0]

    running = True
    clock = pygame.time.Clock()

    while running:
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                # Toggle Sound Mode (Tab key)
                if event.key == pygame.K_TAB:
                    audio.mode = "synth" if audio.mode == "guitar" else "guitar"

                # Check if a Chord Modifier is held
                if event.key in CHORDS:
                    name, frets = CHORDS[event.key]
                    current_chord = name
                    active_fret_offset = frets

                # Strum entire chord (Spacebar)
                if event.key == pygame.K_SPACE:
                    audio.strum_chord(active_fret_offset)

                # Pluck individual string (1-6)
                if event.key in STRING_KEYS:
                    string_idx = STRING_KEYS[event.key]
                    fret = active_fret_offset[string_idx]
                    if fret >= 0:
                        freq = fret_to_freq(
                            GUITAR_STRINGS[string_idx], fret
                        )
                        audio.play_note(freq)

            elif event.type == pygame.KEYUP:
                # Reset to open strings when chord key is released
                if event.key in CHORDS:
                    current_chord = "None (Open)"
                    active_fret_offset = [0, 0, 0, 0, 0, 0]

        # Draw UI
        screen.fill((25, 25, 30))
        lines = [
            f"Mode: [ {audio.mode.upper()} ]  (Press TAB to switch)",
            f"Active Chord: {current_chord}",
            "-" * 50,
            "CHORD KEYS (Hold while plucking or press SPACE to strum):",
            "  [E] = E Maj    [A] = A Maj    [D] = D Maj",
            "  [G] = G Maj    [C] = C Maj",
            "-" * 50,
            "STRING PLUCK KEYS:",
            "  [1] E2   [2] A2   [3] D3   [4] G3   [5] B3   [6] E4",
            "-" * 50,
            f"Active Voices Playing: {len(audio.voices)}",
        ]

        for idx, text in enumerate(lines):
            color = (
                (0, 255, 180)
                if "Mode:" in text or "Active Chord:" in text
                else (200, 200, 200)
            )
            surf = font.render(text, True, color)
            screen.blit(surf, (30, 25 + idx * 24))

        pygame.display.flip()

    stream.stop()
    stream.close()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()