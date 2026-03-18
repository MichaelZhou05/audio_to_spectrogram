#!/usr/bin/env python3
"""
Convert audio files (WAV) to spectrogram PNG images for SqueakOut processing.

This script generates spectrograms from ultrasonic vocalization (USV) recordings
and saves them as 512x512 grayscale PNG images compatible with SqueakOut.

Usage:
    python audio_to_spectrogram.py --input_dir /path/to/wav/files --output_dir /path/to/output

For USV recordings, typical parameters:
    - Sample rate: 250 kHz (common for mouse USV recordings)
    - Frequency range: 20-120 kHz (covers typical mouse USV range)

Dependencies: numpy, Pillow (no scipy required)
"""

import os
import argparse
import struct
import wave
import numpy as np
from PIL import Image
from pathlib import Path


def read_wav(filepath):
    """Read a WAV file and return sample rate and audio data as float32 array."""
    with wave.open(filepath, 'rb') as wav_file:
        sample_rate = wav_file.getframerate()
        n_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        n_frames = wav_file.getnframes()

        raw_data = wav_file.readframes(n_frames)

        if sample_width == 2:  # 16-bit
            fmt = f'<{n_frames * n_channels}h'
            audio_data = np.array(struct.unpack(fmt, raw_data), dtype=np.float32) / 32768.0
        elif sample_width == 4:  # 32-bit
            fmt = f'<{n_frames * n_channels}i'
            audio_data = np.array(struct.unpack(fmt, raw_data), dtype=np.float32) / 2147483648.0
        elif sample_width == 1:  # 8-bit
            audio_data = np.frombuffer(raw_data, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")

        # Handle stereo by taking first channel
        if n_channels > 1:
            audio_data = audio_data[::n_channels]

    return sample_rate, audio_data


def compute_spectrogram(audio_data, sample_rate, nperseg=512, noverlap=384):
    """Compute spectrogram using numpy FFT (no scipy required)."""
    step = nperseg - noverlap

    # Ensure we have enough data
    if len(audio_data) < nperseg:
        audio_data = np.pad(audio_data, (0, nperseg - len(audio_data)))

    n_segments = max(1, (len(audio_data) - nperseg) // step + 1)

    # Hann window
    window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(nperseg) / nperseg))

    # Compute STFT
    spectrogram = np.zeros((nperseg // 2 + 1, n_segments))
    for i in range(n_segments):
        start = i * step
        segment = audio_data[start:start + nperseg] * window
        fft_result = np.fft.rfft(segment)
        spectrogram[:, i] = np.abs(fft_result) ** 2

    # Frequency bins
    frequencies = np.fft.rfftfreq(nperseg, 1.0 / sample_rate)

    return frequencies, spectrogram


def generate_spectrogram(audio_path, output_path,
                         freq_min=20000, freq_max=120000,
                         nperseg=512, noverlap=384,
                         target_size=(512, 512),
                         window_duration=None):
    """
    Generate a spectrogram from an audio file and save as PNG.

    Args:
        audio_path: Path to input WAV file
        output_path: Path for output PNG file
        freq_min: Minimum frequency to display (Hz)
        freq_max: Maximum frequency to display (Hz)
        nperseg: Length of each segment for STFT
        noverlap: Number of points to overlap between segments
        target_size: Output image size (width, height)
        window_duration: If set, process audio in chunks of this duration (seconds)
                        and save multiple spectrograms

    Returns:
        List of output file paths created
    """
    sample_rate, audio_data = read_wav(audio_path)

    output_files = []

    if window_duration is not None:
        # Process in chunks
        samples_per_window = int(window_duration * sample_rate)
        num_windows = int(np.ceil(len(audio_data) / samples_per_window))

        for i in range(num_windows):
            start = i * samples_per_window
            end = min((i + 1) * samples_per_window, len(audio_data))
            chunk = audio_data[start:end]

            # Pad last chunk if needed
            if len(chunk) < samples_per_window:
                chunk = np.pad(chunk, (0, samples_per_window - len(chunk)))

            # Generate output path for this chunk
            base_path = Path(output_path)
            chunk_output = base_path.parent / f"{base_path.stem}_{i:04d}{base_path.suffix}"

            _save_spectrogram(chunk, sample_rate, str(chunk_output),
                            freq_min, freq_max, nperseg, noverlap, target_size)
            output_files.append(str(chunk_output))
    else:
        # Process entire file as one spectrogram
        _save_spectrogram(audio_data, sample_rate, output_path,
                         freq_min, freq_max, nperseg, noverlap, target_size)
        output_files.append(output_path)

    return output_files


def _save_spectrogram(audio_data, sample_rate, output_path,
                      freq_min, freq_max, nperseg, noverlap, target_size):
    """Generate and save a single spectrogram."""
    # Compute spectrogram using numpy
    frequencies, Sxx = compute_spectrogram(audio_data, sample_rate, nperseg, noverlap)

    # Filter to frequency range of interest
    freq_mask = (frequencies >= freq_min) & (frequencies <= freq_max)
    Sxx_filtered = Sxx[freq_mask, :]

    # Handle edge case of empty frequency range
    if Sxx_filtered.size == 0:
        print(f"  Warning: No frequencies in range {freq_min}-{freq_max} Hz. "
              f"Sample rate may be too low. Max freq: {frequencies[-1]:.0f} Hz")
        Sxx_filtered = Sxx

    # Convert to dB scale
    Sxx_db = 10 * np.log10(Sxx_filtered + 1e-10)

    # Normalize to 0-255 for image
    Sxx_normalized = Sxx_db - Sxx_db.min()
    if Sxx_normalized.max() > 0:
        Sxx_normalized = Sxx_normalized / Sxx_normalized.max()
    Sxx_uint8 = (Sxx_normalized * 255).astype(np.uint8)

    # Flip vertically so low frequencies are at bottom
    Sxx_uint8 = np.flipud(Sxx_uint8)

    # Convert to PIL Image and resize
    img = Image.fromarray(Sxx_uint8, mode='L')

    # Use LANCZOS if available, otherwise fall back to ANTIALIAS for older Pillow
    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        resample = Image.LANCZOS if hasattr(Image, 'LANCZOS') else Image.ANTIALIAS

    img = img.resize(target_size, resample)

    # Save
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    img.save(output_path)


def process_directory(input_dir, output_dir, window_duration=1.0, **kwargs):
    """
    Process all WAV files in a directory.

    Args:
        input_dir: Directory containing WAV files
        output_dir: Directory for output PNG files
        window_duration: Duration of each spectrogram window in seconds
        **kwargs: Additional arguments passed to generate_spectrogram

    Returns:
        Dictionary mapping input files to list of output files
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    wav_files = list(input_path.glob('*.wav')) + list(input_path.glob('*.WAV'))

    results = {}
    total_spectrograms = 0

    for wav_file in wav_files:
        print(f"Processing: {wav_file.name}")

        # Create output filename
        output_file = output_path / f"{wav_file.stem}.png"

        try:
            output_files = generate_spectrogram(
                str(wav_file),
                str(output_file),
                window_duration=window_duration,
                **kwargs
            )
            results[str(wav_file)] = output_files
            total_spectrograms += len(output_files)
            print(f"  -> Generated {len(output_files)} spectrogram(s)")
        except Exception as e:
            print(f"  -> ERROR: {e}")
            results[str(wav_file)] = []

    print(f"\nProcessed {len(wav_files)} audio files -> {total_spectrograms} spectrograms")
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Convert audio files to spectrograms for SqueakOut',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input_dir', '-i', required=True,
                        help='Directory containing WAV files')
    parser.add_argument('--output_dir', '-o', required=True,
                        help='Directory for output PNG files')
    parser.add_argument('--freq_min', type=int, default=20000,
                        help='Minimum frequency in Hz')
    parser.add_argument('--freq_max', type=int, default=120000,
                        help='Maximum frequency in Hz')
    parser.add_argument('--window_duration', '-w', type=float, default=1.0,
                        help='Duration of each spectrogram window in seconds. '
                             'Set to 0 to process entire file as single spectrogram.')
    parser.add_argument('--nperseg', type=int, default=512,
                        help='FFT window size (samples)')
    parser.add_argument('--noverlap', type=int, default=384,
                        help='FFT window overlap (samples)')
    parser.add_argument('--target_size', type=int, default=512,
                        help='Output image size (square)')

    args = parser.parse_args()

    window_duration = args.window_duration if args.window_duration > 0 else None

    process_directory(
        args.input_dir,
        args.output_dir,
        freq_min=args.freq_min,
        freq_max=args.freq_max,
        window_duration=window_duration,
        nperseg=args.nperseg,
        noverlap=args.noverlap,
        target_size=(args.target_size, args.target_size)
    )


if __name__ == '__main__':
    main()
