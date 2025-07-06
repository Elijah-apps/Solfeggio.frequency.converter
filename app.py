"""
Solfeggio & Solfège Pitch Shifter Library

A Python library for adjusting pitch and frequency of audio files (MP3/WAV) 
to reproduce songs in different solfège frequencies AND sacred Solfeggio frequencies.

The library supports both:
1. Traditional Musical Solfège (Do, Re, Mi, Fa, Sol, La, Si)
2. Sacred Solfeggio Frequencies (174-963 Hz) - believed to have healing properties

Dependencies:
    pip install librosa soundfile pydub

Usage:
    from solfege_pitch_shifter import SolfeggioShifter
    
    shifter = SolfeggioShifter()
    
    # Traditional solfège
    shifter.shift_to_solfege("input.mp3", "output.wav", target_note="C", octave=4)
    
    # Sacred Solfeggio frequencies
    shifter.shift_to_solfeggio("input.mp3", "output.wav", frequency=528)
"""

import librosa
import soundfile as sf
import numpy as np
from pydub import AudioSegment
import os
from typing import Union, Optional, Tuple, List
import warnings


class SolfeggioShifter:
    """
    A class for shifting audio pitch to different solfège and Solfeggio frequencies.
    
    Supports both traditional musical solfège and sacred Solfeggio frequencies.
    """
    
    # Traditional solfège note frequencies (in Hz) for octave 4
    SOLFEGE_FREQUENCIES = {
        'C': 261.63,   # Do
        'C#': 277.18,  # Do#
        'Db': 277.18,  # Re♭
        'D': 293.66,   # Re
        'D#': 311.13,  # Re#
        'Eb': 311.13,  # Mi♭
        'E': 329.63,   # Mi
        'F': 349.23,   # Fa
        'F#': 369.99,  # Fa#
        'Gb': 369.99,  # Sol♭
        'G': 392.00,   # Sol
        'G#': 415.30,  # Sol#
        'Ab': 415.30,  # La♭
        'A': 440.00,   # La
        'A#': 466.16,  # La#
        'Bb': 466.16,  # Si♭
        'B': 493.88,   # Si
    }
    
    # Sacred Solfeggio frequencies (174-963 Hz) with their properties
    SOLFEGGIO_FREQUENCIES = {
        174: {
            'name': 'Foundation',
            'description': 'Pain relief, natural anaesthetic',
            'properties': 'Reduces physical and emotional pain, promotes healing'
        },
        285: {
            'name': 'Healing',
            'description': 'Tissue regeneration and healing',
            'properties': 'Accelerates healing process, influences energy fields'
        },
        396: {
            'name': 'Liberation',
            'description': 'Liberating guilt and fear',
            'properties': 'Releases fear, guilt, and negative beliefs'
        },
        417: {
            'name': 'Transformation',
            'description': 'Facilitating change and removing negative energy',
            'properties': 'Breaks up negative energy patterns, facilitates change'
        },
        528: {
            'name': 'Love',
            'description': 'Love, DNA repair, miracle tone',
            'properties': 'The "Love Frequency" - promotes love, healing, and DNA repair'
        },
        639: {
            'name': 'Connection',
            'description': 'Harmonious relationships and communication',
            'properties': 'Enhances communication, love, understanding, and tolerance'
        },
        741: {
            'name': 'Expression',
            'description': 'Awakening intuition and problem-solving',
            'properties': 'Awakens intuition, promotes self-expression and solutions'
        },
        852: {
            'name': 'Awakening',
            'description': 'Returning to spiritual order',
            'properties': 'Awakens intuition, returns to spiritual order'
        },
        963: {
            'name': 'Unity',
            'description': 'Divine connection and higher consciousness',
            'properties': 'Connects to divine consciousness, unity with the universe'
        }
    }
    
    # Extended Solfeggio frequencies (some practitioners include these)
    EXTENDED_SOLFEGGIO_FREQUENCIES = {
        111: {'name': 'Cellular Healing', 'description': 'Cell regeneration'},
        222: {'name': 'Harmony', 'description': 'Balance and harmony'},
        333: {'name': 'Ascension', 'description': 'Spiritual elevation'},
        444: {'name': 'Angelic', 'description': 'Angelic guidance'},
        555: {'name': 'Transformation', 'description': 'Personal transformation'},
        666: {'name': 'Material World', 'description': 'Material plane balance'},
        777: {'name': 'Spiritual', 'description': 'Spiritual awakening'},
        888: {'name': 'Abundance', 'description': 'Infinite abundance'},
        999: {'name': 'Universal', 'description': 'Universal connection'}
    }
    
    # Traditional solfège syllables mapping
    SOLFEGE_NAMES = {
        'C': 'Do',
        'D': 'Re',
        'E': 'Mi',
        'F': 'Fa',
        'G': 'Sol',
        'A': 'La',
        'B': 'Si'
    }
    
    def __init__(self, include_extended: bool = False):
        """
        Initialize the pitch shifter.
        
        Args:
            include_extended: Whether to include extended Solfeggio frequencies
        """
        self.supported_formats = ['.mp3', '.wav', '.flac', '.m4a']
        self.include_extended = include_extended
        
        # Combine core and extended frequencies if requested
        self.all_solfeggio_frequencies = self.SOLFEGGIO_FREQUENCIES.copy()
        if include_extended:
            self.all_solfeggio_frequencies.update(self.EXTENDED_SOLFEGGIO_FREQUENCIES)
    
    def _load_audio(self, filepath: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file and return audio data and sample rate.
        
        Args:
            filepath: Path to the audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        file_ext = os.path.splitext(filepath)[1].lower()
        
        if file_ext not in self.supported_formats:
            raise ValueError(f"Unsupported format: {file_ext}")
        
        try:
            # Use librosa for loading (handles most formats)
            audio, sr = librosa.load(filepath, sr=None, mono=False)
            return audio, sr
        except Exception as e:
            raise RuntimeError(f"Error loading audio file: {e}")
    
    def _save_audio(self, audio: np.ndarray, filepath: str, sample_rate: int):
        """
        Save audio data to file.
        
        Args:
            audio: Audio data array
            filepath: Output file path
            sample_rate: Sample rate
        """
        file_ext = os.path.splitext(filepath)[1].lower()
        
        if file_ext == '.mp3':
            # Convert to WAV first, then to MP3 using pydub
            temp_wav = filepath.replace('.mp3', '_temp.wav')
            sf.write(temp_wav, audio.T if audio.ndim > 1 else audio, sample_rate)
            
            # Convert to MP3
            audio_segment = AudioSegment.from_wav(temp_wav)
            audio_segment.export(filepath, format="mp3")
            
            # Clean up temp file
            os.remove(temp_wav)
        else:
            # Direct save for WAV and other formats
            sf.write(filepath, audio.T if audio.ndim > 1 else audio, sample_rate)
    
    def _calculate_pitch_shift(self, target_freq: float, reference_freq: float = 440.0) -> float:
        """
        Calculate the number of semitones to shift.
        
        Args:
            target_freq: Target frequency in Hz
            reference_freq: Reference frequency in Hz (default A4 = 440Hz)
            
        Returns:
            Number of semitones to shift (positive = up, negative = down)
        """
        return 12 * np.log2(target_freq / reference_freq)
    
    def get_solfege_frequency(self, note: str, octave: int = 4) -> float:
        """
        Get frequency for a specific musical note and octave.
        
        Args:
            note: Note name (e.g., 'C', 'C#', 'Db')
            octave: Octave number (default 4)
            
        Returns:
            Frequency in Hz
        """
        if note not in self.SOLFEGE_FREQUENCIES:
            raise ValueError(f"Invalid note: {note}. Valid notes: {list(self.SOLFEGE_FREQUENCIES.keys())}")
        
        base_freq = self.SOLFEGE_FREQUENCIES[note]
        # Adjust for octave (each octave doubles/halves the frequency)
        octave_multiplier = 2 ** (octave - 4)
        return base_freq * octave_multiplier
    
    def get_solfeggio_info(self, frequency: int) -> dict:
        """
        Get information about a specific Solfeggio frequency.
        
        Args:
            frequency: Solfeggio frequency in Hz
            
        Returns:
            Dictionary with frequency information
        """
        if frequency not in self.all_solfeggio_frequencies:
            available_freqs = list(self.all_solfeggio_frequencies.keys())
            raise ValueError(f"Invalid Solfeggio frequency: {frequency}. Available: {available_freqs}")
        
        return self.all_solfeggio_frequencies[frequency]
    
    def list_solfeggio_frequencies(self) -> dict:
        """
        List all available Solfeggio frequencies with their properties.
        
        Returns:
            Dictionary of all Solfeggio frequencies
        """
        return self.all_solfeggio_frequencies
    
    def shift_pitch(self, audio: np.ndarray, sample_rate: int, semitones: float) -> np.ndarray:
        """
        Shift audio pitch by specified number of semitones.
        
        Args:
            audio: Audio data array
            sample_rate: Sample rate
            semitones: Number of semitones to shift
            
        Returns:
            Pitch-shifted audio array
        """
        # Handle stereo audio
        if audio.ndim > 1:
            shifted_channels = []
            for channel in audio:
                shifted_channel = librosa.effects.pitch_shift(
                    channel, sr=sample_rate, n_steps=semitones
                )
                shifted_channels.append(shifted_channel)
            return np.array(shifted_channels)
        else:
            return librosa.effects.pitch_shift(
                audio, sr=sample_rate, n_steps=semitones
            )
    
    def shift_to_frequency(self, input_file: str, output_file: str, 
                          target_freq: float, reference_freq: float = 440.0) -> None:
        """
        Shift audio to a specific target frequency.
        
        Args:
            input_file: Path to input audio file
            output_file: Path to output audio file
            target_freq: Target frequency in Hz
            reference_freq: Reference frequency in Hz (default A4 = 440Hz)
        """
        # Load audio
        audio, sr = self._load_audio(input_file)
        
        # Calculate pitch shift
        semitones = self._calculate_pitch_shift(target_freq, reference_freq)
        
        # Apply pitch shift
        shifted_audio = self.shift_pitch(audio, sr, semitones)
        
        # Save result
        self._save_audio(shifted_audio, output_file, sr)
        
        print(f"Successfully shifted audio by {semitones:.2f} semitones to {target_freq:.2f} Hz")
    
    def shift_to_solfege(self, input_file: str, output_file: str, 
                        target_note: str, octave: int = 4) -> None:
        """
        Shift audio to a specific traditional solfège note.
        
        Args:
            input_file: Path to input audio file
            output_file: Path to output audio file
            target_note: Target note (e.g., 'C', 'D', 'E', etc.)
            octave: Target octave (default 4)
        """
        target_freq = self.get_solfege_frequency(target_note, octave)
        solfege_name = self.SOLFEGE_NAMES.get(target_note.replace('#', '').replace('b', ''), target_note)
        
        self.shift_to_frequency(input_file, output_file, target_freq)
        print(f"Shifted to {solfege_name} ({target_note}{octave}) = {target_freq:.2f} Hz")
    
    def shift_to_solfeggio(self, input_file: str, output_file: str, frequency: int) -> None:
        """
        Shift audio to a specific sacred Solfeggio frequency.
        
        Args:
            input_file: Path to input audio file
            output_file: Path to output audio file
            frequency: Solfeggio frequency in Hz (e.g., 528, 639, 741)
        """
        # Validate frequency
        solfeggio_info = self.get_solfeggio_info(frequency)
        
        # Shift to the frequency
        self.shift_to_frequency(input_file, output_file, float(frequency))
        
        print(f"Shifted to Solfeggio frequency: {frequency} Hz")
        print(f"'{solfeggio_info['name']}' - {solfeggio_info['description']}")
        print(f"Properties: {solfeggio_info['properties']}")
    
    def create_solfege_scale(self, input_file: str, output_dir: str, 
                           start_octave: int = 4, scale_type: str = 'major') -> None:
        """
        Create a complete traditional solfège scale from input audio.
        
        Args:
            input_file: Path to input audio file
            output_dir: Directory to save scale files
            start_octave: Starting octave for the scale
            scale_type: Type of scale ('major' or 'chromatic')
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if scale_type == 'major':
            # Major scale: Do, Re, Mi, Fa, Sol, La, Si
            notes = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
        elif scale_type == 'chromatic':
            # Chromatic scale: all 12 semitones
            notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        else:
            raise ValueError("scale_type must be 'major' or 'chromatic'")
        
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        
        for note in notes:
            solfege_name = self.SOLFEGE_NAMES.get(note.replace('#', ''), note)
            output_file = os.path.join(output_dir, f"{base_name}_{solfege_name}_{note}{start_octave}.wav")
            
            try:
                self.shift_to_solfege(input_file, output_file, note, start_octave)
            except Exception as e:
                print(f"Error processing {note}: {e}")
        
        print(f"Created {scale_type} scale in {output_dir}")
    
    def create_solfeggio_collection(self, input_file: str, output_dir: str, 
                                   frequencies: Optional[List[int]] = None) -> None:
        """
        Create a collection of Solfeggio frequency versions from input audio.
        
        Args:
            input_file: Path to input audio file
            output_dir: Directory to save Solfeggio files
            frequencies: List of specific frequencies to create (defaults to all core frequencies)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if frequencies is None:
            frequencies = list(self.SOLFEGGIO_FREQUENCIES.keys())
        
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        
        for freq in frequencies:
            if freq not in self.all_solfeggio_frequencies:
                print(f"Warning: Skipping unknown frequency {freq} Hz")
                continue
            
            freq_info = self.all_solfeggio_frequencies[freq]
            safe_name = freq_info['name'].replace(' ', '_').replace('/', '_')
            output_file = os.path.join(output_dir, f"{base_name}_{freq}Hz_{safe_name}.wav")
            
            try:
                self.shift_to_solfeggio(input_file, output_file, freq)
            except Exception as e:
                print(f"Error processing {freq} Hz: {e}")
        
        print(f"Created Solfeggio collection in {output_dir}")
    
    def create_healing_frequencies_set(self, input_file: str, output_dir: str) -> None:
        """
        Create a complete set of healing Solfeggio frequencies (174-963 Hz).
        
        Args:
            input_file: Path to input audio file
            output_dir: Directory to save healing frequency files
        """
        healing_frequencies = list(self.SOLFEGGIO_FREQUENCIES.keys())
        self.create_solfeggio_collection(input_file, output_dir, healing_frequencies)
        
        # Create a summary file
        summary_file = os.path.join(output_dir, "solfeggio_frequencies_guide.txt")
        with open(summary_file, 'w') as f:
            f.write("Sacred Solfeggio Frequencies - Healing Properties\n")
            f.write("=" * 50 + "\n\n")
            
            for freq in sorted(healing_frequencies):
                info = self.SOLFEGGIO_FREQUENCIES[freq]
                f.write(f"{freq} Hz - {info['name']}\n")
                f.write(f"Description: {info['description']}\n")
                f.write(f"Properties: {info['properties']}\n")
                f.write("-" * 40 + "\n")
        
        print(f"Created healing frequencies set with guide in {output_dir}")
    
    def batch_process_solfeggio(self, input_dir: str, output_dir: str, 
                               frequency: int) -> None:
        """
        Process multiple audio files to a specific Solfeggio frequency.
        
        Args:
            input_dir: Directory containing input audio files
            output_dir: Directory to save processed files
            frequency: Target Solfeggio frequency
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Validate frequency
        freq_info = self.get_solfeggio_info(frequency)
        
        for filename in os.listdir(input_dir):
            if any(filename.lower().endswith(ext) for ext in self.supported_formats):
                input_file = os.path.join(input_dir, filename)
                base_name = os.path.splitext(filename)[0]
                safe_name = freq_info['name'].replace(' ', '_')
                output_file = os.path.join(output_dir, f"{base_name}_{frequency}Hz_{safe_name}.wav")
                
                try:
                    self.shift_to_solfeggio(input_file, output_file, frequency)
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
        
        print(f"Batch processing to {frequency} Hz completed in {output_dir}")
    
    def batch_process_solfege(self, input_dir: str, output_dir: str, 
                             target_note: str, octave: int = 4) -> None:
        """
        Process multiple audio files to a specific traditional solfège note.
        
        Args:
            input_dir: Directory containing input audio files
            output_dir: Directory to save processed files
            target_note: Target note for all files
            octave: Target octave
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for filename in os.listdir(input_dir):
            if any(filename.lower().endswith(ext) for ext in self.supported_formats):
                input_file = os.path.join(input_dir, filename)
                base_name = os.path.splitext(filename)[0]
                output_file = os.path.join(output_dir, f"{base_name}_{target_note}{octave}.wav")
                
                try:
                    self.shift_to_solfege(input_file, output_file, target_note, octave)
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
        
        print(f"Batch processing to {target_note}{octave} completed in {output_dir}")
    
    def analyze_frequency_impact(self, frequency: int) -> str:
        """
        Get detailed analysis of a Solfeggio frequency's impact on well-being.
        
        Args:
            frequency: Solfeggio frequency in Hz
            
        Returns:
            Detailed analysis string
        """
        if frequency not in self.all_solfeggio_frequencies:
            return f"Unknown frequency: {frequency} Hz"
        
        info = self.all_solfeggio_frequencies[frequency]
        
        analysis = f"""
Solfeggio Frequency Analysis: {frequency} Hz
{'=' * 50}

Name: {info['name']}
Description: {info['description']}
Properties: {info['properties']}

Frequency Range: The Solfeggio frequencies range from 174 Hz to 963 Hz.
This frequency ({frequency} Hz) falls within the sacred melodious spectrum 
that has been shown to have potential impact on well-being.

Usage Recommendations:
- Best used in meditation or relaxation settings
- Can be combined with other wellness practices
- Listen at comfortable volume levels
- Regular exposure may enhance the beneficial effects
        """
        
        return analysis.strip()
    
    def get_audio_info(self, filepath: str) -> dict:
        """
        Get information about an audio file.
        
        Args:
            filepath: Path to audio file
            
        Returns:
            Dictionary with audio information
        """
        audio, sr = self._load_audio(filepath)
        
        return {
            'sample_rate': sr,
            'duration': len(audio) / sr if audio.ndim == 1 else len(audio[0]) / sr,
            'channels': 1 if audio.ndim == 1 else audio.shape[0],
            'samples': len(audio) if audio.ndim == 1 else len(audio[0]),
            'format': os.path.splitext(filepath)[1].lower()
        }
    
    def create_binaural_solfeggio(self, input_file: str, output_file: str, 
                                 base_frequency: int, beat_frequency: float = 10.0) -> None:
        """
        Create binaural beats using Solfeggio frequencies.
        
        Args:
            input_file: Path to input audio file
            output_file: Path to output audio file
            base_frequency: Base Solfeggio frequency
            beat_frequency: Beat frequency in Hz (difference between ears)
        """
        # Load audio
        audio, sr = self._load_audio(input_file)
        
        # Get Solfeggio info
        solfeggio_info = self.get_solfeggio_info(base_frequency)
        
        # Calculate frequencies for left and right channels
        left_freq = float(base_frequency)
        right_freq = float(base_frequency + beat_frequency)
        
        # Shift audio for each channel
        if audio.ndim == 1:
            # Mono audio - create stereo
            left_shifted = self.shift_pitch(audio, sr, 
                                          self._calculate_pitch_shift(left_freq))
            right_shifted = self.shift_pitch(audio, sr, 
                                           self._calculate_pitch_shift(right_freq))
            binaural_audio = np.array([left_shifted, right_shifted])
        else:
            # Stereo audio - modify each channel
            left_shifted = self.shift_pitch(audio[0], sr, 
                                          self._calculate_pitch_shift(left_freq))
            right_shifted = self.shift_pitch(audio[1], sr, 
                                           self._calculate_pitch_shift(right_freq))
            binaural_audio = np.array([left_shifted, right_shifted])
        
        # Save result
        self._save_audio(binaural_audio, output_file, sr)
        
        print(f"Created binaural Solfeggio audio:")
        print(f"Left ear: {left_freq:.2f} Hz")
        print(f"Right ear: {right_freq:.2f} Hz")
        print(f"Beat frequency: {beat_frequency:.2f} Hz")
        print(f"Base frequency: {base_frequency} Hz ({solfeggio_info['name']})")


# Convenience functions for quick access
def quick_shift_to_solfeggio(input_file: str, output_file: str, frequency: int):
    """
    Quick function to shift audio to a specific Solfeggio frequency.
    
    Args:
        input_file: Path to input audio file
        output_file: Path to output audio file
        frequency: Target Solfeggio frequency (e.g., 528, 639, 741)
    """
    shifter = SolfeggioShifter()
    shifter.shift_to_solfeggio(input_file, output_file, frequency)


def quick_shift_to_solfege(input_file: str, output_file: str, note: str, octave: int = 4):
    """
    Quick function to shift audio to a specific traditional solfège note.
    
    Args:
        input_file: Path to input audio file
        output_file: Path to output audio file
        note: Target note (e.g., 'C', 'D', 'E')
        octave: Target octave (default 4)
    """
    shifter = SolfeggioShifter()
    shifter.shift_to_solfege(input_file, output_file, note, octave)


def create_healing_collection(input_file: str, output_dir: str):
    """
    Create a complete collection of healing Solfeggio frequencies.
    
    Args:
        input_file: Path to input audio file
        output_dir: Directory to save healing frequency files
    """
    shifter = SolfeggioShifter()
    shifter.create_healing_frequencies_set(input_file, output_dir)


def create_do_re_mi_scale(input_file: str, output_dir: str):
    """
    Create a Do-Re-Mi scale from input audio.
    
    Args:
        input_file: Path to input audio file
        output_dir: Directory to save scale files
    """
    shifter = SolfeggioShifter()
    shifter.create_solfege_scale(input_file, output_dir, scale_type='major')


def analyze_solfeggio_frequency(frequency: int) -> str:
    """
    Get detailed analysis of a Solfeggio frequency.
    
    Args:
        frequency: Solfeggio frequency in Hz
        
    Returns:
        Detailed analysis string
    """
    shifter = SolfeggioShifter()
    return shifter.analyze_frequency_impact(frequency)


if __name__ == "__main__":
    # Example usage
    shifter = SolfeggioShifter()
    
    print("Solfeggio & Solfège Pitch Shifter Library loaded successfully!")
    print("\n" + "="*60)
    
    # Display available frequencies
    print("TRADITIONAL SOLFÈGE NOTES:")
    for note, freq in shifter.SOLFEGE_FREQUENCIES.items():
        solfege_name = shifter.SOLFEGE_NAMES.get(note.replace('#', '').replace('b', ''), note)
        print(f"  {note} ({solfege_name}): {freq:.2f} Hz")
    
    print("\nSACRED SOLFEGGIO FREQUENCIES (174-963 Hz):")
    for freq in sorted(shifter.SOLFEGGIO_FREQUENCIES.keys()):
        info = shifter.SOLFEGGIO_FREQUENCIES[freq]
        print(f"  {freq} Hz - {info['name']}: {info['description']}")
    
    print("\n" + "="*60)
    print("EXAMPLE USAGE:")
    print("# Shift to 528 Hz (Love frequency)")
    print("shifter.shift_to_solfeggio('song.mp3', 'song_528hz.wav', 528)")
    print("\n# Create complete healing collection")
    print("shifter.create_healing_frequencies_set('song.mp3', 'healing_versions/')")
    print("\n# Traditional solfège")
    print("shifter.shift_to_solfege('song.mp3', 'song_do.wav', 'C', 4)")
    print("\n# Create binaural beats")
    print("shifter.create_binaural_solfeggio('song.mp3', 'binaural.wav', 528, 10.0)")