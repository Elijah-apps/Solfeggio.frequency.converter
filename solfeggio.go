package solfeggio

import (
	"errors"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/go-audio/audio"
	"github.com/go-audio/wav"
	"github.com/youpy/go-flac"
	"github.com/youpy/go-mp3"
	rubigo "github.com/ziutek/rubgo/rubberband"
)

// Solfeggio frequencies (Hz)
var (
	// Original 6 Solfeggio frequencies
	StandardSolfeggioFreqs = map[string]float64{
		"UT":  396.0, // Liberating Guilt and Fear
		"RE":  417.0, // Undoing Situations and Facilitating Change
		"MI":  528.0, // Transformation and DNA Repair (Love frequency)
		"FA":  639.0, // Connecting/Relationships
		"SOL": 741.0, // Awakening Intuition
		"LA":  852.0, // Returning to Spiritual Order
	}

	// Extended Solfeggio frequencies (includes 9 additional frequencies)
	ExtendedSolfeggioFreqs = map[string]float64{
		// Original 6
		"UT":  396.0,
		"RE":  417.0,
		"MI":  528.0,
		"FA":  639.0,
		"SOL": 741.0,
		"LA":  852.0,
		// Extended 9
		"174": 174.0, // Foundation of Conscious Evolution
		"285": 285.0, // Quantum Cognition
		"963": 963.0, // Divine Consciousness
		"1074": 1074.0, // Spiritual Insight
		"1185": 1185.0, // Higher Dimensions
		"1296": 1296.0, // Angelic Realms
		"1407": 1407.0, // Cosmic Consciousness
		"1517": 1517.0, // Universal Love
		"1628": 1628.0, // Universal Harmony
	}

	// Common musical note frequencies for reference
	MusicalNoteFreqs = map[string]float64{
		"C4":  261.63,
		"C#4": 277.18,
		"D4":  293.66,
		"D#4": 311.13,
		"E4":  329.63,
		"F4":  349.23,
		"F#4": 369.99,
		"G4":  392.00,
		"G#4": 415.30,
		"A4":  440.00, // Standard tuning
		"A#4": 466.16,
		"B4":  493.88,
	}
)

// SolfeggioShifter is the main struct for pitch shifting operations
type SolfeggioShifter struct {
	IncludeExtended bool
	rubberBand      *rubigo.RubberBand
	mutex           sync.RWMutex
	sampleRate      int
	channels        int
	initialized     bool
}

// AudioBuffer contains audio data and metadata
type AudioBuffer struct {
	Data       []float32
	SampleRate int
	Channels   int
	Duration   float64 // Duration in seconds
}

// ProcessingOptions contains options for audio processing
type ProcessingOptions struct {
	Quality        int     // 0-4, higher is better quality but slower
	PreserveFormants bool   // Preserve vocal formants
	Smoothing      float64 // 0.0-1.0, amount of smoothing
	BlockSize      int     // Processing block size
}

// DefaultProcessingOptions returns sensible default processing options
func DefaultProcessingOptions() *ProcessingOptions {
	return &ProcessingOptions{
		Quality:        2,
		PreserveFormants: true,
		Smoothing:      0.3,
		BlockSize:      4096,
	}
}

// NewSolfeggioShifter creates a new SolfeggioShifter instance
func NewSolfeggioShifter(includeExtended bool) (*SolfeggioShifter, error) {
	return &SolfeggioShifter{
		IncludeExtended: includeExtended,
		sampleRate:      44100,
		channels:        2,
		initialized:     false,
	}, nil
}

// Initialize sets up the RubberBand instance with specific parameters
func (s *SolfeggioShifter) Initialize(sampleRate, channels int, options *ProcessingOptions) error {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	if s.rubberBand != nil {
		s.rubberBand.Close()
	}

	if options == nil {
		options = DefaultProcessingOptions()
	}

	// RubberBand options
	rbOptions := 0
	if options.Quality >= 3 {
		rbOptions |= rubigo.OptionProcessHighQuality
	}
	if options.PreserveFormants {
		rbOptions |= rubigo.OptionFormantPreserved
	}
	if options.Smoothing > 0.5 {
		rbOptions |= rubigo.OptionSmoothingOn
	}

	rb, err := rubigo.NewRubberBand(sampleRate, channels, rbOptions, 1.0, 1.0)
	if err != nil {
		return fmt.Errorf("failed to initialize RubberBand: %v", err)
	}

	s.rubberBand = rb
	s.sampleRate = sampleRate
	s.channels = channels
	s.initialized = true

	return nil
}

// Close releases resources
func (s *SolfeggioShifter) Close() {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	if s.rubberBand != nil {
		s.rubberBand.Close()
		s.rubberBand = nil
	}
	s.initialized = false
}

// GetAvailableFrequencies returns all available Solfeggio frequencies
func (s *SolfeggioShifter) GetAvailableFrequencies() map[string]float64 {
	if s.IncludeExtended {
		return ExtendedSolfeggioFreqs
	}
	return StandardSolfeggioFreqs
}

// LoadAudio loads an audio file and returns an AudioBuffer
func (s *SolfeggioShifter) LoadAudio(filePath string) (*AudioBuffer, error) {
	ext := strings.ToLower(filepath.Ext(filePath))

	switch ext {
	case ".wav":
		return s.loadWAV(filePath)
	case ".mp3":
		return s.loadMP3(filePath)
	case ".flac":
		return s.loadFLAC(filePath)
	default:
		return nil, fmt.Errorf("unsupported file format: %s", ext)
	}
}

func (s *SolfeggioShifter) loadWAV(path string) (*AudioBuffer, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open WAV file: %v", err)
	}
	defer file.Close()

	decoder := wav.NewDecoder(file)
	if !decoder.IsValidFile() {
		return nil, errors.New("invalid WAV file")
	}

	buf, err := decoder.FullPCMBuffer()
	if err != nil {
		return nil, fmt.Errorf("failed to decode WAV: %v", err)
	}

	sampleRate := int(decoder.SampleRate)
	channels := buf.Format.NumChannels
	duration := float64(len(buf.Data)) / float64(sampleRate*channels)

	return &AudioBuffer{
		Data:       intBufferToFloat32(buf.Data),
		SampleRate: sampleRate,
		Channels:   channels,
		Duration:   duration,
	}, nil
}

func (s *SolfeggioShifter) loadMP3(path string) (*AudioBuffer, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open MP3 file: %v", err)
	}
	defer file.Close()

	decoder, err := mp3.NewDecoder(file)
	if err != nil {
		return nil, fmt.Errorf("failed to create MP3 decoder: %v", err)
	}

	// Read all MP3 data
	var audioData []byte
	buffer := make([]byte, 4096)
	for {
		n, err := decoder.Read(buffer)
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("failed to read MP3 data: %v", err)
		}
		audioData = append(audioData, buffer[:n]...)
	}

	sampleRate := int(decoder.SampleRate())
	channels := 2 // MP3 is typically stereo
	floatData := byteBufferToFloat32(audioData, channels)
	duration := float64(len(floatData)) / float64(sampleRate*channels)

	return &AudioBuffer{
		Data:       floatData,
		SampleRate: sampleRate,
		Channels:   channels,
		Duration:   duration,
	}, nil
}

func (s *SolfeggioShifter) loadFLAC(path string) (*AudioBuffer, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open FLAC file: %v", err)
	}
	defer file.Close()

	decoder := flac.NewDecoder(file)
	if !decoder.IsValidFile() {
		return nil, errors.New("invalid FLAC file")
	}

	var buf []float32
	for {
		frame, err := decoder.ParseNextFrame()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("failed to parse FLAC frame: %v", err)
		}

		// Interleave channels
		for i := 0; i < frame.BlockSize; i++ {
			for ch := 0; ch < frame.Channels; ch++ {
				sample := float32(frame.Subframes[ch].Samples[i]) / 32768.0
				buf = append(buf, sample)
			}
		}
	}

	sampleRate := int(decoder.Info.SampleRate)
	channels := decoder.Info.Channels
	duration := float64(len(buf)) / float64(sampleRate*channels)

	return &AudioBuffer{
		Data:       buf,
		SampleRate: sampleRate,
		Channels:   channels,
		Duration:   duration,
	}, nil
}

// SaveAudio saves an AudioBuffer to a file
func (s *SolfeggioShifter) SaveAudio(buffer *AudioBuffer, path string) error {
	ext := strings.ToLower(filepath.Ext(path))

	switch ext {
	case ".wav":
		return s.saveWAV(buffer, path)
	default:
		return fmt.Errorf("unsupported output format: %s", ext)
	}
}

func (s *SolfeggioShifter) saveWAV(buffer *AudioBuffer, path string) error {
	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("failed to create WAV file: %v", err)
	}
	defer file.Close()

	enc := wav.NewEncoder(file, buffer.SampleRate, 16, buffer.Channels, 1)
	defer enc.Close()

	intBuf := float32BufferToInt(buffer.Data)
	audioBuf := &audio.IntBuffer{
		Data:   intBuf,
		Format: &audio.Format{NumChannels: buffer.Channels, SampleRate: buffer.SampleRate},
	}

	if err := enc.Write(audioBuf); err != nil {
		return fmt.Errorf("failed to write WAV data: %v", err)
	}

	return nil
}

// ShiftPitch performs pitch shifting on an audio buffer
func (s *SolfeggioShifter) ShiftPitch(buffer *AudioBuffer, semitones float64, options *ProcessingOptions) (*AudioBuffer, error) {
	if !s.initialized {
		if err := s.Initialize(buffer.SampleRate, buffer.Channels, options); err != nil {
			return nil, err
		}
	}

	s.mutex.RLock()
	defer s.mutex.RUnlock()

	if s.rubberBand == nil {
		return nil, errors.New("rubberband not initialized")
	}

	// Calculate pitch ratio
	pitchRatio := math.Pow(2.0, semitones/12.0)

	// Configure RubberBand
	s.rubberBand.SetTimeRatio(1.0)
	s.rubberBand.SetPitchScale(pitchRatio)
	s.rubberBand.SetExpectedInputDuration(len(buffer.Data) / buffer.Channels)

	// Process audio in blocks
	if options == nil {
		options = DefaultProcessingOptions()
	}

	blockSize := options.BlockSize
	if blockSize <= 0 {
		blockSize = 4096
	}

	output := make([]float32, 0, len(buffer.Data))

	for i := 0; i < len(buffer.Data); i += blockSize * buffer.Channels {
		end := i + blockSize*buffer.Channels
		if end > len(buffer.Data) {
			end = len(buffer.Data)
		}

		block := buffer.Data[i:end]
		isLast := end == len(buffer.Data)

		s.rubberBand.Process(block, isLast)

		for {
			tempBuf := make([]float32, blockSize*buffer.Channels)
			retrieved := s.rubberBand.Retrieve(tempBuf)
			if retrieved == 0 {
				break
			}
			output = append(output, tempBuf[:retrieved]...)
		}
	}

	return &AudioBuffer{
		Data:       output,
		SampleRate: buffer.SampleRate,
		Channels:   buffer.Channels,
		Duration:   float64(len(output)) / float64(buffer.SampleRate*buffer.Channels),
	}, nil
}

// ShiftToSolfeggio shifts audio to match a specific Solfeggio frequency
func (s *SolfeggioShifter) ShiftToSolfeggio(inputPath, outputPath string, targetFreq float64) error {
	// Load audio
	buffer, err := s.LoadAudio(inputPath)
	if err != nil {
		return fmt.Errorf("failed to load audio: %v", err)
	}

	// Detect fundamental frequency (simplified approach)
	fundamentalFreq := s.detectFundamentalFrequency(buffer)
	if fundamentalFreq <= 0 {
		return errors.New("could not detect fundamental frequency")
	}

	// Calculate semitones to shift
	semitones := 12.0 * math.Log2(targetFreq/fundamentalFreq)

	// Perform pitch shift
	shifted, err := s.ShiftPitch(buffer, semitones, nil)
	if err != nil {
		return fmt.Errorf("failed to shift pitch: %v", err)
	}

	// Save result
	if err := s.SaveAudio(shifted, outputPath); err != nil {
		return fmt.Errorf("failed to save audio: %v", err)
	}

	return nil
}

// ShiftToSolfeggioByName shifts audio to match a named Solfeggio frequency
func (s *SolfeggioShifter) ShiftToSolfeggioByName(inputPath, outputPath, frequencyName string) error {
	freqs := s.GetAvailableFrequencies()
	targetFreq, exists := freqs[strings.ToUpper(frequencyName)]
	if !exists {
		return fmt.Errorf("unknown Solfeggio frequency: %s", frequencyName)
	}

	return s.ShiftToSolfeggio(inputPath, outputPath, targetFreq)
}

// detectFundamentalFrequency uses a simple autocorrelation method to detect fundamental frequency
func (s *SolfeggioShifter) detectFundamentalFrequency(buffer *AudioBuffer) float64 {
	// Convert to mono for analysis
	mono := s.stereoToMono(buffer.Data, buffer.Channels)
	
	// Simple autocorrelation-based pitch detection
	minPeriod := buffer.SampleRate / 800  // 800 Hz max
	maxPeriod := buffer.SampleRate / 80   // 80 Hz min
	
	maxCorr := 0.0
	bestPeriod := 0
	
	for period := minPeriod; period <= maxPeriod; period++ {
		corr := 0.0
		for i := 0; i < len(mono)-period; i++ {
			corr += mono[i] * mono[i+period]
		}
		
		if corr > maxCorr {
			maxCorr = corr
			bestPeriod = period
		}
	}
	
	if bestPeriod > 0 {
		return float64(buffer.SampleRate) / float64(bestPeriod)
	}
	
	return 0
}

// stereoToMono converts stereo audio to mono
func (s *SolfeggioShifter) stereoToMono(data []float32, channels int) []float32 {
	if channels == 1 {
		return data
	}
	
	mono := make([]float32, len(data)/channels)
	for i := 0; i < len(mono); i++ {
		sum := float32(0)
		for ch := 0; ch < channels; ch++ {
			sum += data[i*channels+ch]
		}
		mono[i] = sum / float32(channels)
	}
	return mono
}

// Helper functions for buffer conversion
func intBufferToFloat32(data []int) []float32 {
	result := make([]float32, len(data))
	const maxVal = 32768.0 // For 16-bit audio

	for i, v := range data {
		result[i] = float32(v) / maxVal
	}
	return result
}

func float32BufferToInt(data []float32) []int {
	result := make([]int, len(data))
	const maxVal = 32767.0

	for i, v := range data {
		// Clamp values
		if v > 1.0 {
			v = 1.0
		} else if v < -1.0 {
			v = -1.0
		}
		result[i] = int(v * maxVal)
	}
	return result
}

func byteBufferToFloat32(data []byte, channels int) []float32 {
	result := make([]float32, len(data)/2) // Assuming 16-bit samples
	for i := 0; i < len(data)-1; i += 2 {
		// Little-endian 16-bit sample
		sample := int16(data[i]) | int16(data[i+1])<<8
		result[i/2] = float32(sample) / 32768.0
	}
	return result
}

// BatchProcess processes multiple files with the same settings
func (s *SolfeggioShifter) BatchProcess(inputPaths []string, outputDir string, targetFreq float64, options *ProcessingOptions) error {
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		return fmt.Errorf("failed to create output directory: %v", err)
	}

	for i, inputPath := range inputPaths {
		filename := filepath.Base(inputPath)
		ext := filepath.Ext(filename)
		basename := strings.TrimSuffix(filename, ext)
		outputPath := filepath.Join(outputDir, fmt.Sprintf("%s_solfeggio_%dHz%s", basename, int(targetFreq), ext))

		log.Printf("Processing file %d/%d: %s", i+1, len(inputPaths), filename)

		if err := s.ShiftToSolfeggio(inputPath, outputPath, targetFreq); err != nil {
			log.Printf("Error processing %s: %v", filename, err)
			continue
		}

		log.Printf("Successfully processed: %s", filename)
	}

	return nil
}

// Example usage function
func ExampleUsage() {
	shifter, err := NewSolfeggioShifter(true) // Include extended frequencies
	if err != nil {
		log.Fatalf("Failed to create shifter: %v", err)
	}
	defer shifter.Close()

	// High-quality processing options
	options := &ProcessingOptions{
		Quality:          4,
		PreserveFormants: true,
		Smoothing:        0.5,
		BlockSize:        8192,
	}

	// Initialize with custom options
	if err := shifter.Initialize(44100, 2, options); err != nil {
		log.Fatalf("Failed to initialize shifter: %v", err)
	}

	// Shift to 528Hz (Love frequency)
	if err := shifter.ShiftToSolfeggio("input.wav", "output_528.wav", 528); err != nil {
		log.Fatalf("Failed to shift to 528Hz: %v", err)
	}

	// Shift to MI (528Hz) by name
	if err := shifter.ShiftToSolfeggioByName("input.wav", "output_mi.wav", "MI"); err != nil {
		log.Fatalf("Failed to shift to MI: %v", err)
	}

	// Batch process multiple files
	inputFiles := []string{"song1.wav", "song2.wav", "song3.wav"}
	if err := shifter.BatchProcess(inputFiles, "solfeggio_output", 528, options); err != nil {
		log.Fatalf("Failed to batch process: %v", err)
	}

	fmt.Println("Audio processing complete!")
}