#!/usr/bin/env python3
"""
Live Audio Analyzer v4 - Industry-Grade Voice & Singing Detection + Enhanced Beat Detection
Real-time audio analysis for performance and live applications

Version 4 Features:
- Industry-standard voice detection (WebRTC VAD, YIN pitch, LPC formants)
- Professional singing analysis (vibrato detection, voice classification)
- Enhanced beat detection with value persistence
- Multi-band onset detection
- Groove pattern recognition
- Voice-reactive visualizations
"""

import numpy as np
import pygame
import sys
import threading
import queue
import time
import os
import subprocess
import signal
from collections import deque
from typing import Dict, List, Optional, Tuple, Any
import argparse

# Import our industry voice detection module
from industry_voice_detection import IndustryVoiceDetector

# Audio analysis constants
SAMPLE_RATE = 48000
CHUNK_SIZE = 512
BARS_DEFAULT = 512
BARS_MAX = 1024
MAX_FREQ = 20000
FFT_SIZE = 2048

class EnhancedKickDetector:
    """Professional kick/bass drum detection with multi-band analysis and value persistence"""
    
    def __init__(self, sample_rate: int = SAMPLE_RATE, sensitivity: float = 1.0):
        self.sample_rate = sample_rate
        self.sensitivity = sensitivity
        
        # Multi-band frequency ranges (industry standard)
        self.sub_bass_range = (20, 60)      # Sub-bass punch
        self.kick_body_range = (60, 120)    # Kick body/thump
        self.kick_click_range = (2000, 5000) # Kick click/beater
        
        # Spectral flux calculation for each band
        self.prev_magnitude = None
        self.sub_flux_history = deque(maxlen=21)
        self.body_flux_history = deque(maxlen=21)
        self.click_flux_history = deque(maxlen=21)
        
        # Adaptive thresholding
        self.onset_history = deque(maxlen=43)
        self.min_kick_interval = 0.1
        self.last_kick_time = 0
        
        # Detection state with persistence
        self.kick_detected = False
        self.kick_strength = 0.0
        self.kick_velocity = 0.0
        
        # Value persistence for display
        self.display_strength = 0.0
        self.display_velocity = 0
        self.decay_rate = 0.92
        self.hold_time = 0.2
        self.last_detection_time = 0
        
    def calculate_band_flux(self, magnitude: np.ndarray, start_bin: int, end_bin: int, band_type: str) -> float:
        """Calculate spectral flux for specific frequency band"""
        if self.prev_magnitude is None:
            self.prev_magnitude = magnitude.copy()
            return 0.0
        
        # Calculate positive spectral differences in band
        diff = magnitude[start_bin:end_bin] - self.prev_magnitude[start_bin:end_bin]
        positive_flux = np.sum(np.maximum(diff, 0))
        
        # Store in appropriate history
        if band_type == 'sub':
            self.sub_flux_history.append(positive_flux)
        elif band_type == 'body':
            self.body_flux_history.append(positive_flux)
        elif band_type == 'click':
            self.click_flux_history.append(positive_flux)
        
        return positive_flux
    
    def calculate_adaptive_threshold(self, flux_history: deque) -> float:
        """Calculate adaptive threshold using median + scaled MAD"""
        if len(flux_history) < 10:
            return 0.0
        
        flux_array = np.array(list(flux_history))
        median = np.median(flux_array)
        mad = np.median(np.abs(flux_array - median))
        
        return median + self.sensitivity * 2.8 * mad
    
    def detect_kick_onset(self, magnitude: np.ndarray) -> Dict[str, Any]:
        """Enhanced kick detection with multi-band analysis"""
        current_time = time.time()
        nyquist = self.sample_rate / 2
        
        # Calculate flux in each frequency band
        sub_start = int(self.sub_bass_range[0] * len(magnitude) / nyquist)
        sub_end = int(self.sub_bass_range[1] * len(magnitude) / nyquist)
        sub_flux = self.calculate_band_flux(magnitude, sub_start, sub_end, 'sub')
        
        body_start = int(self.kick_body_range[0] * len(magnitude) / nyquist)
        body_end = int(self.kick_body_range[1] * len(magnitude) / nyquist)
        body_flux = self.calculate_band_flux(magnitude, body_start, body_end, 'body')
        
        click_start = int(self.kick_click_range[0] * len(magnitude) / nyquist)
        click_end = int(self.kick_click_range[1] * len(magnitude) / nyquist)
        click_flux = self.calculate_band_flux(magnitude, click_start, click_end, 'click')
        
        self.prev_magnitude = magnitude.copy()
        
        # Calculate adaptive thresholds for each band
        sub_threshold = self.calculate_adaptive_threshold(self.sub_flux_history)
        body_threshold = self.calculate_adaptive_threshold(self.body_flux_history)
        click_threshold = self.calculate_adaptive_threshold(self.click_flux_history)
        
        # Multi-criteria detection
        kick_detected = False
        kick_strength = 0.0
        kick_velocity = 0.0
        
        time_since_last = current_time - self.last_kick_time
        
        if (len(self.sub_flux_history) >= 10 and 
            len(self.body_flux_history) >= 10 and
            time_since_last > self.min_kick_interval):
            
            # Multi-band detection criteria
            sub_hit = sub_flux > sub_threshold
            body_hit = body_flux > body_threshold
            click_present = click_flux > click_threshold * 0.7
            
            # Kick requires strong sub-bass AND body presence
            if sub_hit and body_hit:
                kick_detected = True
                
                # Enhanced strength calculation combining all bands
                sub_strength = sub_flux / (sub_threshold + 1e-6)
                body_strength = body_flux / (body_threshold + 1e-6)
                click_strength = click_flux / (click_threshold + 1e-6) if click_threshold > 0 else 0
                
                # Weighted combination
                kick_strength = min(1.0, (sub_strength * 0.4 + body_strength * 0.5 + click_strength * 0.1))
                kick_velocity = min(127, int(kick_strength * 127))
                
                self.last_kick_time = current_time
                self.last_detection_time = current_time
        
        # Value persistence system
        if kick_detected and kick_strength > 0:
            self.display_strength = kick_strength
            self.display_velocity = kick_velocity
        else:
            # Hold values for specified time, then decay
            time_since_detection = current_time - self.last_detection_time
            if time_since_detection > self.hold_time:
                self.display_strength *= self.decay_rate
                self.display_velocity = int(self.display_velocity * self.decay_rate)
            # Values below threshold are zeroed
            if self.display_strength < 0.05:
                self.display_strength = 0.0
                self.display_velocity = 0
        
        self.kick_detected = kick_detected
        self.kick_strength = kick_strength
        self.kick_velocity = kick_velocity
        
        return {
            'kick_detected': kick_detected,
            'kick_strength': kick_strength,
            'kick_velocity': kick_velocity,
            'display_strength': self.display_strength,
            'display_velocity': self.display_velocity,
            'sub_flux': sub_flux,
            'body_flux': body_flux,
            'click_flux': click_flux,
            'sub_threshold': sub_threshold,
            'body_threshold': body_threshold,
            'multi_band_score': kick_strength
        }

class EnhancedSnareDetector:
    """Professional snare detection with multi-band analysis and value persistence"""
    
    def __init__(self, sample_rate: int = SAMPLE_RATE, sensitivity: float = 1.0):
        self.sample_rate = sample_rate
        self.sensitivity = sensitivity
        
        # Enhanced frequency ranges for snare detection
        self.snare_fundamental_range = (150, 400)
        self.snare_body_range = (400, 1000)
        self.snare_snap_range = (2000, 8000)
        self.snare_rattle_range = (8000, 15000)
        
        # Multi-resolution analysis
        self.prev_magnitude = None
        self.fundamental_flux_history = deque(maxlen=21)
        self.body_flux_history = deque(maxlen=21)
        self.snap_flux_history = deque(maxlen=21)
        self.rattle_flux_history = deque(maxlen=21)
        self.centroid_history = deque(maxlen=21)
        
        # Adaptive thresholding
        self.min_snare_interval = 0.08
        self.last_snare_time = 0
        
        # Detection state with persistence
        self.snare_detected = False
        self.snare_strength = 0.0
        self.snare_velocity = 0.0
        
        # Value persistence for display
        self.display_strength = 0.0
        self.display_velocity = 0
        self.decay_rate = 0.90
        self.hold_time = 0.15
        self.last_detection_time = 0
        
    def calculate_spectral_centroid(self, magnitude: np.ndarray) -> float:
        """Calculate spectral centroid for snare characterization"""
        freqs = np.fft.rfftfreq(len(magnitude) * 2 - 1, 1/self.sample_rate)
        
        # Focus on snare-relevant frequencies
        nyquist = self.sample_rate / 2
        relevant_start = int(self.snare_fundamental_range[0] * len(magnitude) / nyquist)
        relevant_end = int(self.snare_rattle_range[1] * len(magnitude) / nyquist)
        
        relevant_freqs = freqs[relevant_start:relevant_end]
        relevant_magnitude = magnitude[relevant_start:relevant_end]
        
        if np.sum(relevant_magnitude) > 0:
            centroid = np.sum(relevant_freqs * relevant_magnitude) / np.sum(relevant_magnitude)
        else:
            centroid = 0
        
        return centroid
    
    def calculate_multi_band_flux(self, magnitude: np.ndarray) -> Tuple[float, float, float, float]:
        """Calculate spectral flux in all snare frequency ranges"""
        if self.prev_magnitude is None:
            self.prev_magnitude = magnitude.copy()
            return 0.0, 0.0, 0.0, 0.0
        
        nyquist = self.sample_rate / 2
        
        # Fundamental range flux
        fund_start = int(self.snare_fundamental_range[0] * len(magnitude) / nyquist)
        fund_end = int(self.snare_fundamental_range[1] * len(magnitude) / nyquist)
        fund_diff = magnitude[fund_start:fund_end] - self.prev_magnitude[fund_start:fund_end]
        fundamental_flux = np.sum(np.maximum(fund_diff, 0))
        
        # Body range flux
        body_start = int(self.snare_body_range[0] * len(magnitude) / nyquist)
        body_end = int(self.snare_body_range[1] * len(magnitude) / nyquist)
        body_diff = magnitude[body_start:body_end] - self.prev_magnitude[body_start:body_end]
        body_flux = np.sum(np.maximum(body_diff, 0))
        
        # Snap range flux
        snap_start = int(self.snare_snap_range[0] * len(magnitude) / nyquist)
        snap_end = int(self.snare_snap_range[1] * len(magnitude) / nyquist)
        snap_diff = magnitude[snap_start:snap_end] - self.prev_magnitude[snap_start:snap_end]
        snap_flux = np.sum(np.maximum(snap_diff, 0))
        
        # Rattle range flux
        rattle_start = int(self.snare_rattle_range[0] * len(magnitude) / nyquist)
        rattle_end = int(self.snare_rattle_range[1] * len(magnitude) / nyquist)
        rattle_diff = magnitude[rattle_start:rattle_end] - self.prev_magnitude[rattle_start:rattle_end]
        rattle_flux = np.sum(np.maximum(rattle_diff, 0))
        
        self.prev_magnitude = magnitude.copy()
        return fundamental_flux, body_flux, snap_flux, rattle_flux
    
    def detect_snare_onset(self, magnitude: np.ndarray) -> Dict[str, Any]:
        """Enhanced snare detection using multi-band analysis"""
        current_time = time.time()
        
        # Calculate spectral features
        fundamental_flux, body_flux, snap_flux, rattle_flux = self.calculate_multi_band_flux(magnitude)
        centroid = self.calculate_spectral_centroid(magnitude)
        
        # Store history
        self.fundamental_flux_history.append(fundamental_flux)
        self.body_flux_history.append(body_flux)
        self.snap_flux_history.append(snap_flux)
        self.rattle_flux_history.append(rattle_flux)
        self.centroid_history.append(centroid)
        
        # Calculate adaptive thresholds
        snare_detected = False
        snare_strength = 0.0
        snare_velocity = 0.0
        
        if len(self.fundamental_flux_history) >= 10:
            # Calculate thresholds for each band
            fund_array = np.array(list(self.fundamental_flux_history))
            fund_median = np.median(fund_array)
            fund_mad = np.median(np.abs(fund_array - fund_median))
            fund_threshold = fund_median + self.sensitivity * 2.5 * fund_mad
            
            body_array = np.array(list(self.body_flux_history))
            body_median = np.median(body_array)
            body_mad = np.median(np.abs(body_array - body_median))
            body_threshold = body_median + self.sensitivity * 2.3 * body_mad
            
            snap_array = np.array(list(self.snap_flux_history))
            snap_median = np.median(snap_array)
            snap_mad = np.median(np.abs(snap_array - snap_median))
            snap_threshold = snap_median + self.sensitivity * 2.0 * snap_mad
            
            # Spectral centroid in snare range
            centroid_in_range = 800 <= centroid <= 6000
            
            # Time-based gating
            time_since_last = current_time - self.last_snare_time
            
            # Multi-criteria detection
            fundamental_hit = fundamental_flux > fund_threshold
            body_hit = body_flux > body_threshold
            snap_hit = snap_flux > snap_threshold
            timing_ok = time_since_last > self.min_snare_interval
            
            # Enhanced snare detection (requires fundamental + body + snap)
            if fundamental_hit and body_hit and snap_hit and timing_ok and centroid_in_range:
                snare_detected = True
                
                # Multi-band strength calculation
                fund_strength = fundamental_flux / (fund_threshold + 1e-6)
                body_strength = body_flux / (body_threshold + 1e-6)
                snap_strength = snap_flux / (snap_threshold + 1e-6)
                
                # Weighted combination (snap is most characteristic of snare)
                snare_strength = min(1.0, (fund_strength * 0.2 + body_strength * 0.3 + snap_strength * 0.5))
                
                snare_velocity = min(127, int(snare_strength * 127))
                self.last_snare_time = current_time
                self.last_detection_time = current_time
        
        # Value persistence system
        if snare_detected and snare_strength > 0:
            self.display_strength = snare_strength
            self.display_velocity = snare_velocity
        else:
            # Hold values for specified time, then decay
            time_since_detection = current_time - self.last_detection_time
            if time_since_detection > self.hold_time:
                self.display_strength *= self.decay_rate
                self.display_velocity = int(self.display_velocity * self.decay_rate)
            # Values below threshold are zeroed
            if self.display_strength < 0.05:
                self.display_strength = 0.0
                self.display_velocity = 0
        
        self.snare_detected = snare_detected
        self.snare_strength = snare_strength
        self.snare_velocity = snare_velocity
        
        return {
            'snare_detected': snare_detected,
            'snare_strength': snare_strength,
            'snare_velocity': snare_velocity,
            'display_strength': self.display_strength,
            'display_velocity': self.display_velocity,
            'fundamental_flux': fundamental_flux,
            'body_flux': body_flux,
            'snap_flux': snap_flux,
            'rattle_flux': rattle_flux,
            'spectral_centroid': centroid,
            'multi_band_score': snare_strength
        }

class GrooveAnalyzer:
    """Industry-grade groove pattern recognition and tempo analysis"""
    
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.beat_grid = deque(maxlen=64)
        self.tempo_candidates = deque(maxlen=16)
        self.stable_bpm = 0
        self.beat_confidence = 0.0
        self.groove_stability = 0.0
        
        # Extended groove pattern library
        self.groove_patterns = {
            'four_four_basic': [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            'backbeat': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            'shuffle': [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
            'latin_clave': [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],
            'breakbeat': [1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
            'drum_and_bass': [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0],
            'reggae': [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
            'rock_basic': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        }
        
        # Tempo analysis
        self.kick_intervals = deque(maxlen=8)
        self.snare_intervals = deque(maxlen=8)
        self.last_kick_time = 0
        self.last_snare_time = 0
        
        # Pattern matching state
        self.current_pattern = 'unknown'
        self.pattern_confidence = 0.0
        self.pattern_lock_time = 0
        self.pattern_lock_duration = 8.0
        
    def estimate_tempo_from_intervals(self, intervals: List[float]) -> float:
        """Estimate tempo using autocorrelation of inter-onset intervals"""
        if len(intervals) < 3:
            return 0
        
        # Filter reasonable intervals (0.2-2.0 seconds = 30-300 BPM)
        valid_intervals = [i for i in intervals if 0.2 <= i <= 2.0]
        
        if len(valid_intervals) < 2:
            return 0
        
        # Use median for stability
        median_interval = np.median(valid_intervals)
        
        # Convert to BPM
        bpm = 60.0 / median_interval
        
        # Quantize to common BPM values for stability
        common_bpms = [60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180]
        closest_bpm = min(common_bpms, key=lambda x: abs(x - bpm))
        
        # Only use if close enough to a common BPM
        if abs(bpm - closest_bpm) < 8:
            return closest_bpm
        
        return bpm
    
    def match_groove_pattern(self, recent_hits: List[Tuple[float, str]]) -> str:
        """Match recent beat pattern to known groove patterns"""
        if len(recent_hits) < 8:
            return 'insufficient_data'
        
        current_time = time.time()
        
        # Check if pattern is locked
        if (current_time - self.pattern_lock_time) < self.pattern_lock_duration:
            return self.current_pattern
        
        # Build quantized beat grid
        if self.stable_bpm > 60:
            beat_interval = 60.0 / self.stable_bpm
            start_time = recent_hits[0][0]
            
            # Create 16-beat grid
            beat_grid = [0] * 16
            
            for hit_time, hit_type in recent_hits[-16:]:
                relative_time = hit_time - start_time
                beat_position = int((relative_time / beat_interval) % 16)
                
                if 0 <= beat_position < 16:
                    if hit_type in ['kick', 'snare']:
                        beat_grid[beat_position] = max(beat_grid[beat_position], 1)
            
            # Compare with known patterns
            best_match = 'unknown'
            best_score = 0
            
            for pattern_name, pattern in self.groove_patterns.items():
                # Calculate correlation score
                matches = sum(1 for i in range(16) if beat_grid[i] == pattern[i])
                score = matches / 16.0
                
                if score > best_score and score > 0.6:
                    best_score = score
                    best_match = pattern_name
            
            self.pattern_confidence = best_score
            
            # Lock pattern if confidence is high
            if best_score > 0.8:
                self.current_pattern = best_match
                self.pattern_lock_time = current_time
            
            return best_match
        
        return 'no_tempo'
    
    def analyze_groove(self, kick_detected: bool, snare_detected: bool, 
                      kick_strength: float, snare_strength: float) -> Dict[str, Any]:
        """Analyze musical groove pattern in real-time"""
        current_time = time.time()
        
        # Track kick timing
        if kick_detected:
            if self.last_kick_time > 0:
                interval = current_time - self.last_kick_time
                self.kick_intervals.append(interval)
            self.last_kick_time = current_time
            self.beat_grid.append((current_time, 'kick', kick_strength))
        
        # Track snare timing
        if snare_detected:
            if self.last_snare_time > 0:
                interval = current_time - self.last_snare_time
                self.snare_intervals.append(interval)
            self.last_snare_time = current_time
            self.beat_grid.append((current_time, 'snare', snare_strength))
        
        # Tempo estimation
        all_intervals = list(self.kick_intervals) + list(self.snare_intervals)
        
        if len(all_intervals) >= 3:
            tempo_estimate = self.estimate_tempo_from_intervals(all_intervals)
            
            if tempo_estimate > 0:
                self.tempo_candidates.append(tempo_estimate)
            
            # Stable BPM using weighted average of recent estimates
            if len(self.tempo_candidates) >= 4:
                recent_tempos = list(self.tempo_candidates)
                weights = np.exp(np.linspace(-1, 0, len(recent_tempos)))
                self.stable_bpm = np.average(recent_tempos, weights=weights)
                
                # Calculate tempo stability
                tempo_std = np.std(recent_tempos)
                self.groove_stability = max(0, 1.0 - (tempo_std / 20.0))
        
        # Pattern analysis
        recent_hits = [(t, hit_type) for t, hit_type, strength in list(self.beat_grid)[-16:]]
        pattern_match = self.match_groove_pattern(recent_hits)
        
        # Beat confidence based on tempo stability and pattern match
        self.beat_confidence = (self.groove_stability * 0.6 + self.pattern_confidence * 0.4)
        
        return {
            'stable_bpm': round(self.stable_bpm, 1),
            'groove_pattern': pattern_match,
            'pattern_confidence': self.pattern_confidence,
            'beat_confidence': self.beat_confidence,
            'groove_stability': self.groove_stability,
            'tempo_std': np.std(list(self.tempo_candidates)) if len(self.tempo_candidates) > 1 else 0,
            'active_beats': len(self.beat_grid)
        }

class EnhancedDrumDetector:
    """Enhanced drum detection system with all industry improvements"""
    
    def __init__(self, sample_rate: int = SAMPLE_RATE, sensitivity: float = 1.0):
        self.sample_rate = sample_rate
        self.sensitivity = sensitivity
        
        # Initialize enhanced detectors
        self.kick_detector = EnhancedKickDetector(sample_rate, sensitivity)
        self.snare_detector = EnhancedSnareDetector(sample_rate, sensitivity)
        self.groove_analyzer = GrooveAnalyzer(sample_rate)
        
        # Legacy BPM calculation (maintained for compatibility)
        self.kick_times = deque(maxlen=8)
        self.current_bpm = 0
        
        # Enhanced pattern detection
        self.pattern_history = deque(maxlen=32)
        
    def process_audio(self, fft_data: np.ndarray, band_values: np.ndarray) -> Dict[str, Any]:
        """Process audio for enhanced drum detection"""
        current_time = time.time()
        
        # Detect kicks and snares with enhancements
        kick_info = self.kick_detector.detect_kick_onset(fft_data)
        snare_info = self.snare_detector.detect_snare_onset(fft_data)
        
        # Groove analysis
        groove_info = self.groove_analyzer.analyze_groove(
            kick_info['kick_detected'], 
            snare_info['snare_detected'],
            kick_info['kick_strength'],
            snare_info['snare_strength']
        )
        
        # Legacy BPM calculation (for compatibility)
        if kick_info['kick_detected']:
            self.kick_times.append(current_time)
            self.pattern_history.append(('kick', current_time, kick_info['kick_velocity']))
            
            if len(self.kick_times) >= 2:
                recent_intervals = []
                for i in range(1, len(self.kick_times)):
                    interval = self.kick_times[i] - self.kick_times[i-1]
                    if 0.3 < interval < 2.0:
                        recent_intervals.append(interval)
                
                if recent_intervals:
                    avg_interval = np.mean(recent_intervals)
                    self.current_bpm = 60.0 / avg_interval
        
        # Update pattern history for snares
        if snare_info['snare_detected']:
            self.pattern_history.append(('snare', current_time, snare_info['snare_velocity']))
        
        # Detect simultaneous hits
        simultaneous_hit = (kick_info['kick_detected'] and snare_info['snare_detected'])
        
        return {
            'kick': kick_info,
            'snare': snare_info,
            'groove': groove_info,
            'bpm': max(self.current_bpm, groove_info['stable_bpm']),
            'simultaneous_hit': simultaneous_hit,
            'pattern_length': len(self.pattern_history),
            'beat_detected': kick_info['kick_detected'] or snare_info['snare_detected'],
            'beat_strength': max(kick_info['kick_strength'], snare_info['snare_strength'])
        }
    
    def set_sensitivity(self, sensitivity: float):
        """Update sensitivity for both detectors"""
        self.sensitivity = sensitivity
        self.kick_detector.sensitivity = sensitivity
        self.snare_detector.sensitivity = sensitivity

class PipeWireMonitorCapture:
    """Reuse the capture class from output monitor"""
    
    def __init__(self, source_name: str, sample_rate: int = SAMPLE_RATE, chunk_size: int = CHUNK_SIZE):
        self.source_name = source_name
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio_queue = queue.Queue(maxsize=5)
        self.capture_process = None
        self.capture_thread = None
        self.running = False
        
        # Noise gating
        self.noise_floor = 0.001
        self.silence_samples = 0
        self.silence_threshold = sample_rate // 4
        self.background_level = 0.0
        self.background_alpha = 0.001
        
    def list_monitor_sources(self):
        """List available monitor sources"""
        try:
            result = subprocess.run(['pactl', 'list', 'sources', 'short'], 
                                  capture_output=True, text=True)
            sources = []
            
            print("\n🔊 AVAILABLE OUTPUT MONITOR SOURCES:")
            print("="*70)
            
            for line in result.stdout.strip().split('\n'):
                if 'monitor' in line:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        source_id = parts[0]
                        source_name = parts[1]
                        
                        # Categorize sources
                        quality_indicators = []
                        name_lower = source_name.lower()
                        
                        if 'focusrite' in name_lower or 'scarlett' in name_lower:
                            quality_indicators.append("🎧 FOCUSRITE-OUTPUT")
                        elif 'gsx' in name_lower or 'sennheiser' in name_lower:
                            quality_indicators.append("🎮 GSX-OUTPUT")
                        elif 'hdmi' in name_lower:
                            quality_indicators.append("📺 HDMI-OUTPUT")
                        else:
                            quality_indicators.append("🔊 SYSTEM-OUTPUT")
                        
                        # Check if currently active
                        if len(parts) >= 4:
                            state = parts[3] if len(parts) > 3 else "UNKNOWN"
                            if state == "RUNNING":
                                quality_indicators.append("✅ ACTIVE")
                            elif state == "IDLE":
                                quality_indicators.append("💤 IDLE")
                            else:
                                quality_indicators.append("⚪ SUSPENDED")
                        
                        indicators_str = " ".join(quality_indicators)
                        
                        print(f"ID {source_id:3s}: {source_name}")
                        print(f"       {indicators_str}")
                        print()
                        
                        sources.append((source_id, source_name))
            
            print("="*70)
            return sources
            
        except Exception as e:
            print(f"❌ Error listing sources: {e}")
            return []
    
    def select_monitor_source(self):
        """Interactive monitor source selection"""
        sources = self.list_monitor_sources()
        
        if not sources:
            print("❌ No monitor sources found!")
            return None
        
        # Auto-select Focusrite if available
        focusrite_sources = [s for s in sources if 'focusrite' in s[1].lower() or 'scarlett' in s[1].lower()]
        
        if len(focusrite_sources) == 1:
            selected = focusrite_sources[0]
            print(f"🎯 Auto-selected Focusrite: {selected[1]}")
            return selected[1]
        
        # Interactive selection
        print("📋 SELECT MONITOR SOURCE:")
        print("💡 Choose the output device you want to analyze")
        
        if focusrite_sources:
            print(f"🌟 RECOMMENDED: ID {focusrite_sources[0][0]} (Focusrite)")
        
        print(f"\nEnter source ID or press Enter for auto-select:")
        
        try:
            user_input = input("Source ID: ").strip()
            
            if user_input == "":
                if focusrite_sources:
                    selected = focusrite_sources[0]
                    print(f"🎯 Auto-selected: {selected[1]}")
                    return selected[1]
                else:
                    selected = sources[0]
                    print(f"🎯 Using first available: {selected[1]}")
                    return selected[1]
            else:
                source_id = user_input
                for sid, sname in sources:
                    if sid == source_id:
                        print(f"✅ Selected: {sname}")
                        return sname
                
                print(f"❌ Invalid source ID: {source_id}")
                return None
                
        except (KeyboardInterrupt):
            print("❌ Selection cancelled")
            return None
    
    def start_capture(self):
        """Start audio capture from monitor source"""
        if not self.source_name:
            self.source_name = self.select_monitor_source()
        
        if not self.source_name:
            print("❌ No monitor source selected")
            return False
        
        print(f"\n🎵 STARTING VOICE & BEAT ANALYSIS V4:")
        print(f"   Source: {self.source_name}")
        print(f"   Sample Rate: {self.sample_rate}Hz")
        print(f"   Chunk Size: {self.chunk_size} samples")
        
        # Start parec process
        try:
            self.capture_process = subprocess.Popen([
                'parec',
                '--device=' + self.source_name,
                '--format=float32le',
                f'--rate={self.sample_rate}',
                '--channels=1'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.running = True
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            
            print("✅ Voice & beat analysis v4 started!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start audio capture: {e}")
            return False
    
    def _capture_loop(self):
        """Audio capture loop"""
        bytes_per_sample = 4  # float32
        chunk_bytes = self.chunk_size * bytes_per_sample
        
        while self.running and self.capture_process:
            try:
                # Read audio data
                data = self.capture_process.stdout.read(chunk_bytes)
                if not data:
                    break
                
                # Convert to numpy array
                audio_data = np.frombuffer(data, dtype=np.float32)
                
                if len(audio_data) == self.chunk_size:
                    # Apply noise gating
                    rms_level = np.sqrt(np.mean(audio_data ** 2))
                    
                    # Update background noise estimate
                    if rms_level < self.noise_floor * 2:
                        self.background_level = (1 - self.background_alpha) * self.background_level + self.background_alpha * rms_level
                    
                    # Noise gate
                    if rms_level < max(self.noise_floor, self.background_level * 3):
                        self.silence_samples += self.chunk_size
                        if self.silence_samples > self.silence_threshold:
                            # Send silence
                            audio_data = np.zeros_like(audio_data)
                    else:
                        self.silence_samples = 0
                    
                    # Put in queue
                    try:
                        self.audio_queue.put_nowait(audio_data)
                    except queue.Full:
                        # Drop oldest sample
                        try:
                            self.audio_queue.get_nowait()
                            self.audio_queue.put_nowait(audio_data)
                        except:
                            pass
                
            except Exception as e:
                print(f"❌ Capture error: {e}")
                break
    
    def get_audio_data(self):
        """Get latest audio data"""
        try:
            return self.audio_queue.get_nowait()
        except queue.Empty:
            return None
    
    def stop_capture(self):
        """Stop audio capture"""
        self.running = False
        
        if self.capture_process:
            try:
                self.capture_process.terminate()
                self.capture_process.wait(timeout=2)
            except:
                try:
                    self.capture_process.kill()
                except:
                    pass
            self.capture_process = None
        
        if self.capture_thread:
            self.capture_thread.join(timeout=1)

class VoiceReactiveLiveAudioAnalyzer:
    """Enhanced live audio analyzer with industry-grade voice and beat detection"""
    
    def __init__(self, width=2000, height=900, bars=BARS_DEFAULT, source_name=None):
        self.width = width
        self.height = height
        self.bars = bars
        
        # Initialize all components
        self.capture = PipeWireMonitorCapture(source_name)
        self.drum_detector = EnhancedDrumDetector()
        self.voice_detector = IndustryVoiceDetector(SAMPLE_RATE)  # Industry-grade voice detection
        
        # Audio analysis
        self.window = np.hanning(FFT_SIZE).astype(np.float32)
        self.audio_buffer = np.zeros(FFT_SIZE, dtype=np.float32)
        self.ring_buffer = np.zeros(FFT_SIZE * 2, dtype=np.float32)
        self.buffer_pos = 0
        
        # Frequency mapping
        self.freqs = np.fft.rfftfreq(FFT_SIZE, 1/SAMPLE_RATE)
        self.band_indices = self._create_band_mapping()
        
        # Pygame setup with enhanced UI
        pygame.init()
        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        pygame.display.set_caption("Voice & Beat Analyzer v4 - Industry-Grade Detection")
        self.clock = pygame.time.Clock()
        
        # Enhanced fonts
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 24)
        self.font_tiny = pygame.font.Font(None, 20)
        
        # Visualization state
        self.bar_heights = np.zeros(bars, dtype=np.float32)
        self.colors = self._generate_colors()
        
        # Performance tracking
        self.last_process_time = 0
        self.fps_counter = deque(maxlen=30)
        
        # Enhanced drum visualization with persistence
        self.kick_flash_time = 0
        self.snare_flash_time = 0
        self.kick_flash_duration = 0.20
        self.snare_flash_duration = 0.15
        
        # Voice visualization
        self.voice_flash_time = 0
        self.voice_flash_duration = 0.25
        self.singing_flash_time = 0
        self.singing_flash_duration = 0.30
        
        # UI state
        self.show_advanced_info = False
        self.show_groove_info = True
        self.show_voice_info = True
        self.show_formants = False
        self.debug_spectrum_snapshot = False
        
        # Auto debug output
        self.auto_debug_enabled = True
        self.last_auto_debug_time = 0
        self.auto_debug_interval = 3.0  # seconds
        
        print(f"\n🎵 Voice & Beat Analyzer v4")
        print(f"Resolution: {width}x{height}, Bars: {bars}")
        print("Industry-Grade Features:")
        print("  🎤 Voice detection (WebRTC VAD, YIN pitch, LPC formants)")
        print("  🎵 Singing analysis (vibrato detection, voice classification)")
        print("  🥁 Multi-band beat detection with value persistence")
        print("  🎼 Groove pattern recognition")
        print("  📊 Voice-reactive visualizations")
        print("Enhanced Controls:")
        print("  ESC: Quit")
        print("  S: Save screenshot")
        print("  B: Toggle drum sensitivity")
        print("  V: Toggle voice info")
        print("  F: Toggle formant display")
        print("  A: Toggle advanced info")
        print("  G: Toggle groove info")
        print("Window Sizes:")
        print("  1: Compact (1200x720)")
        print("  2: Standard (1600x900)")
        print("  3: Wide (2000x900) - Default")
        print("  4: Extra Wide (2400x1080)")
        print("  5: Ultra Wide (2880x1200)")
        print("  6: Professional (3200x1440)")
        print("Debug Controls:")
        print("  D: Debug spectrum snapshot to terminal")
        print("  T: Toggle auto debug output (every 3s)")
        print("  K/N: Kick/Snare debug")
        print("  P: Pitch detection debug")
    
    def _create_band_mapping(self):
        """Create frequency band mapping with enhanced low-end resolution"""
        freq_min = 20
        freq_max = min(MAX_FREQ, self.freqs[-1])
        
        # Enhanced allocation with 512 total bars for maximum low-end detail
        # Keep same absolute resolution for mids/highs, double the low-end resolution
        low_end_bars = int(self.bars * 0.6)    # 60% for low end (20-500 Hz) - maximum resolution
        mid_bars = int(self.bars * 0.3)        # 30% for mids/voice (500-5000 Hz) - maintains ~154 bars
        high_bars = self.bars - low_end_bars - mid_bars  # 10% for highs (5000+ Hz) - maintains ~52 bars
        
        # Create frequency points with enhanced low-end detail
        # Sub-bass and bass get linear spacing for maximum resolution
        sub_bass_bars = int(low_end_bars * 0.4)  # 40% of low-end for sub-bass (20-80 Hz)
        bass_bars = low_end_bars - sub_bass_bars  # 60% for bass (80-500 Hz)
        
        # High-resolution sub-bass (20-80 Hz) - linear for maximum detail
        sub_bass_freqs = np.linspace(freq_min, 80, sub_bass_bars + 1)
        
        # High-resolution bass (80-500 Hz) - slightly compressed linear
        bass_freqs = np.linspace(80, 500, bass_bars + 1)
        
        # Logarithmic spacing for mids/voice and highs to maintain perceptual balance
        mid_freqs = np.logspace(np.log10(500), np.log10(5000), mid_bars + 1)
        high_freqs = np.logspace(np.log10(5000), np.log10(freq_max), high_bars + 1)
        
        # Combine all frequency ranges
        all_freqs = np.concatenate([
            sub_bass_freqs[:-1],  # 20-80 Hz (high resolution)
            bass_freqs[:-1],      # 80-500 Hz (high resolution) 
            mid_freqs[:-1],       # 500-5000 Hz (logarithmic)
            high_freqs            # 5000+ Hz (logarithmic)
        ])
        
        band_indices = []
        for i in range(len(all_freqs) - 1):
            low_freq = all_freqs[i]
            high_freq = all_freqs[i + 1] if i < len(all_freqs) - 1 else freq_max
            
            # Find FFT bins in this range
            indices = np.where((self.freqs >= low_freq) & (self.freqs <= high_freq))[0]
            if len(indices) == 0:
                # Find nearest bin
                nearest_idx = np.argmin(np.abs(self.freqs - (low_freq + high_freq) / 2))
                indices = np.array([nearest_idx])
            
            band_indices.append(indices)
        
        return band_indices
    
    def _generate_colors(self):
        """Generate enhanced color gradient matching the new frequency allocation"""
        colors = []
        for i in range(self.bars):
            hue = i / self.bars
            
            # Enhanced color scheme for 512 bars with 60/30/10 allocation
            if hue < 0.6:  # Low-end region (20-500 Hz) - maximum resolution with warm colors
                progress = hue / 0.6
                if hue < 0.24:  # Sub-bass (20-80 Hz) - deep reds (40% of 60%)
                    sub_progress = hue / 0.24
                    r = 255
                    g = int(sub_progress * 60)
                    b = int(sub_progress * 30)
                else:  # Bass (80-500 Hz) - red to orange (60% of 60%)
                    bass_progress = (hue - 0.24) / 0.36
                    r = 255
                    g = int(60 + bass_progress * 120)
                    b = int(30 + bass_progress * 50)
            elif hue < 0.9:  # Mid/voice region (500-5000 Hz) - bright, varied colors
                progress = (hue - 0.6) / 0.3
                r = int((1 - progress) * 255 + progress * 120)
                g = 255
                b = int(progress * 255)
            else:  # High region (5000+ Hz) - cool colors
                progress = (hue - 0.9) / 0.1
                r = int(progress * 180)
                g = int((1 - progress) * 255 + progress * 220)
                b = 255
            
            colors.append((max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))))
        
        return colors
    
    def process_audio_chunk(self, audio_data: np.ndarray) -> np.ndarray:
        """Process audio chunk and return FFT magnitude"""
        chunk_len = len(audio_data)
        
        # Write to ring buffer
        if self.buffer_pos + chunk_len <= len(self.ring_buffer):
            self.ring_buffer[self.buffer_pos:self.buffer_pos + chunk_len] = audio_data
        else:
            first_part = len(self.ring_buffer) - self.buffer_pos
            self.ring_buffer[self.buffer_pos:] = audio_data[:first_part]
            self.ring_buffer[:chunk_len - first_part] = audio_data[first_part:]
        
        self.buffer_pos = (self.buffer_pos + chunk_len) % len(self.ring_buffer)
        
        # Extract latest FFT_SIZE samples
        if self.buffer_pos >= FFT_SIZE:
            self.audio_buffer[:] = self.ring_buffer[self.buffer_pos - FFT_SIZE:self.buffer_pos]
        elif self.buffer_pos > 0:
            self.audio_buffer[-self.buffer_pos:] = self.ring_buffer[:self.buffer_pos]
            self.audio_buffer[:-self.buffer_pos] = self.ring_buffer[-(FFT_SIZE - self.buffer_pos):]
        
        # Apply window and compute FFT
        windowed = self.audio_buffer * self.window
        fft_result = np.fft.rfft(windowed)
        magnitude = np.abs(fft_result)
        
        return magnitude
    
    def process_frame(self):
        """Process one frame of audio with voice and beat detection"""
        audio_data = self.capture.get_audio_data()
        if audio_data is None:
            return False
        
        start = time.perf_counter()
        
        # Analyze audio
        fft_data = self.process_audio_chunk(audio_data)
        
        # Map to bands
        band_values = np.zeros(self.bars)
        for i, indices in enumerate(self.band_indices):
            if len(indices) > 0:
                band_values[i] = np.mean(fft_data[indices])
        
        # Normalize
        max_val = np.max(band_values)
        if max_val > 0:
            band_values = band_values / max_val
        
        # Enhanced drum detection
        drum_info = self.drum_detector.process_audio(fft_data, band_values)
        
        # Industry-grade voice detection
        voice_info = self.voice_detector.detect_voice_realtime(audio_data)
        
        # Update flash times for visual feedback
        if drum_info['kick']['kick_detected']:
            self.kick_flash_time = time.time()
        if drum_info['snare']['snare_detected']:
            self.snare_flash_time = time.time()
        if voice_info['has_voice'] and voice_info['voice_confidence'] > 0.7:
            self.voice_flash_time = time.time()
        if voice_info.get('is_singing', False):
            self.singing_flash_time = time.time()
        
        # Store analysis results
        self.drum_info = drum_info
        self.voice_info = voice_info
        
        # Voice-reactive smoothing for spectrum bars
        for i in range(self.bars):
            target = band_values[i]
            current = self.bar_heights[i]
            
            # Voice-reactive smoothing parameters
            freq_range = self.get_frequency_range_for_bar(i)
            
            if freq_range[0] <= 150:  # Bass/kick region
                attack = 0.95 if drum_info['kick']['kick_detected'] else 0.7
                release = 0.08
            elif 150 < freq_range[0] <= 500:  # Low-mid/snare region
                attack = 0.90 if drum_info['snare']['snare_detected'] else 0.75
                release = 0.12
            elif 500 < freq_range[0] <= 2000:  # Voice fundamental region
                voice_boost = 1.2 if voice_info['has_voice'] else 1.0
                attack = 0.85 * voice_boost
                release = 0.15
            elif 2000 < freq_range[0] <= 5000:  # Voice harmonics/formants
                voice_boost = 1.3 if voice_info['has_voice'] else 1.0
                singing_boost = 1.1 if voice_info.get('is_singing', False) else 1.0
                attack = 0.8 * voice_boost * singing_boost
                release = 0.18
            else:  # High frequencies
                attack = 0.75
                release = 0.25
            
            if target > current:
                self.bar_heights[i] = current + (target - current) * attack
            else:
                self.bar_heights[i] = current + (target - current) * release
        
        self.last_process_time = time.perf_counter() - start
        return True
    
    def get_frequency_range_for_bar(self, bar_index: int) -> Tuple[float, float]:
        """Get frequency range for a specific bar"""
        if bar_index < len(self.band_indices):
            indices = self.band_indices[bar_index]
            if len(indices) > 0:
                freq_start = self.freqs[indices[0]]
                freq_end = self.freqs[indices[-1]]
                return (freq_start, freq_end)
        return (0, 0)
    
    def debug_spectrum_snapshot_to_terminal(self):
        """Output ASCII spectrum snapshot to terminal for debugging"""
        if not hasattr(self, 'bar_heights') or not hasattr(self, 'drum_info') or not hasattr(self, 'voice_info'):
            print("📊 No spectrum data available yet")
            return
        
        timestamp = time.strftime("%H:%M:%S")
        
        print("\n" + "="*80)
        print(f"📊 SPECTRUM SNAPSHOT - {timestamp}")
        print("="*80)
        
        # Status summary
        drum_info = self.drum_info
        voice_info = self.voice_info
        
        kick_vel = drum_info['kick'].get('display_velocity', 0)
        snare_vel = drum_info['snare'].get('display_velocity', 0)
        bpm = max(drum_info.get('bpm', 0), drum_info.get('groove', {}).get('stable_bpm', 0))
        
        voice_status = "🎤 VOICE" if voice_info['has_voice'] else "○ Silent"
        if voice_info.get('is_singing', False):
            voice_status = "🎵 SINGING"
        
        # Header with current detections
        print(f"Status: {voice_status} | 🥁 Kick:{kick_vel} | 🔥 Snare:{snare_vel} | BPM:{bpm:.1f}")
        
        if voice_info['has_voice']:
            pitch_info = f"{voice_info['pitch']:.1f}Hz"
            if voice_info.get('pitch_note'):
                pitch_info += f" ({voice_info['pitch_note']})"
            voice_type = voice_info.get('voice_type', 'unknown')
            confidence = voice_info['voice_confidence']
            print(f"Voice: {pitch_info} | Type: {voice_type} | Confidence: {confidence:.1%}")
            
            # Formants
            if voice_info.get('formants'):
                formants = voice_info['formants'][:4]
                formant_str = " | ".join([f"F{i+1}:{f:.0f}Hz" for i, f in enumerate(formants)])
                print(f"Formants: {formant_str}")
            
            # Vibrato
            vibrato = voice_info.get('vibrato')
            if vibrato and vibrato.get('detected'):
                print(f"Vibrato: {vibrato['rate']:.1f}Hz (strength: {vibrato['strength']:.3f})")
        
        print("\n📊 FREQUENCY SPECTRUM (ASCII):")
        print("Freq (Hz)    Level    Bar")
        
        # Create ASCII spectrum with frequency annotations
        spectrum_height = 20  # ASCII bar height
        num_bars_to_show = min(64, self.bars)  # Show up to 64 bars to fit in terminal
        
        # Sample bars across the full range
        step = max(1, self.bars // num_bars_to_show)
        
        for i in range(0, self.bars, step):
            if i >= len(self.bar_heights):
                break
                
            level = self.bar_heights[i]
            freq_range = self.get_frequency_range_for_bar(i)
            freq_center = (freq_range[0] + freq_range[1]) / 2
            
            # Create ASCII bar
            bar_length = int(level * spectrum_height)
            bar_visual = "█" * bar_length + "░" * (spectrum_height - bar_length)
            
            # Level as percentage
            level_pct = int(level * 100)
            
            # Frequency range annotation
            freq_str = f"{freq_center:7.0f}"
            
            # Special annotations for important frequencies
            annotations = []
            
            # Beat detection ranges
            if freq_center <= 120:
                if kick_vel > 0:
                    annotations.append("🥁KICK")
                else:
                    annotations.append("kick")
            elif 150 <= freq_center <= 800:
                if snare_vel > 0:
                    annotations.append("🔥SNARE")
                else:
                    annotations.append("snare")
            
            # Voice detection ranges  
            if voice_info['has_voice']:
                voice_pitch = voice_info.get('pitch', 0)
                formants = voice_info.get('formants', [])
                
                # Pitch highlighting
                if voice_pitch > 0 and abs(freq_center - voice_pitch) < 50:
                    annotations.append("🎤PITCH")
                
                # Formant highlighting
                for j, formant in enumerate(formants[:3]):
                    if abs(freq_center - formant) < 100:
                        annotations.append(f"F{j+1}")
            
            annotation_str = " ".join(annotations) if annotations else ""
            
            print(f"{freq_str}    {level_pct:3d}%    {bar_visual} {annotation_str}")
        
        # Multi-band analysis summary
        if drum_info['kick'].get('sub_flux', 0) > 0 or drum_info['snare'].get('fundamental_flux', 0) > 0:
            print("\n🔍 MULTI-BAND ANALYSIS:")
            
            kick = drum_info['kick']
            if any(kick.get(k, 0) > 0 for k in ['sub_flux', 'body_flux', 'click_flux']):
                print(f"Kick Bands: Sub:{kick.get('sub_flux', 0):.2f} | Body:{kick.get('body_flux', 0):.2f} | Click:{kick.get('click_flux', 0):.2f}")
            
            snare = drum_info['snare']
            if any(snare.get(k, 0) > 0 for k in ['fundamental_flux', 'body_flux', 'snap_flux']):
                print(f"Snare Bands: Fund:{snare.get('fundamental_flux', 0):.2f} | Body:{snare.get('body_flux', 0):.2f} | Snap:{snare.get('snap_flux', 0):.2f}")
        
        # Performance info
        latency_ms = self.last_process_time * 1000
        print(f"\n⚡ Performance: {latency_ms:.1f}ms latency")
        
        print("="*80)
    
    def auto_debug_voice_status(self):
        """Automatic voice detection status output every 3 seconds"""
        if not hasattr(self, 'voice_info') or not hasattr(self, 'drum_info'):
            return
        
        voice_info = self.voice_info
        drum_info = self.drum_info
        timestamp = time.strftime("%H:%M:%S")
        
        # Voice status
        voice_status = "🎤 VOICE" if voice_info['has_voice'] else "○ Silent"
        if voice_info.get('is_singing', False):
            voice_status = "🎵 SINGING"
        
        # Beat status
        kick_vel = drum_info['kick'].get('display_velocity', 0)
        snare_vel = drum_info['snare'].get('display_velocity', 0)
        bpm = max(drum_info.get('bpm', 0), drum_info.get('groove', {}).get('stable_bpm', 0))
        
        print(f"\n[{timestamp}] AUTO-DEBUG: {voice_status} | Kick:{kick_vel} | Snare:{snare_vel} | BPM:{bpm:.1f}")
        
        # Voice details if detected
        if voice_info['has_voice']:
            pitch = voice_info.get('pitch', 0)
            confidence = voice_info.get('voice_confidence', 0)
            voice_type = voice_info.get('voice_type', 'unknown')
            
            pitch_str = f"{pitch:.1f}Hz"
            if voice_info.get('pitch_note'):
                pitch_str += f" ({voice_info['pitch_note']})"
            
            print(f"          Voice: {pitch_str} | Type: {voice_type} | Conf: {confidence:.1%}")
            
            # Formants if available
            if voice_info.get('formants'):
                formants = voice_info['formants'][:3]
                formant_str = " | ".join([f"F{i+1}:{f:.0f}" for i, f in enumerate(formants)])
                print(f"          Formants: {formant_str}")
        else:
            # Debug why voice not detected
            vocal_energy = voice_info.get('vocal_energy', 0)
            print(f"          No Voice Detected - Energy: {vocal_energy:.4f}")
    
    def draw_frame(self):
        """Draw the enhanced voice and beat visualization"""
        # Clear screen
        self.screen.fill((5, 8, 15))
        
        # Enhanced header with voice info
        header_height = 180 if self.show_voice_info else 140
        
        # Background for header
        header_rect = pygame.Rect(0, 0, self.width, header_height)
        pygame.draw.rect(self.screen, (15, 20, 30), header_rect)
        
        # Title with version info
        title = self.font_large.render("Voice & Beat Analyzer v4", True, (255, 255, 255))
        self.screen.blit(title, (10, 10))
        
        # Industry features subtitle
        subtitle = self.font_small.render("Voice Detection • Singing Analysis • Multi-Band Beats • Groove Recognition", True, (180, 200, 220))
        self.screen.blit(subtitle, (10, 45))
        
        # Enhanced status displays
        current_time = time.time()
        kick_flash_active = (current_time - self.kick_flash_time) < self.kick_flash_duration
        snare_flash_active = (current_time - self.snare_flash_time) < self.snare_flash_duration
        voice_flash_active = (current_time - self.voice_flash_time) < self.voice_flash_duration
        singing_flash_active = (current_time - self.singing_flash_time) < self.singing_flash_duration
        
        # Right column - Drum info
        if hasattr(self, 'drum_info'):
            groove_info = self.drum_info.get('groove', {})
            stable_bpm = groove_info.get('stable_bpm', 0)
            bpm_to_show = max(self.drum_info['bpm'], stable_bpm)
            
            bpm_color = (120, 255, 120) if kick_flash_active else (200, 200, 210)
            bpm_text = self.font_medium.render(f"BPM: {bpm_to_show:.1f}", True, bpm_color)
            self.screen.blit(bpm_text, (self.width - 300, 10))
            
            # Groove pattern info
            if self.show_groove_info and groove_info:
                pattern = groove_info.get('groove_pattern', 'unknown')
                pattern_confidence = groove_info.get('pattern_confidence', 0)
                pattern_color = (100, 255, 150) if pattern != 'unknown' else (150, 150, 150)
                pattern_text = self.font_small.render(f"Pattern: {pattern} ({pattern_confidence:.1%})", True, pattern_color)
                self.screen.blit(pattern_text, (self.width - 300, 35))
            
            # Enhanced kick indicator with display persistence
            kick_display_vel = self.drum_info['kick'].get('display_velocity', 0)
            
            if kick_display_vel > 0 or kick_flash_active:
                kick_indicator = f"🥁 KICK! {kick_display_vel}"
                kick_color = (255, 120, 120) if kick_flash_active else (200, 100, 100)
            else:
                kick_indicator = "○ ----"
                kick_color = (80, 80, 80)
            
            kick_text = self.font_medium.render(kick_indicator, True, kick_color)
            self.screen.blit(kick_text, (self.width - 300, 60))
            
            # Enhanced snare indicator with display persistence
            snare_display_vel = self.drum_info['snare'].get('display_velocity', 0)
            
            if snare_display_vel > 0 or snare_flash_active:
                snare_indicator = f"🔥 SNARE! {snare_display_vel}"
                snare_color = (120, 255, 255) if snare_flash_active else (100, 200, 200)
            else:
                snare_indicator = "◇ ----"
                snare_color = (80, 80, 80)
            
            snare_text = self.font_medium.render(snare_indicator, True, snare_color)
            self.screen.blit(snare_text, (self.width - 300, 85))
        
        # Left column - Voice info
        if self.show_voice_info and hasattr(self, 'voice_info'):
            voice_info = self.voice_info
            
            # Voice detection status
            if voice_info['has_voice']:
                voice_indicator = f"🎤 VOICE! {voice_info['voice_confidence']:.1%}"
                voice_color = (255, 220, 120) if voice_flash_active else (200, 180, 100)
            else:
                voice_indicator = "○ ----"
                voice_color = (80, 80, 80)
            
            voice_text = self.font_medium.render(voice_indicator, True, voice_color)
            self.screen.blit(voice_text, (10, 70))
            
            # Pitch information
            if voice_info['pitch'] > 0:
                pitch_text = f"Pitch: {voice_info['pitch']:.1f}Hz"
                if voice_info['pitch_note']:
                    pitch_text += f" ({voice_info['pitch_note']})"
                pitch_color = (150, 255, 150) if voice_flash_active else (120, 200, 120)
                pitch_display = self.font_small.render(pitch_text, True, pitch_color)
                self.screen.blit(pitch_display, (10, 95))
            
            # Voice type and singing
            voice_type = voice_info.get('voice_type', 'unknown')
            is_singing = voice_info.get('is_singing', False)
            
            if is_singing:
                type_text = f"🎵 SINGING ({voice_type})"
                type_color = (255, 150, 255) if singing_flash_active else (200, 120, 200)
            elif voice_type != 'unknown':
                type_text = f"Voice: {voice_type}"
                type_color = (180, 180, 255) if voice_flash_active else (140, 140, 200)
            else:
                type_text = "Voice: detecting..."
                type_color = (120, 120, 120)
            
            type_display = self.font_small.render(type_text, True, type_color)
            self.screen.blit(type_display, (10, 120))
            
            # Vibrato detection
            vibrato_info = voice_info.get('vibrato')
            if vibrato_info and vibrato_info.get('detected', False):
                vibrato_text = f"Vibrato: {vibrato_info['rate']:.1f}Hz"
                vibrato_color = (255, 200, 150)
                vibrato_display = self.font_tiny.render(vibrato_text, True, vibrato_color)
                self.screen.blit(vibrato_display, (10, 145))
            
            # Formant display
            if self.show_formants and voice_info.get('formants'):
                formants = voice_info['formants'][:3]  # Show first 3 formants
                formant_text = f"F1:{formants[0]:.0f}" if len(formants) > 0 else ""
                if len(formants) > 1:
                    formant_text += f" F2:{formants[1]:.0f}"
                if len(formants) > 2:
                    formant_text += f" F3:{formants[2]:.0f}"
                
                if formant_text:
                    formant_color = (200, 255, 200)
                    formant_display = self.font_tiny.render(f"Formants: {formant_text}", True, formant_color)
                    self.screen.blit(formant_display, (200, 145))
        
        # Performance and source info
        latency_ms = self.last_process_time * 1000
        stats_text = self.font_small.render(
            f"Latency: {latency_ms:.1f}ms | Source: {os.path.basename(self.capture.source_name or 'None')}", 
            True, (160, 160, 170))
        self.screen.blit(stats_text, (200, 120))
        
        # Separator line
        pygame.draw.line(self.screen, (60, 70, 90), (0, header_height), (self.width, header_height), 2)
        
        # Enhanced spectrum visualization with voice reactivity and scales
        # Reserve space for scales
        scale_margin_left = 60   # Space for amplitude scale
        scale_margin_bottom = 40 # Space for frequency scale
        
        vis_height = self.height - header_height - scale_margin_bottom
        vis_width = self.width - scale_margin_left
        center_y = header_height + vis_height // 2
        max_bar_height = (vis_height // 2) - 20
        
        bar_width = vis_width / self.bars
        spectrum_start_x = scale_margin_left
        
        # Enhanced flash effects for voice and beats (adjusted for scales)
        flash_width = vis_width // 3
        if kick_flash_active or (hasattr(self, 'drum_info') and self.drum_info['kick'].get('display_strength', 0) > 0.1):
            kick_alpha = 60 if kick_flash_active else int(30 * self.drum_info['kick'].get('display_strength', 0))
            kick_flash_surface = pygame.Surface((flash_width, vis_height))
            kick_flash_surface.set_alpha(kick_alpha)
            kick_flash_surface.fill((kick_alpha, kick_alpha // 4, 0))
            self.screen.blit(kick_flash_surface, (spectrum_start_x, header_height))
        
        if snare_flash_active or (hasattr(self, 'drum_info') and self.drum_info['snare'].get('display_strength', 0) > 0.1):
            snare_alpha = 60 if snare_flash_active else int(30 * self.drum_info['snare'].get('display_strength', 0))
            snare_flash_surface = pygame.Surface((flash_width, vis_height))
            snare_flash_surface.set_alpha(snare_alpha)
            snare_flash_surface.fill((0, snare_alpha // 3, snare_alpha // 2))
            self.screen.blit(snare_flash_surface, (spectrum_start_x + flash_width, header_height))
        
        if voice_flash_active or singing_flash_active:
            voice_alpha = 40 if voice_flash_active else 0
            singing_alpha = 50 if singing_flash_active else 0
            total_alpha = voice_alpha + singing_alpha
            
            if total_alpha > 0:
                voice_flash_surface = pygame.Surface((flash_width, vis_height))
                voice_flash_surface.set_alpha(total_alpha)
                voice_flash_surface.fill((total_alpha // 2, total_alpha, total_alpha // 3))
                self.screen.blit(voice_flash_surface, (spectrum_start_x + 2 * flash_width, header_height))
        
        # Draw amplitude scale (left side)
        self.draw_amplitude_scale(scale_margin_left, header_height, vis_height, max_bar_height)
        
        # Draw frequency scale (bottom)
        self.draw_frequency_scale(spectrum_start_x, header_height + vis_height, vis_width)
        
        # Draw center line
        pygame.draw.line(self.screen, (40, 50, 60), (spectrum_start_x, center_y), (self.width, center_y), 1)
        
        # Draw enhanced spectrum bars with voice and beat reactivity
        for i in range(self.bars):
            if self.bar_heights[i] > 0.01:
                height = int(self.bar_heights[i] * max_bar_height)
                x = spectrum_start_x + int(i * bar_width)
                width = int(bar_width) + 1
                
                color = self.colors[i]
                
                # Enhanced reactivity based on voice and beat detection
                freq_range = self.get_frequency_range_for_bar(i)
                
                # Voice reactivity
                if hasattr(self, 'voice_info') and self.voice_info['has_voice']:
                    pitch = self.voice_info.get('pitch', 0)
                    formants = self.voice_info.get('formants', [])
                    
                    # Highlight pitch frequency
                    if pitch > 0 and freq_range[0] <= pitch <= freq_range[1]:
                        boost = 1.4 if singing_flash_active else 1.2
                        color = tuple(min(255, int(c * boost)) for c in color)
                    
                    # Highlight formant frequencies
                    for formant in formants[:3]:  # First 3 formants
                        if freq_range[0] <= formant <= freq_range[1]:
                            boost = 1.3
                            color = tuple(min(255, int(c * boost)) for c in color)
                
                # Beat reactivity
                if hasattr(self, 'drum_info'):
                    kick_strength = self.drum_info['kick'].get('display_strength', 0)
                    snare_strength = self.drum_info['snare'].get('display_strength', 0)
                    
                    if freq_range[0] <= 120 and kick_strength > 0.1:  # Kick range
                        boost = 1.0 + kick_strength * 0.8
                        color = tuple(min(255, int(c * boost)) for c in color)
                    elif 150 <= freq_range[0] <= 800 and snare_strength > 0.1:  # Snare range
                        boost = 1.0 + snare_strength * 0.6
                        color = tuple(min(255, int(c * boost)) for c in color)
                
                # Upper bar
                pygame.draw.rect(self.screen, color, (x, center_y - height, width, height))
                
                # Lower bar (mirrored, slightly darker)
                lower_color = tuple(int(c * 0.75) for c in color)
                pygame.draw.rect(self.screen, lower_color, (x, center_y, width, height))
    
    def draw_amplitude_scale(self, margin_left: int, vis_start_y: int, vis_height: int, max_bar_height: int):
        """Draw amplitude scale on the left side for mirrored display"""
        scale_color = (120, 130, 140)
        center_y = vis_start_y + vis_height // 2
        
        # Draw scale line
        pygame.draw.line(self.screen, scale_color, (margin_left - 5, vis_start_y), (margin_left - 5, vis_start_y + vis_height), 1)
        
        # Draw amplitude markers for mirrored display (25%, 50%, 75%, 100% from center)
        amplitude_levels = [0.25, 0.5, 0.75, 1.0]
        for level in amplitude_levels:
            # Upper scale (from center upward)
            y_pos = center_y - int(level * max_bar_height)
            pygame.draw.line(self.screen, scale_color, (margin_left - 8, y_pos), (margin_left - 2, y_pos), 1)
            
            # Lower scale (from center downward)
            y_pos_lower = center_y + int(level * max_bar_height)
            pygame.draw.line(self.screen, scale_color, (margin_left - 8, y_pos_lower), (margin_left - 2, y_pos_lower), 1)
            
            # Labels for upper scale
            label = self.font_tiny.render(f"{int(level * 100)}%", True, scale_color)
            label_rect = label.get_rect()
            self.screen.blit(label, (margin_left - 35, y_pos - label_rect.height // 2))
            
            # Labels for lower scale (mirrored)
            label_lower = self.font_tiny.render(f"{int(level * 100)}%", True, scale_color)
            label_rect_lower = label_lower.get_rect()
            self.screen.blit(label_lower, (margin_left - 35, y_pos_lower - label_rect_lower.height // 2))
        
        # Center line label (0% at center)
        center_label = self.font_tiny.render("0%", True, scale_color)
        center_rect = center_label.get_rect()
        self.screen.blit(center_label, (margin_left - 30, center_y - center_rect.height // 2))
    
    def draw_frequency_scale(self, spectrum_start_x: int, scale_y: int, vis_width: int):
        """Draw frequency scale showing frequency ranges for each section"""
        scale_color = (120, 130, 140)
        
        # Draw scale line
        pygame.draw.line(self.screen, scale_color, (spectrum_start_x, scale_y), (spectrum_start_x + vis_width, scale_y), 1)
        
        # Draw frequency markers at regular intervals showing the actual frequency ranges
        num_markers = 8  # Number of frequency markers to show
        
        for i in range(num_markers + 1):
            # Calculate position across the spectrum
            bar_position = int((i / num_markers) * self.bars)
            if bar_position >= self.bars:
                bar_position = self.bars - 1
                
            x_pos = spectrum_start_x + int((i / num_markers) * vis_width)
            
            # Get the actual frequency range for this bar position
            freq_range = self.get_frequency_range_for_bar(bar_position)
            if freq_range[0] > 0 and freq_range[1] > 0:
                # Use the center frequency of the range
                center_freq = (freq_range[0] + freq_range[1]) / 2
                
                # Draw tick mark
                pygame.draw.line(self.screen, scale_color, (x_pos, scale_y), (x_pos, scale_y + 8), 1)
                
                # Format frequency label based on value
                if center_freq >= 1000:
                    if center_freq >= 10000:
                        freq_text = f"{center_freq / 1000:.0f}k"
                    else:
                        freq_text = f"{center_freq / 1000:.1f}k"
                else:
                    if center_freq >= 100:
                        freq_text = f"{center_freq:.0f}"
                    else:
                        freq_text = f"{center_freq:.1f}"
                
                label = self.font_tiny.render(freq_text, True, scale_color)
                label_rect = label.get_rect()
                # Center the label under the tick mark
                self.screen.blit(label, (x_pos - label_rect.width // 2, scale_y + 10))
    
    def find_bar_position_for_frequency(self, target_freq: float) -> Optional[int]:
        """Find which bar corresponds to a given frequency"""
        for i, indices in enumerate(self.band_indices):
            if len(indices) > 0:
                freq_range = self.get_frequency_range_for_bar(i)
                if freq_range[0] <= target_freq <= freq_range[1]:
                    return i
                # If frequency is close to the range, return this bar
                center_freq = (freq_range[0] + freq_range[1]) / 2
                if abs(target_freq - center_freq) < (freq_range[1] - freq_range[0]) / 2:
                    return i
        return None
    
    def resize_window(self, new_width: int, new_height: int):
        """Resize the window to a new size"""
        self.width = new_width
        self.height = new_height
        self.screen = pygame.display.set_mode((new_width, new_height), pygame.RESIZABLE)
        print(f"📏 Window resized to {new_width}x{new_height}")
    
    def run(self):
        """Main enhanced analysis loop"""
        # Start audio capture
        if not self.capture.start_capture():
            print("❌ Failed to start audio capture. Exiting.")
            return
        
        running = True
        frame_count = 0
        fps_timer = time.time()
        
        try:
            while running:
                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        elif event.key == pygame.K_s:
                            timestamp = time.strftime("%Y%m%d_%H%M%S")
                            filename = f"voice_beat_analysis_v4_{timestamp}.png"
                            pygame.image.save(self.screen, filename)
                            print(f"📸 Screenshot saved: {filename}")
                        elif event.key == pygame.K_b:
                            # Toggle drum sensitivity
                            new_sensitivity = 2.0 if self.drum_detector.sensitivity == 1.0 else 1.0
                            self.drum_detector.set_sensitivity(new_sensitivity)
                            print(f"🎛️ Drum sensitivity: {new_sensitivity}")
                        elif event.key == pygame.K_v:
                            # Toggle voice info
                            self.show_voice_info = not self.show_voice_info
                            print(f"🎤 Voice info: {'ON' if self.show_voice_info else 'OFF'}")
                        elif event.key == pygame.K_f:
                            # Toggle formant display
                            self.show_formants = not self.show_formants
                            print(f"🔊 Formant display: {'ON' if self.show_formants else 'OFF'}")
                        elif event.key == pygame.K_a:
                            # Toggle advanced info
                            self.show_advanced_info = not self.show_advanced_info
                            print(f"🔍 Advanced info: {'ON' if self.show_advanced_info else 'OFF'}")
                        elif event.key == pygame.K_g:
                            # Toggle groove info
                            self.show_groove_info = not self.show_groove_info
                            print(f"🎼 Groove info: {'ON' if self.show_groove_info else 'OFF'}")
                        elif event.key == pygame.K_1:
                            # Window size 1: Compact (1200x720)
                            self.resize_window(1200, 720)
                        elif event.key == pygame.K_2:
                            # Window size 2: Standard (1600x900)
                            self.resize_window(1600, 900)
                        elif event.key == pygame.K_3:
                            # Window size 3: Wide (2000x900) - Default
                            self.resize_window(2000, 900)
                        elif event.key == pygame.K_4:
                            # Window size 4: Extra Wide (2400x1080)
                            self.resize_window(2400, 1080)
                        elif event.key == pygame.K_5:
                            # Window size 5: Ultra Wide (2880x1200)
                            self.resize_window(2880, 1200)
                        elif event.key == pygame.K_6:
                            # Window size 6: Full Professional (3200x1440)
                            self.resize_window(3200, 1440)
                        elif event.key == pygame.K_d:
                            # Debug spectrum snapshot to terminal
                            self.debug_spectrum_snapshot_to_terminal()
                        elif event.key == pygame.K_t:
                            # Toggle auto debug output
                            self.auto_debug_enabled = not self.auto_debug_enabled
                            status = "ON" if self.auto_debug_enabled else "OFF"
                            print(f"🔍 Auto debug output: {status}")
                        elif event.key == pygame.K_k:
                            # Enhanced kick detection info
                            if hasattr(self, 'drum_info'):
                                kick = self.drum_info['kick']
                                print(f"🥁 ENHANCED KICK INFO:")
                                print(f"   Sub-bass: {kick.get('sub_flux', 0):.3f}")
                                print(f"   Body: {kick.get('body_flux', 0):.3f}")
                                print(f"   Click: {kick.get('click_flux', 0):.3f}")
                                print(f"   Display: Strength:{kick.get('display_strength', 0):.3f} Velocity:{kick.get('display_velocity', 0)}")
                        elif event.key == pygame.K_n:
                            # Enhanced snare detection info
                            if hasattr(self, 'drum_info'):
                                snare = self.drum_info['snare']
                                print(f"🔥 ENHANCED SNARE INFO:")
                                print(f"   Fundamental: {snare.get('fundamental_flux', 0):.3f}")
                                print(f"   Body: {snare.get('body_flux', 0):.3f}")
                                print(f"   Snap: {snare.get('snap_flux', 0):.3f}")
                                print(f"   Display: Strength:{snare.get('display_strength', 0):.3f} Velocity:{snare.get('display_velocity', 0)}")
                        elif event.key == pygame.K_p:
                            # Enhanced voice/pitch detection info
                            if hasattr(self, 'voice_info'):
                                voice = self.voice_info
                                print(f"🎤 ENHANCED VOICE DETECTION INFO:")
                                print(f"   Has Voice: {voice['has_voice']}")
                                print(f"   Pitch: {voice['pitch']:.1f}Hz ({voice.get('pitch_note', 'N/A')})")
                                print(f"   Pitch Confidence: {voice.get('pitch_confidence', 0):.1%}")
                                print(f"   Overall Confidence: {voice['voice_confidence']:.1%}")
                                print(f"   Voice Type: {voice.get('voice_type', 'unknown')}")
                                print(f"   Is Singing: {voice.get('is_singing', False)}")
                                print(f"   HNR: {voice.get('hnr', 0):.1f}dB")
                                print(f"   Vocal Energy: {voice.get('vocal_energy', 0):.3f}")
                                print(f"   Fundamental Clarity: {voice.get('fundamental_clarity', 0):.3f}")
                                
                                if voice.get('formants'):
                                    formants_str = ", ".join([f"{f:.0f}Hz" for f in voice['formants'][:4]])
                                    print(f"   Formants: {formants_str}")
                                
                                # Enhanced spectral features
                                spectral = voice.get('spectral_features', {})
                                if spectral:
                                    print(f"   Spectral Centroid: {spectral.get('centroid', 0):.0f}Hz")
                                    print(f"   Spectral Rolloff: {spectral.get('rolloff', 0):.0f}Hz")
                                    print(f"   Spectral Flatness: {spectral.get('flatness', 0):.3f}")
                                    print(f"   Zero Crossing Rate: {spectral.get('zcr', 0):.3f}")
                                
                                vibrato = voice.get('vibrato')
                                if vibrato and vibrato.get('detected'):
                                    print(f"   Vibrato: {vibrato['rate']:.1f}Hz (strength: {vibrato['strength']:.3f})")
                                
                                # Debug info for mixed music analysis
                                print(f"   DEBUG - Voice detection optimized for Taylor Swift-style vocals")
                                print(f"   DEBUG - Female vocal range (150-500Hz) prioritized")
                                print(f"   DEBUG - Drum suppression and vocal isolation active")
                
                # Process audio and draw
                self.process_frame()
                
                # Auto debug output every 3 seconds
                current_time = time.time()
                if (self.auto_debug_enabled and 
                    current_time - self.last_auto_debug_time >= self.auto_debug_interval):
                    self.auto_debug_voice_status()
                    self.last_auto_debug_time = current_time
                
                self.draw_frame()
                
                # Update display
                pygame.display.flip()
                self.clock.tick(60)
                
                # FPS tracking
                frame_count += 1
                if time.time() - fps_timer >= 1.0:
                    self.fps_counter.append(frame_count)
                    frame_count = 0
                    fps_timer = time.time()
                
        finally:
            self.capture.stop_capture()
            pygame.quit()

def main():
    parser = argparse.ArgumentParser(description='Voice & Beat Analyzer v4 - Industry-Grade Detection')
    parser.add_argument('--width', type=int, default=2000, help='Window width')
    parser.add_argument('--height', type=int, default=900, help='Window height')
    parser.add_argument('--bars', type=int, default=512, help='Number of spectrum bars')
    parser.add_argument('--source', type=str, default=None, help='PipeWire source name')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("VOICE & BEAT ANALYZER V4")
    print("="*80)
    print("Industry-Grade Features:")
    print("  🎤 Voice Detection: WebRTC VAD, YIN pitch, LPC formants")
    print("  🎵 Singing Analysis: Vibrato detection, voice classification")
    print("  🥁 Enhanced Beat Detection: Multi-band onset with value persistence")
    print("  🎼 Groove Recognition: 8 musical patterns with tempo tracking")
    print("  📊 Voice-Reactive Visualization: Pitch/formant highlighting")
    print("  🔊 Optimized for Focusrite Scarlett 2i2")
    print("  ⚡ Real-time performance with 60+ FPS")
    print("="*80)
    
    analyzer = VoiceReactiveLiveAudioAnalyzer(
        width=args.width,
        height=args.height,
        bars=args.bars,
        source_name=args.source
    )
    
    analyzer.run()

if __name__ == "__main__":
    main()