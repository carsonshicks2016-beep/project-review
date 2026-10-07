// Cross-browser SpeechRecognition wrapper with watchdog and auto-flush

interface IWindow extends Window {
  SpeechRecognition?: any;
  webkitSpeechRecognition?: any;
}

export type SpeechStatus = 'inactive' | 'listening' | 'paused' | 'error';

export interface SpeechCallbacks {
  onInterimResult: (text: string) => void;
  onFinalResult: (text: string) => void;
  onError: (error: string) => void;
  onStatusChange: (status: SpeechStatus) => void;
}

export class SpeechService {
  private recognition: any = null;
  private isExplicitlyStopped = true;
  private status: SpeechStatus = 'inactive';
  private callbacks: SpeechCallbacks;
  private lang = 'en-US';
  private watchdogTimer: number | null = null;
  private lastInterimText = '';
  private interimFlushTimer: number | null = null;
  private lastSpeechTimestamp = Date.now();
  /**
   * Text already committed by the idle auto-flush. The recognizer usually goes on
   * to emit its own final result for the same utterance a moment later; without
   * this guard those words would be counted twice, inflating every frequency.
   */
  private flushedText = '';

  constructor(callbacks: SpeechCallbacks, lang = 'en-US') {
    this.callbacks = callbacks;
    this.lang = lang;
    this.initRecognition();
    this.startWatchdog();
  }

  /** Loose comparison key: case and punctuation vary between interim and final results. */
  private static speechKey(text: string): string {
    return text.toLowerCase().replace(/[^a-z0-9\s]/g, '').replace(/\s+/g, ' ').trim();
  }

  /**
   * Returns the part of `full` that follows an already-emitted `flushed` prefix,
   * or null when `full` is not a continuation of it.
   *
   * Words are consumed one at a time and compared on the normalized key rather
   * than sliced by word count, because the two strings tokenize differently: a
   * stray "—" is a whitespace-delimited token but contributes nothing to the key.
   */
  private static remainderAfter(full: string, flushed: string): string | null {
    const flushedKey = SpeechService.speechKey(flushed);
    if (!flushedKey) return null;

    const words = full.split(/\s+/);
    let consumed = '';
    for (let i = 0; i < words.length; i++) {
      const candidate = consumed ? `${consumed} ${words[i]}` : words[i];
      const candidateKey = SpeechService.speechKey(candidate);
      if (candidateKey === flushedKey) return words.slice(i + 1).join(' ');
      if (!flushedKey.startsWith(candidateKey)) return null;
      consumed = candidate;
    }
    return null;
  }

  /**
   * Commits recognized speech, dropping any prefix the auto-flush already emitted.
   */
  private commitFinal(text: string) {
    const trimmed = text.trim();
    if (!trimmed) return;

    if (this.flushedText) {
      const remainder = SpeechService.remainderAfter(trimmed, this.flushedText);
      this.flushedText = '';
      if (remainder !== null) {
        if (remainder.trim()) this.callbacks.onFinalResult(remainder.trim());
        return;
      }
    }

    this.callbacks.onFinalResult(trimmed);
  }

  private initRecognition() {
    const win = typeof window !== 'undefined' ? (window as unknown as IWindow) : null;
    const SpeechRecognitionClass = win?.SpeechRecognition || win?.webkitSpeechRecognition;

    if (!SpeechRecognitionClass) {
      this.status = 'error';
      this.callbacks.onError('Web Speech API is not supported in this browser. Please use Google Chrome, Edge, or Safari.');
      this.callbacks.onStatusChange(this.status);
      return;
    }

    try {
      this.recognition = new SpeechRecognitionClass();
      this.recognition.continuous = true;
      this.recognition.interimResults = true;
      this.recognition.lang = this.lang;
      this.recognition.maxAlternatives = 1;

      this.recognition.onresult = (event: any) => {
        this.lastSpeechTimestamp = Date.now();
        let interimTranscript = '';
        let finalTranscript = '';

        for (let i = event.resultIndex; i < event.results.length; ++i) {
          const transcriptPiece = event.results[i][0].transcript;
          if (event.results[i].isFinal) {
            finalTranscript += transcriptPiece;
          } else {
            interimTranscript += transcriptPiece;
          }
        }

        if (finalTranscript.trim()) {
          this.commitFinal(finalTranscript);
          this.lastInterimText = '';
          if (this.interimFlushTimer) clearTimeout(this.interimFlushTimer);
        }

        if (interimTranscript.trim()) {
          this.lastInterimText = interimTranscript.trim();
          this.callbacks.onInterimResult(interimTranscript);

          // Auto-flush interim text if the browser hesitates/pauses for 1.8s
          if (this.interimFlushTimer) clearTimeout(this.interimFlushTimer);
          this.interimFlushTimer = window.setTimeout(() => {
            if (this.lastInterimText && !this.isExplicitlyStopped) {
              this.flushedText = this.lastInterimText;
              this.callbacks.onFinalResult(this.lastInterimText);
              this.lastInterimText = '';
              this.callbacks.onInterimResult('');
            }
          }, 1800);
        } else {
          this.callbacks.onInterimResult('');
        }
      };

      this.recognition.onerror = (event: any) => {
        if (event.error === 'no-speech') {
          // Normal silence, ignore
          return;
        }
        if (event.error === 'not-allowed' || event.error === 'service-not-allowed') {
          this.callbacks.onError('Microphone access denied. Allow the microphone in browser settings.');
          this.status = 'error';
          this.callbacks.onStatusChange(this.status);
          return;
        }
        if (event.error === 'audio-capture') {
          this.callbacks.onError('No microphone was found. Check that an input device is connected.');
          this.status = 'error';
          this.callbacks.onStatusChange(this.status);
          return;
        }
        if (event.error === 'network') {
          // Recognition is a cloud service in Chrome; it recovers when the network does.
          this.callbacks.onError('Speech recognition lost its network connection. Retrying...');
          return;
        }
        console.warn('Speech recognition notice:', event.error);
      };

      this.recognition.onend = () => {
        // Flush any hanging interim text so nothing is lost
        if (this.lastInterimText) {
          this.commitFinal(this.lastInterimText);
          this.lastInterimText = '';
          this.callbacks.onInterimResult('');
        }

        // Auto-restart immediately if user did not stop it
        if (!this.isExplicitlyStopped && this.status === 'listening') {
          setTimeout(() => {
            if (!this.isExplicitlyStopped && this.status === 'listening') {
              try {
                this.recognition.start();
              } catch {
                // Ignore start collision
              }
            }
          }, 30);
        } else if (this.isExplicitlyStopped) {
          this.status = 'inactive';
          this.callbacks.onStatusChange(this.status);
        }
      };
    } catch (err) {
      console.error('Failed to initialize speech recognition:', err);
      this.status = 'error';
      this.callbacks.onError('Failed to initialize speech recognition.');
      this.callbacks.onStatusChange(this.status);
    }
  }

  private startWatchdog() {
    // Watchdog every 2 seconds ensures the engine doesn't go dormant on long silence
    this.watchdogTimer = window.setInterval(() => {
      if (!this.isExplicitlyStopped && this.status === 'listening') {
        const timeSinceSpeech = Date.now() - this.lastSpeechTimestamp;
        if (timeSinceSpeech > 12000) {
          // Restart recognition to keep audio buffer fresh
          try {
            this.recognition?.stop();
          } catch {}
          this.lastSpeechTimestamp = Date.now();
        }
      }
    }, 2000);
  }

  public isSupported(): boolean {
    if (typeof window === 'undefined') return false;
    const win = window as unknown as IWindow;
    return Boolean(win.SpeechRecognition || win.webkitSpeechRecognition);
  }

  public start() {
    if (!this.recognition) {
      this.initRecognition();
      if (!this.recognition) return;
    }

    this.isExplicitlyStopped = false;
    this.status = 'listening';
    this.flushedText = '';
    this.lastSpeechTimestamp = Date.now();
    this.callbacks.onStatusChange(this.status);

    try {
      this.recognition.start();
    } catch (e: any) {
      if (e.name !== 'InvalidStateError') {
        console.error('Error starting recognition:', e);
      }
    }
  }

  public pause() {
    this.isExplicitlyStopped = true;
    this.status = 'paused';
    this.callbacks.onStatusChange(this.status);
    if (this.lastInterimText) {
      this.commitFinal(this.lastInterimText);
      this.lastInterimText = '';
      this.callbacks.onInterimResult('');
    }
    try {
      this.recognition?.stop();
    } catch (e) {
      console.warn('Error pausing recognition:', e);
    }
  }

  public stop() {
    this.isExplicitlyStopped = true;
    this.status = 'inactive';
    this.callbacks.onStatusChange(this.status);
    if (this.lastInterimText) {
      this.commitFinal(this.lastInterimText);
      this.lastInterimText = '';
      this.callbacks.onInterimResult('');
    }
    try {
      this.recognition?.stop();
    } catch (e) {
      console.warn('Error stopping recognition:', e);
    }
  }

  public cleanup() {
    if (this.watchdogTimer) clearInterval(this.watchdogTimer);
    if (this.interimFlushTimer) clearTimeout(this.interimFlushTimer);
    this.stop();
  }

  public setLanguage(lang: string) {
    this.lang = lang;
    if (this.recognition) {
      this.recognition.lang = lang;
    }
  }
}
