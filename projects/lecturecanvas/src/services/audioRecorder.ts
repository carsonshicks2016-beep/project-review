export class AudioRecorderService {
  private mediaRecorder: MediaRecorder | null = null;
  private audioChunks: Blob[] = [];
  private audioStream: MediaStream | null = null;
  private audioContext: AudioContext | null = null;
  private analyser: AnalyserNode | null = null;
  private gainNode: GainNode | null = null;
  private compressorNode: DynamicsCompressorNode | null = null;
  private mediaStreamDestination: MediaStreamAudioDestinationNode | null = null;
  private dataArray: Uint8Array | null = null;
  private isRecording = false;
  private currentGain = 2.0; // Default 2.0x boost for distant lectures

  public async startRecording(): Promise<boolean> {
    try {
      this.audioChunks = [];
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: false, // Turn off aggressive noise suppression so distant speech isn't eaten
          autoGainControl: true,
        },
      });

      this.audioStream = stream;

      let recordStream = stream;

      // Setup Web Audio API with GainNode & DynamicsCompressor for distant lecture amplification
      const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
      if (AudioContextClass) {
        this.audioContext = new AudioContextClass();
        if (this.audioContext.state === 'suspended') {
          await this.audioContext.resume();
        }

        const source = this.audioContext.createMediaStreamSource(stream);

        // Sensitivity gain boost
        this.gainNode = this.audioContext.createGain();
        this.gainNode.gain.setValueAtTime(this.currentGain, this.audioContext.currentTime);

        // Dynamic range compressor to prevent clipping and bring up quiet speech
        this.compressorNode = this.audioContext.createDynamicsCompressor();
        this.compressorNode.threshold.setValueAtTime(-24, this.audioContext.currentTime);
        this.compressorNode.knee.setValueAtTime(30, this.audioContext.currentTime);
        this.compressorNode.ratio.setValueAtTime(12, this.audioContext.currentTime);
        this.compressorNode.attack.setValueAtTime(0.003, this.audioContext.currentTime);
        this.compressorNode.release.setValueAtTime(0.25, this.audioContext.currentTime);

        // Analyser for spectrum visualization
        this.analyser = this.audioContext.createAnalyser();
        this.analyser.fftSize = 64;

        // Destination for amplified recording
        this.mediaStreamDestination = this.audioContext.createMediaStreamDestination();

        // Connect chain: source -> gain -> compressor -> analyser & destination
        source.connect(this.gainNode);
        this.gainNode.connect(this.compressorNode);
        this.compressorNode.connect(this.analyser);
        this.compressorNode.connect(this.mediaStreamDestination);

        recordStream = this.mediaStreamDestination.stream;
        this.dataArray = new Uint8Array(this.analyser.frequencyBinCount);
      }

      // Check supported MIME type for recording
      let mimeType = 'audio/webm;codecs=opus';
      if (!MediaRecorder.isTypeSupported(mimeType)) {
        mimeType = MediaRecorder.isTypeSupported('audio/webm')
          ? 'audio/webm'
          : MediaRecorder.isTypeSupported('audio/mp4')
          ? 'audio/mp4'
          : '';
      }

      const options = mimeType ? { mimeType } : undefined;
      this.mediaRecorder = new MediaRecorder(recordStream, options);

      this.mediaRecorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          this.audioChunks.push(event.data);
        }
      };

      this.mediaRecorder.start(1000); // 1-second chunks
      this.isRecording = true;
      return true;
    } catch (err) {
      console.error('Failed to access microphone or start MediaRecorder:', err);
      return false;
    }
  }

  public setGain(multiplier: number) {
    this.currentGain = Math.max(0.5, Math.min(6.0, multiplier));
    if (this.gainNode && this.audioContext) {
      this.gainNode.gain.setValueAtTime(this.currentGain, this.audioContext.currentTime);
    }
  }

  public getGain(): number {
    return this.currentGain;
  }

  public getAudioFrequencyData(): Uint8Array | null {
    if (!this.analyser || !this.dataArray) return null;
    this.analyser.getByteFrequencyData(this.dataArray as any);
    return this.dataArray;
  }

  public pause() {
    if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
      this.mediaRecorder.pause();
    }
  }

  public resume() {
    if (this.mediaRecorder && this.mediaRecorder.state === 'paused') {
      this.mediaRecorder.resume();
    }
  }

  public stopRecording(): Promise<Blob | null> {
    return new Promise((resolve) => {
      if (!this.mediaRecorder || this.mediaRecorder.state === 'inactive') {
        this.cleanup();
        resolve(null);
        return;
      }

      this.mediaRecorder.onstop = () => {
        const mimeType = this.mediaRecorder?.mimeType || 'audio/webm';
        const audioBlob = new Blob(this.audioChunks, { type: mimeType });
        this.cleanup();
        resolve(audioBlob);
      };

      this.mediaRecorder.stop();
      this.isRecording = false;
    });
  }

  private cleanup() {
    if (this.audioStream) {
      this.audioStream.getTracks().forEach((track) => track.stop());
      this.audioStream = null;
    }
    if (this.audioContext) {
      this.audioContext.close().catch(() => {});
      this.audioContext = null;
    }
    this.analyser = null;
    this.gainNode = null;
    this.compressorNode = null;
    this.mediaStreamDestination = null;
    this.dataArray = null;
  }

  public getIsRecording(): boolean {
    return this.isRecording;
  }
}
