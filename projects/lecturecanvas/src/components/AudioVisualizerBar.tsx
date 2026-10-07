import React, { useEffect, useRef } from 'react';
import type { ColorTheme } from '../types';
import { THEMES } from '../utils/themeStyles';

interface Props {
  getAudioData: () => Uint8Array | null;
  isRecording: boolean;
  theme: ColorTheme;
}

export const AudioVisualizerBar: React.FC<Props> = ({
  getAudioData,
  isRecording,
  theme,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const themeColors = THEMES[theme];

  useEffect(() => {
    let animationId: number;

    const draw = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const width = canvas.width;
      const height = canvas.height;

      ctx.clearRect(0, 0, width, height);

      const freqData = isRecording ? getAudioData() : null;
      const barCount = 28;
      const barWidth = 3;
      const gap = (width - barCount * barWidth) / (barCount - 1);

      for (let i = 0; i < barCount; i++) {
        let barHeight = 4; // minimum height

        if (freqData && isRecording) {
          // Sample frequency data proportionally
          const index = Math.floor((i / barCount) * freqData.length);
          const val = freqData[index] || 0;
          barHeight = Math.max(4, (val / 255) * (height - 4));
        }

        const x = i * (barWidth + gap);
        const y = (height - barHeight) / 2;

        ctx.fillStyle = isRecording ? themeColors.palette[i % themeColors.palette.length] : 'rgba(148, 163, 184, 0.2)';
        ctx.beginPath();
        ctx.roundRect(x, y, barWidth, barHeight, 2);
        ctx.fill();
      }

      animationId = requestAnimationFrame(draw);
    };

    draw();
    return () => cancelAnimationFrame(animationId);
  }, [getAudioData, isRecording, themeColors]);

  return (
    <div className="flex items-center gap-2 px-2.5 py-1 rounded-full bg-black/40 border border-white/10 backdrop-blur-md">
      <div className={`w-2 h-2 rounded-full ${isRecording ? 'bg-red-500 recording-pulse' : 'bg-slate-500'}`} />
      <canvas ref={canvasRef} width={140} height={20} className="w-[140px] h-[20px]" />
    </div>
  );
};
