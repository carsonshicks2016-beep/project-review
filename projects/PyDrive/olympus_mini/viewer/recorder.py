import os
import numpy as np
import imageio
from PIL import Image, ImageDraw, ImageFont
import cv2

class EpisodeRecorder:
    """
    Renders MuJoCo simulation frames offscreen and saves to GIF or MP4
    with real-time HUD telemetry overlay.
    """
    def __init__(self, fps=30):
        self.fps = fps
        self.frames = []

    def add_frame(self, frame_rgb, telemetry=None):
        """
        Add a frame with optional telemetry dictionary overlay:
        e.g. {'task': 'Sprint', 'speed': 4.2, 'dist': 18.5, 'height': 0.98}
        """
        img = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(img)
        
        if telemetry:
            # Render HUD background banner
            draw.rectangle([(10, 10), (320, 110)], fill=(0, 0, 0, 180), outline=(255, 140, 0))
            
            lines = [
                f"OLYMPUS MINI: {telemetry.get('task', 'ATHLETE').upper()}",
                f"Speed: {telemetry.get('vx', 0.0):.2f} m/s",
                f"Distance: {telemetry.get('dist', 0.0):.2f} m",
                f"Height: {telemetry.get('height', 0.0):.2f} m",
                f"Step: {telemetry.get('step', 0)}"
            ]
            y_offset = 16
            for line in lines:
                draw.text((20, y_offset), line, fill=(255, 255, 255))
                y_offset += 18
                
        self.frames.append(np.array(img))

    def save_gif(self, output_path, loop=0):
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        if len(self.frames) == 0:
            print("Warning: No frames to save.")
            return
        imageio.mimsave(output_path, self.frames, fps=self.fps, loop=loop)
        print(f"Saved GIF ({len(self.frames)} frames) to {output_path}")

    def save_video(self, output_path):
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        if len(self.frames) == 0:
            print("Warning: No frames to save.")
            return
        h, w, _ = self.frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (w, h))
        for f in self.frames:
            # Convert RGB to BGR for OpenCV
            out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        out.release()
        print(f"Saved MP4 video to {output_path}")

    def reset(self):
        self.frames = []
