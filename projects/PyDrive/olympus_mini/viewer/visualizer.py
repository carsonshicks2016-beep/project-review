import os
import numpy as np

class LiveViewer:
    """
    Lightweight Pygame Interactive Viewer for Olympus Mini.
    Allows real-time playback and manual inspection.
    """
    def __init__(self, width=640, height=480, title="Olympus Mini - Live Viewer"):
        self.width = width
        self.height = height
        self.title = title
        self.screen = None
        self.clock = None
        self.is_headless = os.environ.get("SDL_VIDEODRIVER", "") == "dummy"

    def init_display(self):
        if self.is_headless:
            return False
        try:
            import pygame
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption(self.title)
            self.clock = pygame.time.Clock()
            return True
        except Exception as e:
            print(f"Live display initialization failed ({e}). Falling back to headless.")
            self.is_headless = True
            return False

    def render_frame(self, frame_rgb, fps=30):
        if self.is_headless or self.screen is None:
            return False
        import pygame
        # Convert RGB numpy array to Pygame Surface
        surf = pygame.surfarray.make_surface(np.transpose(frame_rgb, (1, 0, 2)))
        self.screen.blit(surf, (0, 0))
        pygame.display.flip()
        self.clock.tick(fps)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
        return True

    def close(self):
        if not self.is_headless and self.screen is not None:
            try:
                import pygame
                pygame.quit()
            except:
                pass
            self.screen = None
