import { App } from './app';

async function main(): Promise<void> {
  const canvas = document.getElementById('sim') as HTMLCanvasElement | null;
  const toolbar = document.getElementById('toolbar');
  const panel = document.getElementById('panel');
  const hud = document.getElementById('hud');
  if (!canvas || !toolbar || !panel || !hud) {
    throw new Error('Missing #sim, #toolbar, #panel, or #hud');
  }

  const app = new App(canvas, toolbar, panel, hud);
  await app.init();
  app.start();

  (window as unknown as { __sandbox?: App }).__sandbox = app;
}

main().catch((err) => {
  console.error(err);
  const hud = document.getElementById('hud');
  if (hud) hud.textContent = `Failed to start: ${err}`;
});
