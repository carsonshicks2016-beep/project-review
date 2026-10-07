import {renderAudio} from './physics.mjs';

self.onmessage = ({data}) => {
  try {
    const result = renderAudio(data.config);
    const transfer = [];
    const seen = new Set();
    for (const buf of [result.samples?.buffer, result.samplesL?.buffer, result.samplesR?.buffer]) {
      if (buf && !seen.has(buf)) {
        seen.add(buf);
        transfer.push(buf);
      }
    }
    self.postMessage({id: data.id, ...result}, transfer);
  } catch (error) {
    self.postMessage({id: data.id, error: error.message});
  }
};
