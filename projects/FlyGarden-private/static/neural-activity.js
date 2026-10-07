// Shared conventions: (time - 100 ms, time], with 1 ns numerical tolerance.
export function activityChunkIndices(chunks,time,window=.1){const out=[];for(let i=0;i<chunks.length;i++)if(chunks[i].end>=time-window-1e-9&&chunks[i].start<=time+1e-9)out.push(i);return out;}
export function eventActivity(chunks,time){const rates={},lastSpikes={};for(const chunk of chunks)for(let i=0;i<chunk.times.length;i++){const stamp=chunk.times[i],neuron=chunk.indices[i];if(stamp>time+1e-9||stamp<=time-.1+1e-9)continue;rates[neuron]=(rates[neuron]||0)+10;lastSpikes[neuron]=Math.max(lastSpikes[neuron]??-Infinity,stamp);}return {rates,lastSpikes};}

// Async retrieval only populates the cache. Activity is always reconstructed at
// the current display timestamp; a missing chunk must never reuse old colors.
export function cachedActivity(chunks,time,cache){
 const indices=activityChunkIndices(chunks,time);
 if(indices.some(index=>!cache.has(index)))return null;
 return eventActivity(indices.map(index=>cache.get(index)),time);
}
