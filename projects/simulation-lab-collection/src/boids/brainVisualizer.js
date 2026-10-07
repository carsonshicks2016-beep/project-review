/**
 * ApexFlock - Organic Connectome Brain Visualizer HUD
 * Exhibition-grade neural network rendering:
 * - Dynamic synaptic signal packet pulses flowing through connections
 * - Ethereal gold (excitatory) & carmine (inhibitory) synaptic connections
 * - Concentric pearlescent neural bodies with activation aura
 * - Museum-grade typography and telemetry
 */

export class BrainVisualizer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.agent = null;

    this.layout = null;
    this.lastTopologyKey = "";
    this.pulsePhase = 0;

    // Sensor definitions with categories
    this.boidInputSensors = [
      { name: "Flk dx", cat: "flock" },
      { name: "Flk dy", cat: "flock" },
      { name: "Flk dz", cat: "flock" },
      { name: "Flk dist", cat: "flock" },
      { name: "Flk align", cat: "flock" },
      { name: "Ctr dx", cat: "flock" },
      { name: "Ctr dy", cat: "flock" },
      { name: "Ctr dz", cat: "flock" },
      { name: "Density", cat: "flock" },
      { name: "Pred1 dx", cat: "threat" },
      { name: "Pred1 dy", cat: "threat" },
      { name: "Pred1 dz", cat: "threat" },
      { name: "Pred1 urg", cat: "threat" },
      { name: "Closing", cat: "threat" },
      { name: "Pred2 dx", cat: "threat" },
      { name: "Pred2 dy", cat: "threat" },
      { name: "Pred2 dz", cat: "threat" },
      { name: "Pred2 dist", cat: "threat" },
      { name: "Food dx", cat: "food" },
      { name: "Food dy", cat: "food" },
      { name: "Food dz", cat: "food" },
      { name: "Food dist", cat: "food" },
      { name: "Pheromone", cat: "env" },
      { name: "Whisker", cat: "env" },
      { name: "Stamina", cat: "state" },
      { name: "Speed", cat: "state" }
    ];

    this.predInputSensors = [
      { name: "Prey dx", cat: "prey" },
      { name: "Prey dy", cat: "prey" },
      { name: "Prey dz", cat: "prey" },
      { name: "Prey dist", cat: "prey" },
      { name: "Prey rate", cat: "prey" },
      { name: "Vuln dx", cat: "prey" },
      { name: "Vuln dy", cat: "prey" },
      { name: "Vuln dz", cat: "prey" },
      { name: "Vuln stam", cat: "prey" },
      { name: "Flk CtrX", cat: "flock" },
      { name: "Flk CtrY", cat: "flock" },
      { name: "Flk CtrZ", cat: "flock" },
      { name: "Flk Size", cat: "flock" },
      { name: "Pack dx", cat: "pack" },
      { name: "Pack dy", cat: "pack" },
      { name: "Pack dz", cat: "pack" },
      { name: "Pack dist", cat: "pack" },
      { name: "Pack align", cat: "pack" },
      { name: "Pack howl", cat: "pack" },
      { name: "Pack strike", cat: "pack" },
      { name: "Sight", cat: "env" },
      { name: "Whisker", cat: "env" },
      { name: "Hunger", cat: "state" },
      { name: "Stamina", cat: "state" },
      { name: "Speed", cat: "state" },
      { name: "Starve", cat: "state" }
    ];

    this.boidOutputLabels = ["Pitch", "Yaw", "Roll", "Boost", "Alarm"];
    this.predOutputLabels = ["Pitch", "Yaw", "Roll", "Pounce", "Howl"];
  }

  setAgent(agent) {
    this.agent = agent;
    this.layout = null;
  }

  _computeLayout(topology, width, height) {
    const numLayers = topology.length;
    const paddingLeft = 58;
    const paddingRight = 64;
    const paddingTop = 22;
    const paddingBottom = 16;
    const layerSpacing = (width - paddingLeft - paddingRight) / (numLayers - 1);

    const layout = [];
    for (let l = 0; l < numLayers; l++) {
      const count = topology[l];
      const x = paddingLeft + l * layerSpacing;
      const layerNodes = [];
      const usableHeight = height - paddingTop - paddingBottom;
      const nodeSpacing = usableHeight / Math.max(1, count - 1);

      for (let i = 0; i < count; i++) {
        const y = paddingTop + i * nodeSpacing;
        layerNodes.push({ x, y });
      }
      layout.push(layerNodes);
    }
    return layout;
  }

  render() {
    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;
    this.pulsePhase += 0.04;

    // Velvety dark background
    ctx.fillStyle = 'rgba(8, 12, 18, 0.96)';
    ctx.fillRect(0, 0, w, h);

    if (!this.agent || !this.agent.alive) {
      ctx.fillStyle = 'rgba(180, 170, 155, 0.5)';
      ctx.font = '12px "Cormorant Garamond", Georgia, serif';
      ctx.textAlign = 'center';
      ctx.fillText("Select an agent in the biosphere", w / 2, h / 2 - 10);
      ctx.font = '10px "Space Grotesk", sans-serif';
      ctx.fillText("to observe neural connectome activations", w / 2, h / 2 + 10);
      return;
    }

    const brain = this.agent.brain;
    const topology = brain.topology;
    const key = topology.join('-');
    const netAreaHeight = h - 68;

    if (!this.layout || this.lastTopologyKey !== key) {
      this.layout = this._computeLayout(topology, w, netAreaHeight);
      this.lastTopologyKey = key;
    }

    const isPred = (this.agent.type === 'predator');
    const sensors = isPred ? this.predInputSensors : this.boidInputSensors;
    const outLabels = isPred ? this.predOutputLabels : this.boidOutputLabels;

    // 1. DRAW SYNAPSE CONNECTIONS
    for (let l = 0; l < brain.numLayers - 1; l++) {
      const fromNodes = this.layout[l];
      const toNodes = this.layout[l + 1];
      const weights = brain.weights[l];
      const inCount = topology[l];
      const outCount = topology[l + 1];

      for (let j = 0; j < outCount; j++) {
        for (let i = 0; i < inCount; i++) {
          const weight = weights[j * inCount + i];
          if (Math.abs(weight) < 0.08) continue;

          const n1 = fromNodes[i];
          const n2 = toNodes[j];
          const alpha = Math.min(0.8, 0.12 + Math.abs(weight) * 0.45);

          ctx.beginPath();
          ctx.moveTo(n1.x, n1.y);
          ctx.lineTo(n2.x, n2.y);
          ctx.lineWidth = Math.min(2.0, 0.5 + Math.abs(weight) * 0.7);

          if (weight > 0) {
            // Excitatory: warm champagne gold / amber
            ctx.strokeStyle = `rgba(217, 180, 110, ${alpha})`;
          } else {
            // Inhibitory: delicate carmine / rose
            ctx.strokeStyle = `rgba(210, 88, 104, ${alpha})`;
          }
          ctx.stroke();

          // Animated neural signal packet (traveling along the connection wire)
          if (Math.abs(weight) > 0.45 && (i + j) % 3 === 0) {
            const t = (this.pulsePhase + (i * 0.2 + j * 0.3)) % 1.0;
            const px = n1.x + (n2.x - n1.x) * t;
            const py = n1.y + (n2.y - n1.y) * t;
            ctx.beginPath();
            ctx.arc(px, py, 1.2, 0, Math.PI * 2);
            ctx.fillStyle = weight > 0 ? '#fff3d1' : '#ffb3ba';
            ctx.fill();
          }
        }
      }
    }

    // 2. DRAW NEURONS
    for (let l = 0; l < brain.numLayers; l++) {
      const nodes = this.layout[l];
      const acts = brain.activations[l];
      const isInput = (l === 0);
      const isOutput = (l === brain.numLayers - 1);

      for (let i = 0; i < nodes.length; i++) {
        const node = nodes[i];
        const val = acts[i] || 0;
        const normVal = Math.max(-1, Math.min(1, val));
        const r = isInput || isOutput ? 3.4 : 2.5;

        // Outer aura ring
        ctx.beginPath();
        ctx.arc(node.x, node.y, r + 1.2, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.12)';
        ctx.lineWidth = 0.8;
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(node.x, node.y, r, 0, Math.PI * 2);

        if (isOutput) {
          const intensity = Math.abs(normVal);
          ctx.fillStyle = normVal >= 0 
            ? `rgba(126, 199, 176, ${0.4 + intensity * 0.6})`
            : `rgba(228, 188, 115, ${0.4 + intensity * 0.6})`;
          ctx.fill();

          ctx.font = '8px "Space Grotesk", sans-serif';
          ctx.textAlign = 'left';
          ctx.fillStyle = normVal >= 0 ? '#7ec7b0' : '#e4bc73';
          ctx.fillText(`${outLabels[i]}: ${normVal.toFixed(1)}`, node.x + 8, node.y + 2.5);
        } else if (isInput) {
          const intensity = Math.abs(normVal);
          ctx.fillStyle = normVal >= 0
            ? `rgba(217, 180, 110, ${0.35 + intensity * 0.65})`
            : `rgba(210, 88, 104, ${0.35 + intensity * 0.65})`;
          ctx.fill();

          if (sensors[i] && (i % 2 === 0 || intensity > 0.4)) {
            ctx.font = '7.5px "Space Grotesk", sans-serif';
            ctx.textAlign = 'right';
            ctx.fillStyle = intensity > 0.4 ? '#f2efe9' : 'rgba(160, 170, 185, 0.6)';
            ctx.fillText(sensors[i].name, node.x - 7, node.y + 2.5);
          }
        } else {
          ctx.fillStyle = normVal > 0 
            ? `rgba(217, 180, 110, ${0.25 + Math.min(1, normVal) * 0.65})`
            : `rgba(45, 55, 70, 0.5)`;
          ctx.fill();
        }
      }
    }

    // 3. FOOTER TELEMETRY
    const footerY = h - 62;
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(10, footerY);
    ctx.lineTo(w - 10, footerY);
    ctx.stroke();

    if (isPred) {
      ctx.font = '600 11px "Cormorant Garamond", Georgia, serif';
      ctx.fillStyle = '#d25868';
      ctx.fillText(`APEX #${this.agent.id} · GEN ${this.agent.generation}`, 12, footerY + 15);
      ctx.font = '8.5px "Space Grotesk", sans-serif';
      ctx.fillStyle = '#f2efe9';
      ctx.fillText(`Role: ${(this.agent.role || 'Hunter').replace('_', ' ').toUpperCase()}`, 12, footerY + 28);
      ctx.fillStyle = '#98a2b3';
      ctx.fillText(`Kills: ${this.agent.kills}  ·  Starve: ${Math.round(this.agent.timeSinceLastKill)}s`, 12, footerY + 41);
    } else {
      ctx.font = '600 11px "Cormorant Garamond", Georgia, serif';
      ctx.fillStyle = '#7ec7b0';
      ctx.fillText(`PREY #${this.agent.id} · GEN ${this.agent.generation}`, 12, footerY + 15);
      ctx.font = '8.5px "Space Grotesk", sans-serif';
      const draftTxt = this.agent.isDrafting ? `Drafting (+${Math.round(this.agent.draftingEfficiency * 100)}%)` : "Free Flight";
      ctx.fillStyle = this.agent.isDrafting ? '#d9b46e' : '#f2efe9';
      ctx.fillText(`Lineage #${this.agent.lineageId}  ·  ${draftTxt}`, 12, footerY + 28);
      ctx.fillStyle = '#98a2b3';
      ctx.fillText(`Food: ${this.agent.foodEaten}  ·  Evasions: ${this.agent.evasions}`, 12, footerY + 41);
    }

    // Energy & Stamina gauges
    const barX = w - 85;
    const barW = 74;
    const barH = 5;

    // Energy Bar
    const maxNrg = isPred ? 200 : 180;
    const eRatio = Math.max(0, Math.min(1, this.agent.energy / maxNrg));
    ctx.fillStyle = 'rgba(255, 255, 255, 0.08)';
    ctx.fillRect(barX, footerY + 18, barW, barH);
    ctx.fillStyle = isPred ? '#d25868' : '#7ec7b0';
    ctx.fillRect(barX, footerY + 18, barW * eRatio, barH);

    ctx.font = '8px "Space Grotesk", sans-serif';
    ctx.fillStyle = '#98a2b3';
    ctx.textAlign = 'left';
    ctx.fillText(`ENERGY ${Math.round(this.agent.energy)}`, barX, footerY + 14);

    // Stamina Bar
    const sRatio = Math.max(0, Math.min(1, this.agent.stamina / (this.agent.genome.staminaMax || 100)));
    ctx.fillStyle = 'rgba(255, 255, 255, 0.08)';
    ctx.fillRect(barX, footerY + 36, barW, barH);
    ctx.fillStyle = '#d9b46e';
    ctx.fillRect(barX, footerY + 36, barW * sRatio, barH);
    ctx.fillText(`STAMINA ${Math.round(this.agent.stamina)}`, barX, footerY + 32);
  }
}
