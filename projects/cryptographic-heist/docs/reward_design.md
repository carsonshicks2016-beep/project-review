# Reward Design

The reward model is intentionally componentized so training can inspect why an
agent is improving or collapsing.

Evader reward:
- survival time
- progress toward current waypoint
- waypoint hits
- successful deception through jamming
- penalties for capture, severe impacts, and wasted jamming

Pursuer reward:
- team pressure and distance reduction
- containment and capture
- small individual role/spacing shaping
- penalties for building impacts, team pileups, waypoint losses, and spoof
  susceptibility

Information rewards:
- decoder accuracy is measured by queuing scanner predictions and settling
  them against realized future pursuer positions after a short policy horizon
- pursuers receive an authentication penalty when the evader decoder is accurate
  or spoof offsets remain active on the police formation
- the evader receives information reward from successful jamming/deception and
  active spoof susceptibility
- PPO checkpoints record `decoder_accuracy`, `spoof_susceptibility`,
  `pursuer_auth_penalty`, `evader_information_reward`, and
  `counterfactual_deception` in their stats
- the counterfactual jammer trainer additionally labels spoof actions by paired
  jammed-vs-baseline pursuer trajectory deviation, confidence damage, and
  deception lift
- token entropy and coordination metrics should be tracked during learned comms
