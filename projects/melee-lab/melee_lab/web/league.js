// League controls use the same checkpoint catalog as the existing training form.
const leagueControls=document.createElement('div');
leagueControls.innerHTML=`<label>Controller rules<select id="execution-mode"><option value="assisted">Recovery assistance · identical for both policies</option><option value="raw">Raw neural controller · no assistance</option></select></label>
<label class="check"><input id="league-enabled" type="checkbox"> Historical opponent league</label>
<label id="league-history-label" hidden>Historical opponents<select id="league-history" multiple size="5"></select></label>
<p id="league-description" class="help" hidden>The selected starting policy is the champion. Matches sample 40% champion, 40% historical policies, and 20% varied CPU 7–9 opponents. Choose up to 12 historical policies. With none selected, the champion fills the historical pool. Candidates are retained every 250,000 decisions.</p>`;
$('launch').before(leagueControls);
function refreshLeagueControls(){
  const active=$('league-enabled').checked && $('mode').value==='train';
  $('league-history-label').hidden=$('league-description').hidden=!active;
  if(active){$('opponent-policy').value='';$('curriculum').checked=false;}
  const selected=new Set(Array.from($('league-history').selectedOptions,o=>o.value));
  const character=state?.checkpoints?.find(c=>c.path===$('checkpoint').value)?.config?.character;
  const entries=(state?.checkpoints||[]).filter(c=>c.config?.character===character&&c.config?.action_set==='controller'&&!c.path.includes('opponent-policy'));
  const signature=JSON.stringify(entries.map(c=>[c.path,c.name]));
  if($('league-history').dataset.signature!==signature){
    $('league-history').replaceChildren(...entries.map(c=>new Option(c.name,c.path,false,selected.has(c.path))));
    $('league-history').dataset.signature=signature;
  }
}
$('league-enabled').addEventListener('change',refreshLeagueControls);
$('mode').addEventListener('change',refreshLeagueControls);
$('checkpoint').addEventListener('change',refreshLeagueControls);
setInterval(refreshLeagueControls,2000);
