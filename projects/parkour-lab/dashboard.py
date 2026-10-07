"""Parkour Lab: an evidence-first control room for MuJoCo learning."""
from __future__ import annotations

from datetime import datetime
import html
import json
import math
from pathlib import Path
import time

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import psutil
import streamlit as st
import yaml

from dashboard_data import (
    ROOT, STAGE_NAMES, KINDS, append_segment, atomic_write, checkpoint_choices,
    checkpoint_evidence, discover_runs, finite, launch_training, new_course,
    normalize_evaluation, read_json, read_yaml, stop_training, trainers,
    validate_config, validate_course, verify_process, videos,
)

st.set_page_config(page_title="Parkour Lab · Learning in motion", page_icon="◈", layout="wide", initial_sidebar_state="expanded")
COLORS = ["#0B8D9E", "#F07D45", "#6C71C4", "#517855", "#B55B80", "#7C8FA8"]
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root { --paper:#F5F5F0; --ink:#172C3B; --muted:#667980; --cyan:#0B8D9E; }
html, body, [class*="css"], .stApp { font-family:'DM Sans',sans-serif; color:var(--ink); }
.stApp, [data-testid="stAppViewContainer"] { background:var(--paper); }
.block-container { max-width:1550px; padding:2rem 2.8rem 3rem; }
header[data-testid="stHeader"] { background:rgba(245,245,240,.95); }
[data-testid="stSidebar"] { background:#E9EDE9; border-right:1px solid #D6DFD9; }
[data-testid="stSidebar"] .block-container { padding:1.6rem; }
h1,h2,h3,h4 { font-family:'Space Grotesk',sans-serif!important; letter-spacing:-.04em; color:var(--ink); }
h1 { font-size:2.4rem!important; } h2 {font-size:1.55rem!important;} h3 {font-size:1.08rem!important;}
p,li {line-height:1.65;} .stCaption {color:var(--muted);}
[data-testid="stMetric"] {background:#FFF; border:1px solid #E0E5DF; border-radius:14px; padding:17px 18px; min-height:123px;}
[data-testid="stMetricValue"] {font-family:'Space Grotesk',sans-serif; font-size:2rem; color:var(--ink); letter-spacing:-.04em;}
[data-testid="stMetricLabel"] {font-size:.78rem; text-transform:uppercase; letter-spacing:.065em; color:#6B7A7D;}
[data-testid="stVerticalBlockBorderWrapper"] > div {border-color:#DFE5DF!important; border-radius:16px!important;}
[data-baseweb="select"] > div, [data-baseweb="input"] {background:#FFF; border-color:#D8E1DB;}
.stButton > button, .stDownloadButton > button {border-radius:9px; border:1px solid #CCD9D3; background:white; font-weight:600;}
.stButton > button[kind="primary"] {background:#0B8D9E; border-color:#0B8D9E; color:white;}
[data-testid="stDataFrame"] {border-radius:10px;}
.brand {display:flex; align-items:center; gap:12px; margin:0 0 23px;}
.brand-mark {background:#172C3B; color:#D8F9F2; font-family:'Space Grotesk',sans-serif; font-size:20px; width:43px; height:43px; border-radius:12px; display:flex; justify-content:center; align-items:center;}
.brand-name {font-size:19px; font-weight:700; letter-spacing:-.6px; line-height:1.2;}
.brand-sub {font-size:10px; text-transform:uppercase; letter-spacing:2px; color:#6C7E7C; margin-top:5px;}
.eyebrow {color:#6A7A7D; font:600 10px 'DM Sans',sans-serif; letter-spacing:2px; text-transform:uppercase; margin-bottom:10px;}
.hero {position:relative; overflow:hidden; border-radius:20px; background:#172C3B; color:#EEF7F5; padding:30px 34px; margin-bottom:22px;}
.hero:after {content:''; position:absolute; right:-75px; top:-125px; width:380px; height:380px; border:1px solid #395460; border-radius:50%; box-shadow:0 0 0 50px #1B3342,0 0 0 51px #314C59,0 0 0 100px #1A3040; opacity:.5; pointer-events:none;}
.hero > * {position:relative; z-index:1;} .hero .eyebrow {color:#85BBBD;}
.hero h1 {color:#F2F6F1!important; margin:10px 0 10px; max-width:730px; font-size:2.65rem!important; line-height:1.13;}
.hero p {color:#B7CFD2; font-size:14px; max-width:700px; margin:0;}
.pill {display:inline-block; padding:5px 10px; background:#294C53; color:#B5E6DD; border:1px solid #3C6868; border-radius:30px; font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:1px; margin-right:8px;}
.pill.warn {background:#574732; color:#F3C49E; border-color:#78613E;}
.hero-meta {margin-top:20px; display:flex; flex-wrap:wrap; gap:8px;}
.section-title {font-family:'Space Grotesk',sans-serif; font-size:1.22rem; font-weight:600; letter-spacing:-.03em; margin:7px 0 3px;}
.section-sub {font-size:12px; color:#708081; margin-bottom:16px;}
.checkpoint-tag {font-size:10px; letter-spacing:1.2px; color:#0B8D9E; font-weight:700; text-transform:uppercase;}
.note {border-left:3px solid #0B8D9E; padding:10px 14px; background:#E9F1ED; border-radius:0 8px 8px 0; font-size:12px; color:#526969; margin:12px 0;}
.journey {display:flex; gap:7px; margin:4px 0 23px; flex-wrap:wrap;}
.journey-step {padding:10px 13px; background:#E7EDE8; color:#74847F; border-radius:10px; flex:1; min-width:110px; font-size:11px;}
.journey-step strong {display:block; font-size:12px; color:#506862; margin-top:3px;}
.journey-step.current {background:#D2E8DF; border:1px solid #ADCAC0; color:#267268;}
.journey-step.current strong {color:#1B6058;}
.small-table {width:100%; border-collapse:collapse; font-size:12px;}
.small-table td {padding:9px 0; border-bottom:1px solid #E1E7E0;}.small-table td:last-child {text-align:right; font-weight:600;}
.footer {font-size:10px; letter-spacing:.5px; color:#7F8B85; border-top:1px solid #DDE4DC; padding-top:17px; margin-top:25px;}
@media(max-width:900px) { .block-container {padding:1.2rem 1rem 2rem;} .hero{padding:24px 22px;} .hero h1{font-size:2rem!important;} [data-testid="stMetricValue"]{font-size:1.5rem;} [data-testid="stMetric"]{padding:12px 13px;min-height:110px;} }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def esc(value):
    return html.escape(str(value))


def fmt(value, digits=1, suffix=""):
    return f"{value:,.{digits}f}{suffix}" if finite(value) else "—"


def compact(value):
    if not finite(value): return "—"
    if value >= 1e6: return f"{value / 1e6:.2f}M"
    if value >= 1e3: return f"{value / 1e3:.1f}K"
    return f"{value:,.0f}"


def pct(value):
    return f"{100 * value:.0f}%" if finite(value) else "—"


def age(value):
    seconds = max(0, time.time() - value)
    if seconds < 60: return "just now"
    if seconds < 3600: return f"{int(seconds / 60)} min ago"
    if seconds < 86400: return f"{seconds / 3600:.1f} hours ago"
    return f"{seconds / 86400:.1f} days ago"


def chart_style(fig, title=None, height=300, percent=False):
    fig.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, Arial", size=11, color="#526A70"), margin=dict(l=8,r=12,t=35 if title else 12,b=8),
        title=dict(text=title or "", font=dict(size=14, color="#172C3B")), height=height,
        legend=dict(orientation="h",y=1.14,x=0), hovermode="x unified")
    fig.update_xaxes(showgrid=False, zeroline=False, title_font=dict(size=10), linecolor="#D6E0D9")
    fig.update_yaxes(gridcolor="#E0E7E0", zeroline=False, title_font=dict(size=10))
    if percent: fig.update_yaxes(tickformat=".0%")
    return fig


def line_chart(run_list, metric, label, smooth=1, percent=False, local_steps=False):
    fig = go.Figure()
    for index, run in enumerate(run_list):
        rows = [row for row in run["rows"] if finite(row.get(metric)) and finite(row.get("timesteps"))]
        if not rows: continue
        df = pd.DataFrame(rows)
        offset = run["lineage"].get("parent_steps", 0) if local_steps else 0
        x = (df["timesteps"] - offset) / 1e6
        y = df[metric].rolling(smooth, min_periods=1).mean()
        fig.add_trace(go.Scatter(x=x,y=y,name=run["name"],mode="lines+markers",line=dict(color=COLORS[index % len(COLORS)],width=2.5),marker=dict(size=4),
            hovertemplate="%{x:.3f}M steps<br>%{y:.3f}<extra>%{fullData.name}</extra>"))
    fig.update_xaxes(title="New control decisions (millions)" if local_steps else "Lifetime control decisions (millions)")
    fig.update_yaxes(title=label)
    return chart_style(fig, height=285, percent=percent)


def section(title, subtitle=""):
    st.markdown(f'<div class="section-title">{esc(title)}</div><div class="section-sub">{esc(subtitle)}</div>', unsafe_allow_html=True)


def table_html(pairs):
    st.markdown('<table class="small-table">' + ''.join(f'<tr><td>{esc(k)}</td><td>{esc(v)}</td></tr>' for k,v in pairs) + '</table>', unsafe_allow_html=True)


def metric_row(evaluation, parkour):
    columns = st.columns(4)
    with columns[0]: st.metric("Raw return", fmt(evaluation.get("mean_reward"),0), help="Unnormalized sum of environment rewards. Different reward functions are not directly comparable.")
    with columns[1]: st.metric("Full-horizon survival", pct(evaluation.get("survival_fraction")), help="Fraction reaching the episode time limit without a fall. Surviving is distinct from finishing a course.")
    with columns[2]:
        st.metric("Course progress" if parkour else "Forward speed", pct(evaluation.get("course_progress")) if parkour else fmt(evaluation.get("speed"),2," m/s"), help="Legacy completion_rate records fractional distance progress, not a success rate." if parkour else "Mean center-of-mass speed over each episode.")
    with columns[3]:
        st.metric("Courses finished" if parkour else "Travel distance", pct(evaluation.get("course_success_rate")) if parkour else fmt(evaluation.get("distance"),1," m"), help="Only explicit binary course success counts. A dash means this metric was not recorded." if parkour else "Mean net forward distance across evaluation seeds.")


def state_label(state):
    return {"completed":"Training budget finished", "incomplete":"Incomplete run", "interrupted":"Trainer interrupted", "training":"Training live", "evaluating":"Evaluating live", "starting":"Trainer starting", "stopping":"Saving & stopping", "stopped":"Stopped safely", "failed":"Run failed"}.get(state,state)


def overview(run, all_runs):
    cfg, current = run["config"], run["latest"]
    stage = cfg.get("stage", 0)
    state = run["state"]
    warning = state in ("failed","incomplete","interrupted")
    headline = {1:"Learning to find its feet.",2:"Balance first. Then momentum.",3:"Taking movement beyond flat ground.",4:"A harder course. A better question."}.get(stage,"Learning, one control decision at a time.")
    st.markdown(f'<div class="hero"><div class="eyebrow">PHYSICS / POLICY / PROGRESS</div><h1>{headline}</h1><p>{esc(run["name"])} · {esc(cfg.get("env_id","Unknown environment"))} · Every result traces back to a policy, a course, and an evaluation.</p><div class="hero-meta"><span class="pill {"warn" if warning else ""}">{esc(state_label(state))}</span><span class="pill">Stage {stage} · {esc(STAGE_NAMES.get(stage,"Research"))}</span><span class="pill">{compact(run["status"].get("timesteps",current.get("timesteps",run["performance"].get("steps"))))} decisions</span></div></div>',unsafe_allow_html=True)
    if warning:
        st.warning("This run has no verified successful terminal marker. Existing checkpoints and videos remain available for inspection.")
    if state in ("training","evaluating","starting","stopping"):
        status = run["status"]
        progress = status.get("progress")
        if finite(progress): st.progress(min(1.,max(0.,progress)), text=f'{compact(status.get("new_steps"))} new decisions · {fmt(status.get("steps_per_second"),0)} decisions/s')
    st.markdown('<div class="journey">'+''.join(f'<div class="journey-step {"current" if n==stage else ""}">0{n}<strong>{label}</strong></div>' for n,label in [(1,"Warmup"),(2,"Locomotion"),(3,"Terrain"),(4,"Curriculum"),(5,"Robustness")])+'</div>',unsafe_allow_html=True)
    checkpoints = checkpoint_choices(run)
    chosen = None
    evidence = {}
    if checkpoints:
        chosen = st.selectbox("Inspect a checkpoint", checkpoints, format_func=lambda p: f'{p.name} · {compact(read_json(p / "metadata.json",{}).get("timesteps"))} lifetime decisions', key=f"checkpoint_{run['name']}")
        evidence = checkpoint_evidence(run, chosen)
    evaluation = evidence.get("evaluation") or current
    scope = f"Checkpoint: {chosen.name}" if chosen and evidence.get("evaluation") else "Latest recorded evaluation"
    st.caption(f'{scope} · {len(evaluation.get("episodes",[]))} deterministic evaluation seeds · metrics below belong to this evaluation')
    metric_row(evaluation, cfg.get("env_id") == "ParkourHumanoid")
    st.write("")
    left, right = st.columns([1.7,1],gap="large")
    with left:
        section("See what the policy actually learned", "A video is one rollout. The metrics above summarize all evaluated seeds.")
        if evidence.get("video"):
            st.video(str(evidence["video"]))
            st.caption(f'{chosen.name} checkpoint · review update {evaluation.get("update")} · seed {evaluation.get("episodes",[{}])[0].get("seed","unknown") if evaluation.get("episodes") else "unknown"}')
        else:
            entries=videos(run)
            if entries:
                heldout=next((v for v in entries if "heldout" in v["path"].stem),entries[0])
                st.video(str(heldout["path"]))
                st.info(f'Available recording: {heldout["label"]}. This recording is not automatically attributed to the selected checkpoint; inspect its evidence in Evaluation lab.')
            else: st.info("No recording yet. The first evaluation will appear here when its video is ready.")
    with right:
        with st.container(border=True):
            st.markdown('<div class="checkpoint-tag">POLICY INSPECTOR</div>',unsafe_allow_html=True)
            st.subheader(chosen.name if chosen else "Awaiting first checkpoint")
            table_html([( "Environment",cfg.get("env_id","—")),("Training workers",cfg.get("n_envs","—")),("Policy network"," × ".join(map(str,cfg.get("net_arch",[64,64])))),("Forward speed",fmt(evaluation.get("speed"),2," m/s")),("Upright alignment",fmt(evaluation.get("upright"),3)),("Facing alignment",fmt(evaluation.get("facing"),3)),("Curriculum level",fmt(evaluation.get("curriculum_level"),1)),("Evidence updated",age(run["updated"]))])
            if chosen:
                metadata=evidence.get("metadata",{})
                digest=metadata.get("sha256",{}).get("policy.zip")
                if digest: st.caption(f"Policy SHA-256 · {digest[:16]}…")
                st.code(str(chosen.relative_to(ROOT)),language=None)
                st.caption("The policy and its observation normalizer are a pair. Keep both files when evaluating or resuming.")
        if cfg.get("env_id") == "ParkourHumanoid" and evaluation.get("course_success_rate") is None:
            st.markdown('<div class="note">Earlier runs measured fractional course progress. A high reward or clearance score alone does not establish that the course was finished.</div>',unsafe_allow_html=True)
    section("Learning signal", "Unsmoothed periodic evaluations; hover to inspect the actual values.")
    a,b=st.columns(2)
    with a: st.plotly_chart(line_chart([run],"mean_reward","Raw return"),width="stretch",key=f"room_reward_{run['name']}")
    with b: st.plotly_chart(line_chart([run],"survival_fraction","Full-horizon survival",percent=True),width="stretch",key=f"room_survival_{run['name']}")
    lineage(run)


def lineage(run):
    with st.expander("Lineage, configuration & recorded evidence",expanded=False):
        parent=run["lineage"]
        if parent:
            source=parent.get("parent",parent.get("source","Unknown source"))
            st.write(f'**Parent checkpoint:** `{source}`')
            st.caption("Resumed optimizer state" if "parent" in parent else "Transferred policy weights into a new training run")
        else: st.caption("No parent checkpoint recorded. This run is treated as independent.")
        a,b=st.columns(2)
        with a: st.code(yaml.safe_dump(run["config"],sort_keys=False),language="yaml")
        with b:
            st.json({"state":run["state"],"performance":run["performance"],"lineage":parent},expanded=True)
            st.download_button("Download evaluation history",json.dumps(run["rows"],indent=2),file_name=f'{run["name"]}-evaluations.json',mime="application/json",key=f"download_{run['name']}")


def compare(all_runs, selected):
    section("Compare the learning, not just the score", "Separate balance, distance, and course progress across runs. Different environments and reward versions change what a return means.")
    candidates=[r for r in all_runs if r["rows"]]
    names=[r["name"] for r in candidates]
    defaults=[n for n in ("stage2","stage2-horizon","stage4-deep") if n in names]
    picked=st.multiselect("Runs to overlay",names,default=defaults or [selected["name"]],max_selections=6)
    chosen=[r for r in candidates if r["name"] in picked]
    if not chosen: st.info("Select at least one run to compare."); return
    a,b=st.columns([1,2])
    with a: smooth=st.select_slider("Rolling evaluations",options=[1,2,3,5],value=1,help="Smoothing changes the displayed curve only; metrics and raw evidence remain unchanged.")
    with b: local=st.toggle("Align runs by newly collected decisions",value=False)
    if len({r["config"].get("env_id") for r in chosen}) > 1:
        st.info("Mixed environments selected. Compare survival and movement; raw return is not a common scoring scale.")
    for pair in [(("mean_reward","Raw return",False),("survival_fraction","Full-horizon survival",True)),(("speed","Forward speed · m/s",False),("course_progress","Course progress",True))]:
        cols=st.columns(2)
        for col,(key,label,percent) in zip(cols,pair):
            with col:
                st.markdown(f"**{label}**")
                st.plotly_chart(line_chart(chosen,key,label,smooth,percent,local),width="stretch",key=f"compare_{key}")
    summaries=[]
    for r in chosen:
        ev=r["best"] or r["latest"]
        summaries.append({"Run":r["name"],"Environment":r["config"].get("env_id"),"Run state":state_label(r["state"]),"Best raw return":ev.get("mean_reward"),"Survival":ev.get("survival_fraction"),"Course progress":ev.get("course_progress"),"Courses finished":ev.get("course_success_rate"),"Evaluation seeds":len(ev.get("episodes",[]))})
    st.dataframe(pd.DataFrame(summaries),hide_index=True,width="stretch",column_config={"Survival":st.column_config.NumberColumn(format="percent"),"Course progress":st.column_config.NumberColumn(format="percent"),"Courses finished":st.column_config.NumberColumn(format="percent")})
    st.caption("Best follows the checkpoint selection rule recorded by each run. Older runs selected by raw return and used a small, repeated seed set; held-out evaluations are separate evidence.")


def evidence_lab(run):
    section("Evaluation lab", "Inspect individual seeds, failure modes, and the recording behind each result.")
    entries=videos(run)
    options=[]
    for entry in entries:
        options.append((entry["label"],entry["evaluation"],entry))
    for cp in checkpoint_choices(run)[:2]:
        ev=checkpoint_evidence(run,cp)
        if ev["evaluation"]: options.insert(0,(f"Checkpoint evidence · {cp.name}",ev["evaluation"],{"path":ev["video"],"kind":"paired"}))
    if not options: st.info("No evaluation evidence is available for this run yet."); return
    idx=st.selectbox("Evaluation or recording",range(len(options)),format_func=lambda i:options[i][0])
    label,ev,entry=options[idx]
    if entry.get("kind")=="standalone":
        st.info("Standalone evaluation. Older artifacts may not record the policy hash or environment settings, so this is retained as historical evidence unless provenance is present in the JSON.")
    if entry.get("kind")=="benchmark":
        st.info("Legacy benchmark recording. Its older success threshold is not equivalent to binary course completion.")
    if ev:
        metric_row(ev,run["config"].get("env_id")=="ParkourHumanoid")
    a,b=st.columns([1.35,1])
    with a:
        if entry.get("path") and Path(entry["path"]).exists(): st.video(str(entry["path"]))
        else: st.info("No video paired with this evaluation.")
    with b:
        terms=ev.get("mean_reward_terms") or {}
        if not terms and ev.get("episodes"):
            keys={k for e in ev["episodes"] for k in e.get("reward_terms",{})}
            terms={k:sum(e.get("reward_terms",{}).get(k,0) for e in ev["episodes"])/len(ev["episodes"]) for k in keys}
        if terms:
            section("What the reward is paying for")
            pairs=sorted(terms.items(),key=lambda kv:kv[1])
            fig=go.Figure(go.Bar(x=[v for _,v in pairs],y=[k.removeprefix("reward_").replace("_"," ") for k,_ in pairs],orientation="h",marker_color=["#F07D45" if v<0 else "#0B8D9E" for _,v in pairs]))
            st.plotly_chart(chart_style(fig,height=300),width="stretch")
            st.caption("Episode-summed terms. Survival can dominate without useful progress; forward reward can favor a fast but unstable gait.")
    episodes=ev.get("episodes",[])
    if episodes:
        section("Every seed counts", "Averages can conceal one failing course or a policy that only succeeds from a favorable reset.")
        records=[]
        for e in episodes:
            outcome="Course finished" if e.get("course_success") is True else "Fell / terminated" if e.get("terminated") else "Time limit" if e.get("truncated") else "Unknown"
            records.append({"Seed":e.get("seed"),"Outcome":outcome,"Raw return":e.get("reward"),"Decisions":e.get("length"),"Distance (m)":e.get("distance"),"Speed (m/s)":e.get("speed"),"Course progress":e.get("course_progress",e.get("completion_rate")),"Obstacles cleared":e.get("clearance_rate"),"Lateral drift (m)":e.get("lateral_distance")})
        frame=pd.DataFrame(records)
        fig=go.Figure(go.Bar(x=[str(r["Seed"]) for r in records],y=[r["Raw return"] for r in records],marker_color=["#0B8D9E" if r["Outcome"] in ("Course finished","Time limit") else "#F07D45" for r in records],customdata=[r["Outcome"] for r in records],hovertemplate="Seed %{x}<br>Return %{y:.1f}<br>%{customdata}<extra></extra>"))
        st.plotly_chart(chart_style(fig,height=220),width="stretch")
        st.dataframe(frame,hide_index=True,width="stretch",column_config={"Course progress":st.column_config.NumberColumn(format="percent"),"Obstacles cleared":st.column_config.NumberColumn(format="percent")})
        contact_rows=[]
        for e in episodes:
            for foot,value in e.get("foot_contact_fraction",{}).items():
                contact_rows.append({"Seed":str(e.get("seed")),"Foot":foot.replace("_"," "),"Contact fraction":value,"Touchdowns":e.get("foot_touchdowns",{}).get(foot)})
        if contact_rows:
            with st.expander("Gait diagnostics · left and right foot contact"):
                frame=pd.DataFrame(contact_rows)
                fig=go.Figure()
                for i,foot in enumerate(frame["Foot"].unique()):
                    f=frame[frame["Foot"]==foot]
                    fig.add_trace(go.Bar(x=f["Seed"],y=f["Contact fraction"],name=foot,marker_color=COLORS[i]))
                fig.update_layout(barmode="group")
                st.plotly_chart(chart_style(fig,height=250,percent=True),width="stretch")
                st.caption("A persistent imbalance can reveal one-legged hopping or shuffling. Contact fraction alone does not establish a natural gait.")
    with st.expander("Raw evaluation & provenance"):
        st.json(ev,expanded=False)


def curriculum(run):
    section("Curriculum trajectory", "Difficulty is a training condition, not a certification. Inspect the result at each level before making the course harder.")
    a,b=st.columns(2)
    with a:
        st.plotly_chart(line_chart([run],"curriculum_level","Curriculum level"),width="stretch")
    with b:
        fig=go.Figure()
        for i,key in enumerate(("course_progress","clearance_rate","course_success_rate")):
            rows=[r for r in run["rows"] if finite(r.get(key))]
            fig.add_trace(go.Scatter(x=[r.get("timesteps",0)/1e6 for r in rows],y=[r[key] for r in rows],name={"course_progress":"Course progress","clearance_rate":"Obstacle clearance","course_success_rate":"Courses finished"}[key],line=dict(color=COLORS[i])))
        st.plotly_chart(chart_style(fig,height=285,percent=True),width="stretch")
    cfg=run["config"].get("curriculum",{})
    if cfg: st.json(cfg,expanded=True)
    benchmark=read_json(run["path"] / "benchmark/curriculum_benchmark.json",{})
    if benchmark:
        section("Historical difficulty sweep", "Legacy success_rate used a clearance threshold. The table below preserves progress and clearance without relabeling them as course completion.")
        rows=[]
        for key,r in benchmark.items():
            if not isinstance(r,dict): continue
            rows.append({"Level":r.get("tier",key),"Seeds":len(r.get("episodes",[])),"Raw return":r.get("mean_reward"),"Course progress":r.get("mean_completion_rate"),"Obstacle clearance":r.get("mean_clearance_rate"),"Speed (m/s)":r.get("mean_speed")})
        st.dataframe(pd.DataFrame(rows),hide_index=True,width="stretch",column_config={"Course progress":st.column_config.NumberColumn(format="percent"),"Obstacle clearance":st.column_config.NumberColumn(format="percent")})
    lineage(run)


def course_figure(course, three_d=False):
    fig=go.Figure()
    for index,segment in enumerate(course.get("segments",[])):
        color={"flat":"#93AAA2","gap":"#EF9564","boxes":"#0B8D9E","stairs_up":"#5D84AF","stairs_down":"#6C71C4","low_wall":"#F07D45"}.get(segment.get("kind"),COLORS[0])
        for g in segment.get("geom_specs",[]):
            x,y,z=g["pos"]; hx,hy,hz=g["size"]
            if three_d:
                vertices=[(x+sx*hx,y+sy*hy,z+sz*hz) for sz in (-1,1) for sy in (-1,1) for sx in (-1,1)]
                faces=[(0,1,2),(1,3,2),(4,6,5),(5,6,7),(0,4,1),(1,4,5),(2,3,6),(3,7,6),(0,2,4),(2,6,4),(1,5,3),(3,5,7)]
                fig.add_trace(go.Mesh3d(x=[v[0] for v in vertices],y=[v[1] for v in vertices],z=[v[2] for v in vertices],i=[f[0] for f in faces],j=[f[1] for f in faces],k=[f[2] for f in faces],color=color,name=segment["kind"],flatshading=True,hovertemplate=f'{segment["kind"]}<br>x: %{{x:.2f}} m<extra></extra>',showscale=False))
            else:
                fig.add_shape(type="rect",x0=x-hx,x1=x+hx,y0=z-hz,y1=z+hz,fillcolor=color,line=dict(width=1,color=color))
        if not three_d:
            fig.add_annotation(x=(segment["x_start"]+segment["x_end"])/2,y=-.4,text=f'{index+1} · {segment["kind"].replace("_"," ")}',showarrow=False,textangle=-20,font=dict(size=10,color=color))
            if segment["kind"]=="gap": fig.add_vrect(x0=segment["x_start"],x1=segment["x_end"],fillcolor=color,opacity=.1,line_width=0)
    if three_d:
        fig.update_layout(scene=dict(xaxis_title="Forward · m",yaxis_title="Width · m",zaxis_title="Height · m",aspectmode="data",camera=dict(eye=dict(x=1.3,y=-1.5,z=.85))),showlegend=False)
    else:
        fig.add_trace(go.Scatter(x=[2.5],y=[1.15],mode="markers+text",text=["Spawn"],textposition="top center",marker=dict(size=13,color="#172C3B"),showlegend=False))
        fig.update_xaxes(range=[-.3,max(course.get("length",5)+.5,6)],title="Forward distance · m")
        fig.update_yaxes(range=[-.8,2.1],title="Height · m")
    return chart_style(fig,height=380 if three_d else 300)


def course_studio():
    section("Course studio", "Build the geometry the policy will face. Start safely, vary one challenge, and keep the landing visible.")
    if "course_draft" not in st.session_state:
        st.session_state.course_draft=read_json(ROOT/"custom_course.json") or new_course()
    course=st.session_state.course_draft
    view=st.segmented_control("Preview",["Side elevation","3D terrain"],default="Side elevation")
    st.plotly_chart(course_figure(course,view=="3D terrain"),width="stretch")
    a,b,c=st.columns(3)
    with a: st.metric("Course length",fmt(course.get("length"),1," m"))
    with b: st.metric("Segments",str(len(course.get("segments",[]))))
    with c: st.metric("Physics geometry",f'{sum(len(s.get("geom_specs",[])) for s in course.get("segments",[]))} / 200')
    with st.form("segment_builder"):
        section("Add a challenge")
        c1,c2,c3=st.columns(3)
        with c1: kind=st.selectbox("Terrain type",KINDS,format_func=lambda x:x.replace("_"," ").capitalize())
        with c2: length=st.number_input("Length / gap width (m)",min_value=.15,max_value=15.,value=3.,step=.05)
        with c3: height=st.number_input("Obstacle height / stair rise (m)",min_value=.02,max_value=1.,value=.15,step=.01)
        c1,c2=st.columns(2)
        with c1: width=st.number_input("Platform width (m)",min_value=1.,max_value=8.,value=3.4,step=.1)
        with c2: count=st.number_input("Stair count",min_value=2,max_value=12,value=4)
        add=st.form_submit_button("Add segment",type="primary")
        if add:
            st.session_state.course_draft=append_segment(course,kind,float(length),float(height),float(width),int(count))
            st.rerun()
    rows=[{"Segment":i+1,"Type":s["kind"].replace("_"," "),"Start (m)":s["x_start"],"End (m)":s["x_end"],"Geometries":len(s["geom_specs"])} for i,s in enumerate(course.get("segments",[]))]
    st.dataframe(pd.DataFrame(rows),hide_index=True,width="stretch")
    errors=validate_course(course)
    for error in errors: st.warning(error)
    a,b,c=st.columns(3)
    with a:
        if st.button("Undo last segment",disabled=len(course.get("segments",[]))<=1,width="stretch"):
            course["segments"].pop(); course["length"]=course["segments"][-1]["x_end"]
            st.session_state.course_draft=course; st.rerun()
    with b:
        if st.button("Reset draft to flat",width="stretch"):
            st.session_state.course_draft=new_course(); st.rerun()
    with c:
        if st.button("Save custom course",type="primary",disabled=bool(errors),width="stretch"):
            atomic_write(ROOT/"custom_course.json",json.dumps(course,indent=2)); st.success("Custom course saved. Select initial_level: -1.0 in a new parkour run to use it.")
    st.caption("Draft edits stay in this session until saved. Saving updates custom_course.json; active custom-course trainers can read it on reset, so finish those runs before changing the saved course.")
    st.download_button("Export course JSON",json.dumps(course,indent=2),file_name="custom_course.json",mime="application/json")


def launch_and_config(all_runs):
    section("Launch & configure", "Every launch writes a new run. Policies, normalization state, and previous evidence stay together.")
    live=trainers(ROOT)
    record=read_json(ROOT/"dashboard_process.json",{})
    if live:
        for process in live:
            st.success(f'Trainer active · {Path(process["run"]).name} · PID {process["pid"]}')
            if verify_process(record) and record.get("pid")==process["pid"]:
                if st.button("Save checkpoint & stop gracefully",key=f"stop_{process['pid']}",type="primary"):
                    try: stop_training(record); st.info("Stop requested. The trainer is saving its checkpoint; this can take until the current evaluation finishes.")
                    except Exception as exc: st.error(str(exc))
            else:
                # External trainer can be controlled only when lifecycle identity
                # matches an actual process; a bare PID file is never trusted.
                status=read_json(Path(process["run"])/"status.json",{})
                identity=status.get("process",status)
                if "command" not in identity and "cmdline" in identity: identity=dict(identity,command=identity["cmdline"])
                if verify_process(identity):
                    if st.button("Save checkpoint & stop gracefully",key=f"external_stop_{process['pid']}"):
                        try: stop_training(identity); st.info("Graceful stop requested.")
                        except Exception as exc: st.error(str(exc))
                else: st.caption("This trainer was launched elsewhere. No complete ownership record is available, so stopping is disabled.")
    configs=sorted((ROOT/"configs").glob("*.yaml"))
    if not configs: st.warning("No YAML configurations found."); return
    tab1,tab2=st.tabs(["New training run","Config workspace"])
    with tab1:
        selected=st.selectbox("Configuration",configs,format_func=lambda p:p.name,key="launch_config")
        cfg,errors,warnings=validate_config(selected.read_text())
        a,b=st.columns(2)
        with a:
            name=st.text_input("New run name",value=f"experiment-{datetime.now():%m%d-%H%M}")
            steps=st.number_input("New control decisions",min_value=1000,value=int(cfg.get("total_timesteps",2_000_000)),step=100000,help="Additional budget for resume; the saved lifetime counter is preserved.")
        with b:
            mode=st.selectbox("Initialization",["fresh","resume","transfer"],format_func=lambda x:{"fresh":"Fresh policy","resume":"Resume policy + optimizer","transfer":"Transfer compatible weights"}[x])
            checkpoints=[cp for r in all_runs if not r["utility"] for cp in checkpoint_choices(r)[:2]]
            checkpoint=""
            if mode!="fresh":
                if checkpoints: checkpoint=str(st.selectbox("Source checkpoint",checkpoints,format_func=lambda p:str(p.relative_to(ROOT))))
                else: st.warning("No paired checkpoints available.")
            apply=st.checkbox("Apply this config's PPO settings on resume",value=False,disabled=mode!="resume")
        for error in errors: st.error(error)
        for warning in warnings: st.warning(warning)
        st.caption(f'{STAGE_NAMES.get(cfg.get("stage"),"Planned stage")} · {cfg.get("n_envs","—")} workers · {compact(cfg.get("n_steps",0)*cfg.get("n_envs",0))} decisions/update · {cfg.get("eval_every_updates","—")} updates between videos')
        if mode=="transfer": st.info("Transfer changes the observation contract. The trainer must validate compatible network shapes and preserve the shared observation normalizer.")
        if st.button("Start training",type="primary",disabled=bool(errors or live or (mode!="fresh" and not checkpoint)),width="stretch"):
            try:
                launched=launch_training(selected,name,int(steps),mode,checkpoint,apply)
                st.success(f'Started {Path(launched["run"]).name}. Open Training room after its first evaluation.'); st.rerun()
            except Exception as exc: st.error(str(exc))
    with tab2:
        chosen=st.selectbox("Config file",configs,format_func=lambda p:p.name,key="editor_config")
        with st.form(f"edit_{chosen.name}"):
            content=st.text_area("YAML configuration",chosen.read_text(),height=480)
            save=st.form_submit_button("Validate & save config",type="primary")
        if save:
            _,errors,warnings=validate_config(content)
            if errors:
                for error in errors: st.error(error)
                st.info("The file has not changed.")
            else:
                atomic_write(chosen,content); st.success("Validated and saved. Existing runs retain their copied configuration.")
                for warning in warnings: st.warning(warning)
        st.caption("Planned stages remain visible for reference but cannot be launched until their trainer and evaluation contract are implemented.")
    with st.expander("Trainer log",expanded=bool(live)):
        path=ROOT/"dashboard_train.log"
        if path.exists():
            with path.open("rb") as log:
                log.seek(max(0,path.stat().st_size-12000))
                st.code(log.read().decode(errors="replace"),language=None)
        else: st.caption("No dashboard-launched training log yet.")


with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-mark">PL</div><div><div class="brand-name">Parkour Lab</div><div class="brand-sub">Learning in motion</div></div></div>',unsafe_allow_html=True)
    page=st.radio("Workspace",["Training room","Run comparison","Evaluation lab","Curriculum","Course studio","Launch & configs"],label_visibility="collapsed")
    st.divider()
    show_utility=st.toggle("Show smoke & benchmark runs",value=False)
    initial_runs=discover_runs()
    visible=[r for r in initial_runs if show_utility or not r["utility"]]
    names=[r["name"] for r in visible]
    active_names=[r["name"] for r in visible if r["state"] in ("training","evaluating","starting","stopping")]
    default=active_names[0] if active_names else "stage4-deep" if "stage4-deep" in names else names[0] if names else None
    selected_name=st.selectbox("Run in focus",names,index=names.index(default) if default else None)
    if st.button("Refresh evidence",width="stretch"): st.rerun()
    st.caption("Training views refresh every 20 seconds. No simulation runs inside this dashboard.")
    st.divider()
    st.markdown('<div class="eyebrow">LOCAL RUNTIME</div>',unsafe_allow_html=True)
    cpu=psutil.cpu_percent(interval=None)
    memory=psutil.virtual_memory()
    st.caption(f'{psutil.cpu_count(logical=True)} CPU cores · {memory.total/2**30:.0f} GB memory')
    st.progress(memory.percent/100,text=f'Memory {memory.percent:.0f}% used')
    st.link_button("Open TensorBoard", "http://127.0.0.1:6007",width="stretch")
    st.caption("MuJoCo physics · PPO control · local evidence")


@st.fragment(run_every="20s")
def render_workspace():
    runs=discover_runs()
    run=next((r for r in runs if r["name"]==selected_name),None)
    if page=="Course studio": course_studio()
    elif page=="Launch & configs": launch_and_config(runs)
    elif run is None: st.info("Create a training run to start collecting evidence.")
    elif page=="Training room": overview(run,runs)
    elif page=="Run comparison": compare([r for r in runs if show_utility or not r["utility"]],run)
    elif page=="Evaluation lab": evidence_lab(run)
    elif page=="Curriculum": curriculum(run)
    st.markdown('<div class="footer">PARKOUR LAB / Raw evaluation evidence, reproducible checkpoints, and physics you can inspect.</div>',unsafe_allow_html=True)

render_workspace()
