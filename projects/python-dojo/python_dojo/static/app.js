const state = {
  curriculum: null,
  progress: null,
  stage: null,
  topicId: null,
  exercise: null,
  selectedChoice: "",
  lastFeedback: "",
};

const els = {
  courseCount: document.getElementById("course-count"),
  stageTabs: document.getElementById("stage-tabs"),
  topicList: document.getElementById("topic-list"),
  exerciseStage: document.getElementById("exercise-stage"),
  exerciseTitle: document.getElementById("exercise-title"),
  lessonTitle: document.getElementById("lesson-title"),
  lessonBody: document.getElementById("lesson-body"),
  lessonExample: document.getElementById("lesson-example"),
  exerciseConcept: document.getElementById("exercise-concept"),
  exercisePrompt: document.getElementById("exercise-prompt"),
  exerciseCode: document.getElementById("exercise-code"),
  answerZone: document.getElementById("answer-zone"),
  checkBtn: document.getElementById("check-btn"),
  nextBtn: document.getElementById("next-btn"),
  variantBtn: document.getElementById("variant-btn"),
  hintBtn: document.getElementById("hint-btn"),
  resetBtn: document.getElementById("reset-btn"),
  saveState: document.getElementById("save-state"),
  totalCount: document.getElementById("total-count"),
  correctCount: document.getElementById("correct-count"),
  streakCount: document.getElementById("streak-count"),
  feedbackBox: document.getElementById("feedback-box"),
  feedbackMessage: document.getElementById("feedback-message"),
  testDetails: document.getElementById("test-details"),
  masteryList: document.getElementById("mastery-list"),
};

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || data.message || `${res.status} ${res.statusText}`);
  return data;
}

async function init() {
  bindActions();
  state.curriculum = await api("/api/curriculum");
  state.progress = await api("/api/progress");
  const firstStage = state.curriculum.stages[0];
  state.stage = firstStage.name;
  state.topicId = state.progress.next_topic_id || firstStage.topics[0].id;
  syncStageToTopic();
  renderShell();
  await loadExercise({ recommended: true });
}

function bindActions() {
  els.checkBtn.addEventListener("click", checkCurrent);
  els.nextBtn.addEventListener("click", () => loadExercise({ recommended: true }));
  els.variantBtn.addEventListener("click", () => loadExercise());
  els.hintBtn.addEventListener("click", getHint);
  els.resetBtn.addEventListener("click", resetProgress);
}

function renderShell() {
  els.courseCount.textContent = `${state.curriculum.topic_count} topics`;
  renderStages();
  renderTopics();
  renderScores();
  renderMastery();
}

function renderStages() {
  els.stageTabs.innerHTML = "";
  state.curriculum.stages.forEach((stage) => {
    const button = document.createElement("button");
    button.className = `stage-tab ${stage.name === state.stage ? "active" : ""}`;
    button.textContent = stage.name;
    button.addEventListener("click", async () => {
      state.stage = stage.name;
      const firstUnlocked = stage.topics.find((topic) => isUnlocked(topic.id)) || stage.topics[0];
      state.topicId = firstUnlocked.id;
      renderShell();
      await loadExercise();
    });
    els.stageTabs.appendChild(button);
  });
}

function renderTopics() {
  const stage = currentStage();
  els.topicList.innerHTML = "";
  stage.topics.forEach((topic, index) => {
    const stat = masteryFor(topic.id);
    const unlocked = isUnlocked(topic.id);
    const button = document.createElement("button");
    button.className = `topic-btn ${topic.id === state.topicId ? "active" : ""} ${unlocked ? "" : "locked"}`;
    button.disabled = !unlocked;
    button.innerHTML = `
      <span class="topic-index">${index + 1}</span>
      <span>
        <strong class="topic-title">${escapeHtml(topic.label)}</strong>
        <span class="topic-summary">${escapeHtml(topic.summary)}</span>
      </span>
      <span class="topic-stat">${Math.round((stat?.mastery || 0) * 100)}%</span>
    `;
    button.addEventListener("click", async () => {
      state.topicId = topic.id;
      renderTopics();
      await loadExercise();
    });
    els.topicList.appendChild(button);
  });
}

async function loadExercise(options = {}) {
  setStatus("Generating");
  clearFeedback("Ready.");
  try {
    const topicId = options.recommended ? state.progress.next_topic_id : state.topicId;
    const exercise = await api("/api/exercise", {
      method: "POST",
      body: JSON.stringify({ topic_id: topicId }),
    });
    state.exercise = exercise;
    state.topicId = exercise.topic_id;
    state.stage = exercise.stage;
    state.selectedChoice = "";
    syncStageToTopic();
    renderShell();
    renderLesson(exercise);
    renderExercise();
    setStatus(`Seed ${exercise.seed}`);
  } catch (error) {
    clearFeedback(error.message, false);
    setStatus("Error");
  }
}

function renderLesson(exercise) {
  els.lessonTitle.textContent = exercise.topic_label;
  els.lessonBody.textContent = exercise.concept;
  els.lessonExample.textContent = exercise.starter || exercise.code || "# solve the prompt below";
}

function renderExercise() {
  const exercise = state.exercise;
  if (!exercise) return;
  els.exerciseStage.textContent = `${exercise.stage} / Difficulty ${exercise.difficulty}`;
  els.exerciseTitle.textContent = exercise.title;
  els.exerciseConcept.textContent = exercise.concept;
  els.exercisePrompt.textContent = exercise.prompt;
  if (exercise.code) {
    els.exerciseCode.textContent = exercise.code;
    els.exerciseCode.classList.remove("hidden");
  } else {
    els.exerciseCode.textContent = "";
    els.exerciseCode.classList.add("hidden");
  }
  if (exercise.mode === "code") {
    els.answerZone.innerHTML = `<textarea id="answer-editor" class="answer-editor" spellcheck="false"></textarea>`;
    document.getElementById("answer-editor").value = exercise.starter || "";
    return;
  }
  if (exercise.mode === "choice") {
    els.answerZone.innerHTML = `<div id="choice-grid" class="choice-grid"></div>`;
    const grid = document.getElementById("choice-grid");
    exercise.choices.forEach((choice) => {
      const button = document.createElement("button");
      button.className = "choice-btn";
      button.textContent = choice;
      button.addEventListener("click", () => {
        state.selectedChoice = choice;
        document.querySelectorAll(".choice-btn").forEach((item) => item.classList.remove("selected"));
        button.classList.add("selected");
      });
      grid.appendChild(button);
    });
    return;
  }
  els.answerZone.innerHTML = `<input id="short-answer" class="short-input" autocomplete="off" />`;
  document.getElementById("short-answer").focus();
}

async function checkCurrent() {
  if (!state.exercise) return;
  setStatus("Checking");
  try {
    const result = await api("/api/check", {
      method: "POST",
      body: JSON.stringify({ id: state.exercise.id, answer: collectAnswer() }),
    });
    state.progress = result.progress;
    state.lastFeedback = result.message;
    renderScores();
    renderTopics();
    renderMastery();
    renderFeedback(result);
    setStatus(result.correct ? "Correct" : "Try again");
  } catch (error) {
    clearFeedback(error.message, false);
    setStatus("Error");
  }
}

async function getHint() {
  if (!state.exercise) return;
  setStatus("Hint");
  try {
    const result = await api("/api/hint", {
      method: "POST",
      body: JSON.stringify({ id: state.exercise.id, answer: collectAnswer(), message: state.lastFeedback }),
    });
    clearFeedback(result.hint, false);
    setStatus("Hint ready");
  } catch (error) {
    clearFeedback(error.message, false);
    setStatus("Error");
  }
}

function collectAnswer() {
  if (!state.exercise) return "";
  if (state.exercise.mode === "code") return document.getElementById("answer-editor").value;
  if (state.exercise.mode === "choice") return state.selectedChoice;
  return document.getElementById("short-answer").value;
}

function renderFeedback(result) {
  els.feedbackBox.classList.toggle("correct", result.correct);
  els.feedbackBox.classList.toggle("incorrect", !result.correct);
  els.feedbackMessage.textContent = result.message;
  els.testDetails.innerHTML = "";
  (result.details || []).forEach((row) => {
    const detail = document.createElement("div");
    detail.className = `test-row ${row.ok ? "pass" : "fail"}`;
    detail.innerHTML = `
      <code>${escapeHtml(row.expression || "")}</code>
      <span>actual: ${escapeHtml(row.actual ?? "")}</span>
      <span>expected: ${escapeHtml(row.expected ?? row.raises ?? "")}</span>
    `;
    els.testDetails.appendChild(detail);
  });
}

function clearFeedback(message, neutral = true) {
  els.feedbackBox.classList.toggle("correct", false);
  els.feedbackBox.classList.toggle("incorrect", !neutral);
  els.feedbackMessage.textContent = message;
  els.testDetails.innerHTML = "";
}

function renderScores() {
  els.totalCount.textContent = state.progress?.total_attempts || 0;
  els.correctCount.textContent = state.progress?.total_correct || 0;
  els.streakCount.textContent = state.progress?.streak || 0;
}

function renderMastery() {
  els.masteryList.innerHTML = "";
  (state.progress?.mastery || []).forEach((stat) => {
    const topic = topicById(stat.topic_id);
    if (!topic) return;
    const row = document.createElement("div");
    row.className = "mastery-item";
    row.innerHTML = `
      <div class="mastery-label">
        <strong>${escapeHtml(topic.label)}</strong>
        <small>${escapeHtml(topic.stage)} / ${stat.correct}/${stat.attempts}${stat.needs_review ? " / review" : ""}</small>
        <div class="meter"><span style="width: ${Math.round(stat.mastery * 100)}%"></span></div>
      </div>
      <span class="topic-stat">${Math.round(stat.mastery * 100)}%</span>
    `;
    els.masteryList.appendChild(row);
  });
}

async function resetProgress() {
  if (!window.confirm("Reset local Python Dojo progress?")) return;
  state.progress = await api("/api/progress/reset", { method: "POST", body: "{}" });
  renderShell();
  clearFeedback("Progress reset.");
  await loadExercise({ recommended: true });
}

function currentStage() {
  return state.curriculum.stages.find((stage) => stage.name === state.stage) || state.curriculum.stages[0];
}

function syncStageToTopic() {
  const topic = topicById(state.topicId);
  if (topic) state.stage = topic.stage;
}

function topicById(topicId) {
  return state.curriculum.stages.flatMap((stage) => stage.topics.map((topic) => ({ ...topic, stage: stage.name }))).find((topic) => topic.id === topicId);
}

function masteryFor(topicId) {
  return (state.progress?.mastery || []).find((row) => row.topic_id === topicId);
}

function isUnlocked(topicId) {
  return (state.progress?.unlocked_topics || []).includes(topicId);
}

function setStatus(message) {
  els.saveState.textContent = message;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

init().catch((error) => {
  clearFeedback(error.message, false);
  setStatus("Error");
});
