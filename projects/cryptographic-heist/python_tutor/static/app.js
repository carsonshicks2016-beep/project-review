const STORAGE_KEY = "python-gym-progress-v1";

const state = {
  curriculum: null,
  level: "Foundations",
  topicId: null,
  exercise: null,
  selectedChoice: "",
  progress: loadProgress(),
};

const els = {
  courseCount: document.getElementById("course-count"),
  levelTabs: document.getElementById("level-tabs"),
  topicList: document.getElementById("topic-list"),
  exerciseLevel: document.getElementById("exercise-level"),
  exerciseTitle: document.getElementById("exercise-title"),
  exerciseConcept: document.getElementById("exercise-concept"),
  exercisePrompt: document.getElementById("exercise-prompt"),
  exerciseCode: document.getElementById("exercise-code"),
  answerZone: document.getElementById("answer-zone"),
  checkBtn: document.getElementById("check-btn"),
  newBtn: document.getElementById("new-btn"),
  shuffleBtn: document.getElementById("shuffle-btn"),
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
  if (!res.ok) {
    throw new Error(data.error || data.message || `${res.status} ${res.statusText}`);
  }
  return data;
}

async function init() {
  bindActions();
  state.curriculum = await api("/api/curriculum");
  const firstLevel = state.curriculum.levels[0];
  state.level = firstLevel.name;
  state.topicId = firstLevel.topics[0].id;
  renderShell();
  await loadExercise();
}

function bindActions() {
  els.checkBtn.addEventListener("click", checkCurrent);
  els.newBtn.addEventListener("click", () => loadExercise());
  els.shuffleBtn.addEventListener("click", () => loadExercise({ shuffleTopic: true }));
  els.resetBtn.addEventListener("click", resetProgress);
}

function renderShell() {
  els.courseCount.textContent = `${state.curriculum.topic_count} topics`;
  renderLevels();
  renderTopics();
  renderScores();
  renderMastery();
}

function renderLevels() {
  els.levelTabs.innerHTML = "";
  state.curriculum.levels.forEach((level) => {
    const button = document.createElement("button");
    button.className = `level-tab ${level.name === state.level ? "active" : ""}`;
    button.textContent = level.name;
    button.addEventListener("click", async () => {
      state.level = level.name;
      state.topicId = level.topics[0].id;
      renderShell();
      await loadExercise();
    });
    els.levelTabs.appendChild(button);
  });
}

function renderTopics() {
  const level = currentLevel();
  els.topicList.innerHTML = "";
  level.topics.forEach((topic, index) => {
    const stats = topicStats(topic.id);
    const button = document.createElement("button");
    button.className = `topic-btn ${topic.id === state.topicId ? "active" : ""}`;
    button.innerHTML = `
      <span class="topic-index">${index + 1}</span>
      <span>
        <strong class="topic-title">${escapeHtml(topic.label)}</strong>
        <span class="topic-summary">${escapeHtml(topic.summary)}</span>
      </span>
      <span class="topic-stat">${stats.correct}/${stats.attempts}</span>
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
    const body = options.shuffleTopic ? { level: state.level } : { topic_id: state.topicId };
    const exercise = await api("/api/exercise", {
      method: "POST",
      body: JSON.stringify(body),
    });
    state.exercise = exercise;
    state.topicId = exercise.topic_id;
    state.level = exercise.level;
    state.selectedChoice = "";
    renderShell();
    renderExercise();
    setStatus(`Seed ${exercise.seed}`);
  } catch (error) {
    clearFeedback(error.message, false);
    setStatus("Error");
  }
}

function renderExercise() {
  const exercise = state.exercise;
  if (!exercise) return;
  els.exerciseLevel.textContent = `${exercise.level} / ${exercise.topic_label}`;
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
  const answer = collectAnswer();
  setStatus("Checking");
  try {
    const result = await api("/api/check", {
      method: "POST",
      body: JSON.stringify({ id: state.exercise.id, answer }),
    });
    recordAttempt(state.exercise.topic_id, result.correct);
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

function collectAnswer() {
  if (!state.exercise) return "";
  if (state.exercise.mode === "code") {
    return document.getElementById("answer-editor").value;
  }
  if (state.exercise.mode === "choice") {
    return state.selectedChoice;
  }
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

function recordAttempt(topicId, correct) {
  state.progress.total += 1;
  if (correct) {
    state.progress.correct += 1;
    state.progress.streak += 1;
  } else {
    state.progress.streak = 0;
  }
  const stats = topicStats(topicId);
  stats.attempts += 1;
  if (correct) stats.correct += 1;
  state.progress.topics[topicId] = stats;
  saveProgress();
}

function renderScores() {
  els.totalCount.textContent = state.progress.total;
  els.correctCount.textContent = state.progress.correct;
  els.streakCount.textContent = state.progress.streak;
}

function renderMastery() {
  const topics = state.curriculum.levels.flatMap((level) => level.topics.map((topic) => ({ ...topic, level: level.name })));
  els.masteryList.innerHTML = "";
  topics.forEach((topic) => {
    const stats = topicStats(topic.id);
    const pct = stats.attempts ? Math.round((stats.correct / stats.attempts) * 100) : 0;
    const row = document.createElement("div");
    row.className = "mastery-item";
    row.innerHTML = `
      <div class="mastery-label">
        <strong>${escapeHtml(topic.label)}</strong>
        <small>${escapeHtml(topic.level)} / ${stats.correct}/${stats.attempts}</small>
        <div class="meter"><span style="width: ${pct}%"></span></div>
      </div>
      <span class="topic-stat">${pct}%</span>
    `;
    els.masteryList.appendChild(row);
  });
}

function currentLevel() {
  return state.curriculum.levels.find((level) => level.name === state.level) || state.curriculum.levels[0];
}

function topicStats(topicId) {
  return state.progress.topics[topicId] || { attempts: 0, correct: 0 };
}

function setStatus(message) {
  els.saveState.textContent = message;
}

function resetProgress() {
  if (!window.confirm("Reset local Python Gym progress?")) return;
  state.progress = { total: 0, correct: 0, streak: 0, topics: {} };
  saveProgress();
  renderShell();
  clearFeedback("Progress reset.");
}

function loadProgress() {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return { total: 0, correct: 0, streak: 0, topics: {} };
    const parsed = JSON.parse(raw);
    return {
      total: Number(parsed.total || 0),
      correct: Number(parsed.correct || 0),
      streak: Number(parsed.streak || 0),
      topics: parsed.topics || {},
    };
  } catch {
    return { total: 0, correct: 0, streak: 0, topics: {} };
  }
}

function saveProgress() {
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(state.progress));
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

