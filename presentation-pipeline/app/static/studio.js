"use strict";

const sourcePlayer = document.getElementById("source-player");
function seekSource(time) {
  if (!sourcePlayer || !Number.isFinite(time)) return;
  sourcePlayer.currentTime = Math.max(0, time);
  sourcePlayer.focus();
}
document.querySelectorAll("[data-seek]").forEach((button) => {
  button.addEventListener("click", () => seekSource(Number(button.dataset.seek)));
});
document.addEventListener("invalid", (event) => {
  const group = event.target.closest("details");
  if (group) group.open = true;
}, true);

const overview = document.getElementById("cut-overview");
if (overview) {
  const cuts = JSON.parse(overview.dataset.cuts);
  const draw = () => {
    const width = overview.clientWidth;
    const scale = window.devicePixelRatio || 1;
    overview.width = width * scale;
    overview.height = 48 * scale;
    const context = overview.getContext("2d");
    context.scale(scale, scale);
    context.fillStyle = "#66ccb6";
    context.fillRect(0, 0, width, 48);
    if (cuts.original_duration > 0) {
      context.fillStyle = "#cb5a70";
      for (const [start, end] of cuts.removed) {
        context.fillRect(start / cuts.original_duration * width, 0,
          (end - start) / cuts.original_duration * width, 48);
      }
    }
  };
  draw();
  window.addEventListener("resize", draw);
  overview.addEventListener("click", (event) => {
    const rect = overview.getBoundingClientRect();
    seekSource((event.clientX - rect.left) / rect.width * cuts.original_duration);
  });
}

const job = document.getElementById("job-status");
if (job) {
  const initial = JSON.parse(job.dataset.snapshot);
  const notice = document.getElementById("update-notice");
  const error = document.getElementById("poll-error");
  let dirty = false;
  document.querySelectorAll("[data-settings-form]").forEach((form) => {
    form.addEventListener("input", () => { dirty = true; });
    form.addEventListener("change", () => { dirty = true; });
  });
  const active = (status) => ["queued", "running"].includes(status.status) ||
    ["queued", "running"].includes(status.preview.status);
  const signature = (status) => JSON.stringify([
    status.stage, status.status, status.error, status.preview.status,
    status.preview.error, status.files,
  ]);
  async function poll() {
    try {
      const response = await fetch(job.dataset.statusUrl, {cache: "no-store"});
      if (!response.ok) throw new Error(`Progress request failed (${response.status}).`);
      const status = await response.json();
      error.hidden = true;
      if (signature(status) !== signature(initial)) {
        const playing = [...document.querySelectorAll("video")].some(
          (video) => !video.paused && !video.ended
        );
        if (!playing && !dirty) {
          window.location.reload();
          return;
        }
        notice.hidden = false;
      }
      if (active(status) || !notice.hidden) window.setTimeout(poll, 5000);
    } catch (failure) {
      error.textContent = `${failure.message} Retrying in 5 seconds.`;
      error.hidden = false;
      window.setTimeout(poll, 5000);
    }
  }
  if (active(initial)) window.setTimeout(poll, 5000);
}
