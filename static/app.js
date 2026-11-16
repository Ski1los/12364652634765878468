const form = document.querySelector('#job-form');
const errorBox = document.querySelector('#form-error');
const submitButton = document.querySelector('#submit-button');
const jobsNode = document.querySelector('#jobs');
const emptyNode = document.querySelector('#empty-state');
const runtimeStatus = document.querySelector('#runtime-status');
let jobs = [];
let pollTimer = null;

function esc(value = '') {
  return String(value).replace(/[&<>'"]/g, char => ({ '&':'&amp;', '<':'&lt;', '>':'&gt;', "'":'&#39;', '"':'&quot;' }[char]));
}
function stageLabel(status) {
  return ({ queued:'в очереди', downloading:'скачивание', transcribing:'локальная расшифровка', selecting:'выбор моментов', rendering:'рендер клипов', completed:'готово', failed:'ошибка' })[status] || status;
}
function dateLabel(iso) {
  try { return new Intl.DateTimeFormat('ru-RU',{dateStyle:'medium',timeStyle:'short'}).format(new Date(iso)); } catch { return iso || ''; }
}
function fileUrl(job, filename) {
  return `/api/jobs/${encodeURIComponent(job.id)}/files/${encodeURIComponent(filename)}`;
}
function logsMarkup(job) {
  if (!job.logs?.length) return '';
  const lines = job.logs.map(item => `[${new Date(item.at).toLocaleTimeString('ru-RU')}] ${item.message}`).join('\n');
  return `<details class="logs"><summary>Журнал обработки (${job.logs.length})</summary><pre>${esc(lines)}</pre></details>`;
}
function clipsMarkup(job) {
  if (!job.clips?.length) return '';
  return `<div class="clip-grid">${job.clips.map((clip, index) => `
    <article class="clip-card">
      <video controls preload="metadata" src="${fileUrl(job, clip.video)}"></video>
      <div class="clip-content">
        <h4>${esc(clip.title || `Момент ${index + 1}`)}</h4>
        <p>${esc(clip.reason || 'Выбранный фрагмент')}</p>
        <div class="meta">${esc(String(clip.duration || '—'))} сек · ${Math.round(clip.start || 0)}–${Math.round(clip.end || 0)} сек</div>
        <div class="downloads"><a href="${fileUrl(job, clip.video)}" download>MP4 ↓</a><a class="alt" href="${fileUrl(job, clip.subtitles)}" download>SRT ↓</a></div>
      </div>
    </article>`).join('')}</div>`;
}
function jobMarkup(job) {
  const percent = Math.min(100, Math.max(0, Number(job.progress) || 0));
  const deleting = ['queued','downloading','transcribing','selecting','rendering'].includes(job.status);
  return `<article class="job" id="job-${esc(job.id)}">
    <header class="job-header">
      <div><div class="job-title"><i class="status-dot ${esc(job.status)}"></i>${esc(stageLabel(job.status))}</div><div class="job-subtitle">${esc(dateLabel(job.created_at))} · ${esc(job.id.slice(0, 8))}</div></div>
      ${deleting ? '' : `<button class="delete-job" data-delete="${esc(job.id)}" title="Удалить файлы этой задачи">удалить</button>`}
    </header>
    ${job.status === 'failed' ? `<div class="job-error">${esc(job.error || 'Задача не была завершена.')}</div>` : ''}
    ${job.status !== 'completed' ? `<div class="progress-wrap"><div class="progress-label"><span>${esc(stageLabel(job.status))}</span><span>${percent}%</span></div><div class="progress-bar"><span style="width:${percent}%"></span></div></div>` : ''}
    ${clipsMarkup(job)}${logsMarkup(job)}
  </article>`;
}
function render() {
  emptyNode.hidden = jobs.length > 0;
  jobsNode.innerHTML = jobs.map(jobMarkup).join('');
  document.querySelectorAll('[data-delete]').forEach(button => button.addEventListener('click', async () => {
    if (!confirm('Удалить готовые файлы этой задачи с сервера?')) return;
    const response = await fetch(`/api/jobs/${encodeURIComponent(button.dataset.delete)}`, { method:'DELETE' });
    if (!response.ok) alert('Не удалось удалить задачу.');
    await refresh();
  }));
}
async function refresh() {
  try {
    const response = await fetch('/api/jobs', { cache:'no-store' });
    if (!response.ok) throw new Error('Сервер не ответил');
    jobs = await response.json();
    render();
    const active = jobs.some(job => ['queued','downloading','transcribing','selecting','rendering'].includes(job.status));
    if (active && !pollTimer) pollTimer = setInterval(refresh, 3500);
    if (!active && pollTimer) { clearInterval(pollTimer); pollTimer = null; }
  } catch (error) {
    runtimeStatus.textContent = 'Нет соединения с локальным сервером';
  }
}
function showError(message) { errorBox.textContent = message; errorBox.hidden = false; }
function hideError() { errorBox.hidden = true; errorBox.textContent = ''; }
form.addEventListener('submit', async event => {
  event.preventDefault(); hideError();
  const minSeconds = Number(document.querySelector('#min-seconds').value);
  const maxSeconds = Number(document.querySelector('#max-seconds').value);
  if (maxSeconds <= minSeconds) return showError('Максимальная длительность должна быть больше минимальной.');
  const payload = {
    source_url: document.querySelector('#source-url').value.trim(),
    clip_count: Number(document.querySelector('#clip-count').value),
    min_seconds: minSeconds,
    max_seconds: maxSeconds,
    render_height: Number(document.querySelector('#render-height').value),
    rights_confirmed: document.querySelector('#rights').checked,
  };
  submitButton.disabled = true; submitButton.querySelector('span').textContent = 'Добавляю в очередь…';
  try {
    const response = await fetch('/api/jobs', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Не удалось создать задачу.');
    form.reset(); document.querySelector('#clip-count').value = '3'; document.querySelector('#render-height').value = '720';
    jobs.unshift(data); render();
    document.querySelector('.jobs-section').scrollIntoView({ behavior:'smooth', block:'start' });
    await refresh();
  } catch (error) { showError(error.message || 'Не удалось создать задачу.'); }
  finally { submitButton.disabled = false; submitButton.querySelector('span').textContent = 'Найти моменты'; }
});
document.querySelector('#refresh').addEventListener('click', refresh);
(async () => {
  try {
    const response = await fetch('/api/config');
    const data = await response.json();
    runtimeStatus.textContent = `Whisper ${data.whisper_model} · ${data.local_llm_enabled ? 'Ollama подключена' : 'эвристика + Whisper'}`;
  } catch { runtimeStatus.textContent = 'Локальный режим'; }
  refresh();
})();
