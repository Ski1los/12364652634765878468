const form = document.querySelector('#job-form');
const storyForm = document.querySelector('#story-form');
const planForm = document.querySelector('#plan-form');
const errorBox = document.querySelector('#form-error');
const storyError = document.querySelector('#story-error');
const planError = document.querySelector('#plan-error');
const submitButton = document.querySelector('#submit-button');
const storySubmit = document.querySelector('#story-submit');
const planSubmit = document.querySelector('#plan-submit');
const jobsNode = document.querySelector('#jobs');
const emptyNode = document.querySelector('#empty-state');
const runtimeStatus = document.querySelector('#runtime-status');
const hardwareProfile = document.querySelector('#hardware-profile');
const cloudClipConsent = document.querySelector('#cloud-clip-consent');
const cloudStoryConsent = document.querySelector('#cloud-story-consent');
const cloudPlanConsent = document.querySelector('#cloud-plan-consent');
const activeStatuses = new Set(['queued','downloading','transcribing','selecting','rendering','drafting','narrating','composing','planning']);
let jobs = [];
let pollTimer = null;
let config = null;

function esc(value = '') {
  return String(value).replace(/[&<>'"]/g, char => ({ '&':'&amp;', '<':'&lt;', '>':'&gt;', "'":'&#39;', '"':'&quot;' }[char]));
}
function stageLabel(status) {
  return ({ queued:'в очереди', downloading:'скачивание', transcribing:'локальная расшифровка', selecting:'выбор моментов', rendering:'рендер клипов', drafting:'сценарий', narrating:'локальная озвучка', composing:'сборка Story Reel', planning:'контент-план', completed:'готово', failed:'ошибка' })[status] || status;
}
function dateLabel(iso) {
  try { return new Intl.DateTimeFormat('ru-RU',{dateStyle:'medium',timeStyle:'short'}).format(new Date(iso)); } catch { return iso || ''; }
}
function fileUrl(job, filename) { return `/api/jobs/${encodeURIComponent(job.id)}/files/${encodeURIComponent(filename)}`; }
function showError(node, message) { node.textContent = message; node.hidden = false; }
function hideError(node) { node.hidden = true; node.textContent = ''; }
function logsMarkup(job) {
  if (!job.logs?.length) return '';
  const lines = job.logs.map(item => `[${new Date(item.at).toLocaleTimeString('ru-RU')}] ${item.message}`).join('\n');
  return `<details class="logs"><summary>Журнал обработки (${job.logs.length})</summary><pre>${esc(lines)}</pre></details>`;
}
function clipsMarkup(job) {
  if (!job.clips?.length) return '';
  return `<div class="clip-grid">${job.clips.map((clip, index) => `
    <article class="clip-card">
      <video controls playsinline preload="metadata" src="${fileUrl(job, clip.video)}"></video>
      <div class="clip-content">
        <h4>${esc(clip.title || `Момент ${index + 1}`)}</h4>
        <p>${esc(clip.reason || 'Выбранный фрагмент')}</p>
        <div class="meta">${esc(String(clip.duration || '—'))} сек · ${Math.round(clip.start || 0)}–${Math.round(clip.end || 0)} сек</div>
        <div class="downloads"><a href="${fileUrl(job, clip.video)}" download>MP4 ↓</a><a class="alt" href="${fileUrl(job, clip.subtitles)}" download>SRT ↓</a></div>
      </div>
    </article>`).join('')}</div>`;
}
function copyButton(text, label) {
  return `<button class="copy-button" type="button" data-copy="${esc(text)}">${esc(label)} ⧉</button>`;
}
function storyMarkup(job) {
  const story = job.story;
  if (!story) return '';
  const tags = (story.hashtags || []).map(tag => `<span>${esc(tag)}</span>`).join('');
  const titles = (story.titles || [story.title]).map((title, index) => `<div class="title-option"><b>${index + 1}</b><span>${esc(title)}</span>${copyButton(title, 'копировать')}</div>`).join('');
  const captions = story.platform_captions || {};
  const platformCards = [['youtube_shorts','YouTube Shorts'],['tiktok','TikTok'],['instagram_reels','Instagram Reels']].map(([key, label]) => captions[key] ? `<div class="platform-copy"><strong>${label}</strong><p>${esc(captions[key])}</p>${copyButton(captions[key], 'скопировать подпись')}</div>` : '').join('');
  return `<section class="story-result">
    <div class="story-video-wrap"><video controls playsinline preload="metadata" poster="${story.thumbnail ? fileUrl(job, story.thumbnail) : ''}" src="${fileUrl(job, story.video)}"></video></div>
    <div class="story-result-copy">
      <div class="story-result-eyebrow">AI-ASSISTED STORY REEL · ${esc(String(story.duration || '—'))} СЕК</div>
      <h3>${esc(story.title || 'Новая история')}</h3>
      <p class="story-hook">${esc(story.hook || '')}</p>
      <p class="story-description">${esc(story.description || '')}</p>
      <div class="tag-row">${tags}</div>
      <div class="story-disclosure">${esc(story.disclosure || 'Создано с помощью ИИ.')}</div>
      <div class="downloads story-downloads">
        <a href="${fileUrl(job, story.video)}" download>MP4 ↓</a>
        <a class="alt" href="${fileUrl(job, story.subtitles)}" download>SRT ↓</a>
        <a class="alt" href="${fileUrl(job, story.audio)}" download>WAV ↓</a>
        ${story.thumbnail ? `<a class="alt" href="${fileUrl(job, story.thumbnail)}" download>ОБЛОЖКА ↓</a>` : ''}
        <a class="alt" href="${fileUrl(job, story.content_pack)}" download>ТЕКСТ ↓</a>
        <a class="kit-download" href="/api/jobs/${encodeURIComponent(job.id)}/download-all" download>CREATOR KIT ZIP ↓</a>
      </div>
      <small>Голос: ${esc(story.voice_engine || 'локальный')}</small>
    </div>
    <div class="launch-kit"><div class="launch-kit-label">LAUNCH KIT · ПРОВЕРЬТЕ ПЕРЕД ПУБЛИКАЦИЕЙ</div><div class="title-options">${titles}</div><div class="platform-copies">${platformCards}</div></div>
  </section>`;
}
function planMarkup(job) {
  const plan = job.plan;
  if (!plan) return '';
  const cards = (plan.schedule || []).map(item => `<article class="plan-day"><header><span>ДЕНЬ ${esc(item.day)}</span><b>${esc(item.format)}</b></header><h4>${esc(item.title)}</h4><p>${esc(item.idea)}</p><div class="plan-hook">${esc(item.hook)}</div><small>CTA: ${esc(item.cta)}</small></article>`).join('');
  const pillars = (plan.content_pillars || []).map(item => `<span>${esc(item)}</span>`).join('');
  return `<section class="plan-result"><div class="plan-result-top"><div><div class="story-result-eyebrow">CREATOR PLAN · ${esc(plan.plan_days || '')} ДНЕЙ</div><h3>${esc(plan.positioning || 'Контент-план')}</h3><div class="tag-row">${pillars}</div></div><div class="downloads"><a class="kit-download" href="${fileUrl(job, plan.file)}" download>ПЛАН MD ↓</a><a class="alt" href="/api/jobs/${encodeURIComponent(job.id)}/download-all" download>ZIP ↓</a></div></div><div class="plan-calendar">${cards}</div><div class="plan-checks"><div><strong>Метрики без накруток</strong>${(plan.metrics_focus || []).map(item => `<p>• ${esc(item)}</p>`).join('')}</div><div><strong>Перед публикацией</strong>${(plan.publish_checklist || []).map(item => `<p>• ${esc(item)}</p>`).join('')}</div></div></section>`;
}
function jobMarkup(job) {
  const percent = Math.min(100, Math.max(0, Number(job.progress) || 0));
  const active = activeStatuses.has(job.status);
  const type = job.kind === 'story' ? 'STORY REEL' : job.kind === 'plan' ? 'CREATOR PLAN' : 'ВИДЕО-НАРЕЗКА';
  const result = job.kind === 'story' ? storyMarkup(job) : job.kind === 'plan' ? planMarkup(job) : clipsMarkup(job);
  return `<article class="job ${job.kind === 'story' ? 'story-job' : job.kind === 'plan' ? 'plan-job' : ''}" id="job-${esc(job.id)}">
    <header class="job-header">
      <div><div class="job-title"><i class="status-dot ${esc(job.status)}"></i>${esc(stageLabel(job.status))}</div><div class="job-subtitle">${type} · ${esc(dateLabel(job.created_at))} · ${esc(job.id.slice(0, 8))}</div></div>
      ${active ? '' : `<button class="delete-job" data-delete="${esc(job.id)}" title="Удалить файлы этой задачи">удалить</button>`}
    </header>
    ${job.status === 'failed' ? `<div class="job-error">${esc(job.error || 'Задача не была завершена.')}</div>` : ''}
    ${job.status !== 'completed' ? `<div class="progress-wrap"><div class="progress-label"><span>${esc(stageLabel(job.status))}</span><span>${percent}%</span></div><div class="progress-bar"><span style="width:${percent}%"></span></div></div>` : ''}
    ${result}${logsMarkup(job)}
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
  document.querySelectorAll('[data-copy]').forEach(button => button.addEventListener('click', async () => {
    try { await navigator.clipboard.writeText(button.dataset.copy); button.textContent = 'скопировано ✓'; setTimeout(() => { button.textContent = 'скопировать ⧉'; }, 1400); }
    catch { alert('Не удалось скопировать автоматически. Выделите текст вручную.'); }
  }));
}
async function refresh() {
  try {
    const response = await fetch('/api/jobs', { cache:'no-store' });
    if (!response.ok) throw new Error('Сервер не ответил');
    jobs = await response.json(); render();
    const active = jobs.some(job => activeStatuses.has(job.status));
    if (active && !pollTimer) pollTimer = setInterval(refresh, 3500);
    if (!active && pollTimer) { clearInterval(pollTimer); pollTimer = null; }
  } catch { runtimeStatus.textContent = 'Нет соединения с локальным сервером'; }
}
function apiError(data) {
  if (typeof data?.detail === 'string') return data.detail;
  if (Array.isArray(data?.detail)) return data.detail.map(item => item.msg).join(' ');
  return 'Не удалось создать задачу.';
}
form.addEventListener('submit', async event => {
  event.preventDefault(); hideError(errorBox);
  const minSeconds = Number(document.querySelector('#min-seconds').value);
  const maxSeconds = Number(document.querySelector('#max-seconds').value);
  if (maxSeconds <= minSeconds) return showError(errorBox, 'Максимальная длительность должна быть больше минимальной.');
  const payload = { source_url:document.querySelector('#source-url').value.trim(), clip_count:Number(document.querySelector('#clip-count').value), min_seconds:minSeconds, max_seconds:maxSeconds, render_height:Number(document.querySelector('#render-height').value), rights_confirmed:document.querySelector('#rights').checked, cloud_processing_allowed:document.querySelector('#cloud-clip-allowed').checked };
  submitButton.disabled = true; submitButton.querySelector('span').textContent = 'Добавляю в очередь…';
  try {
    const response = await fetch('/api/jobs', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload) });
    const data = await response.json(); if (!response.ok) throw new Error(apiError(data));
    form.reset(); document.querySelector('#clip-count').value = '3'; document.querySelector('#render-height').value = '720';
    jobs.unshift(data); render(); document.querySelector('.jobs-section').scrollIntoView({ behavior:'smooth', block:'start' }); await refresh();
  } catch (error) { showError(errorBox, error.message || 'Не удалось создать задачу.'); }
  finally { submitButton.disabled = false; submitButton.querySelector('span').textContent = 'Найти моменты'; }
});
storyForm.addEventListener('submit', async event => {
  event.preventDefault(); hideError(storyError);
  if (config && !config.story_ready) return showError(storyError, 'Настройте локальную Ollama или явный облачный OpenAI-совместимый endpoint. Инструкция и безопасные профили есть в README.');
  const payload = {
    topic:document.querySelector('#story-topic').value.trim(), story_kind:document.querySelector('#story-kind').value,
    language:document.querySelector('#story-language').value, duration_seconds:Number(document.querySelector('#story-duration').value),
    visual_style:document.querySelector('#story-style').value, render_height:Number(document.querySelector('#story-height').value),
    cloud_processing_allowed:document.querySelector('#cloud-story-allowed').checked,
    ai_disclosure_confirmed:document.querySelector('#ai-disclosure').checked,
  };
  storySubmit.disabled = true; storySubmit.querySelector('span').textContent = 'Пишу и собираю…';
  try {
    const response = await fetch('/api/stories', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload) });
    const data = await response.json(); if (!response.ok) throw new Error(apiError(data));
    storyForm.reset(); document.querySelector('#story-duration').value = '45'; document.querySelector('#story-height').value = '720';
    jobs.unshift(data); render(); document.querySelector('.jobs-section').scrollIntoView({ behavior:'smooth', block:'start' }); await refresh();
  } catch (error) { showError(storyError, error.message || 'Не удалось собрать Story Reel.'); }
  finally { storySubmit.disabled = false; storySubmit.querySelector('span').textContent = 'Собрать Story Reel'; }
});
planForm.addEventListener('submit', async event => {
  event.preventDefault(); hideError(planError);
  if (config && !config.story_ready) return showError(planError, 'Для Creator Plan настройте локальную Ollama или cloud endpoint.');
  const platforms = [...document.querySelectorAll('[name="plan-platform"]:checked')].map(input => input.value);
  if (!platforms.length) return showError(planError, 'Выберите хотя бы одну площадку.');
  const payload = {
    niche: document.querySelector('#plan-niche').value.trim(), audience: document.querySelector('#plan-audience').value.trim(),
    content_focus: document.querySelector('#plan-focus').value, platforms, plan_days: Number(document.querySelector('#plan-days').value),
    language: document.querySelector('#plan-language').value, cloud_processing_allowed: document.querySelector('#cloud-plan-allowed').checked,
  };
  planSubmit.disabled = true; planSubmit.querySelector('span').textContent = 'Собираю календарь…';
  try {
    const response = await fetch('/api/plans', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload) });
    const data = await response.json(); if (!response.ok) throw new Error(apiError(data));
    planForm.reset(); document.querySelectorAll('[name="plan-platform"]').forEach(input => input.checked = true);
    jobs.unshift(data); render(); document.querySelector('.jobs-section').scrollIntoView({ behavior:'smooth', block:'start' }); await refresh();
  } catch (error) { showError(planError, error.message || 'Не удалось собрать контент-план.'); }
  finally { planSubmit.disabled = false; planSubmit.querySelector('span').textContent = 'Собрать план на неделю'; }
});
document.querySelectorAll('[data-idea]').forEach(button => button.addEventListener('click', () => {
  document.querySelector('#story-topic').value = button.dataset.idea; document.querySelector('#story-topic').focus();
}));
document.querySelectorAll('.tool-tab').forEach(button => button.addEventListener('click', () => {
  document.querySelectorAll('.tool-tab').forEach(tab => tab.classList.toggle('is-active', tab === button));
  document.querySelectorAll('.tool-panel').forEach(panel => panel.hidden = panel.id !== button.dataset.panel);
  document.querySelector(`#${button.dataset.panel}`).scrollIntoView({ behavior:'smooth', block:'start' });
}));
document.querySelector('#refresh').addEventListener('click', refresh);
(async () => {
  try {
    const response = await fetch('/api/config'); config = await response.json();
    const voice = config.piper_voice_configured ? 'Piper настроен' : 'eSpeak local';
    const provider = config.llm_provider === 'cloud' ? 'облачная LLM' : config.llm_provider === 'local' ? 'локальная LLM' : 'LLM не настроена';
    runtimeStatus.textContent = `Whisper ${config.whisper_model} · ${provider} · ${voice}`;
    if (config.cloud_consent_required) {
      cloudClipConsent.hidden = false;
      cloudStoryConsent.hidden = false;
      cloudPlanConsent.hidden = false;
    }
    const limits = config.source_limits || {};
    const minutes = limits.minutes ? `${limits.minutes} мин` : 'без лимита времени';
    const megabytes = limits.megabytes ? `${limits.megabytes} МБ` : 'без лимита размера';
    hardwareProfile.innerHTML = `<strong>CPU-профиль:</strong> Whisper ${esc(config.whisper_model)} · ${esc(config.whisper_cpu_threads)} потоков · FFmpeg ${esc(config.ffmpeg_threads)} потоков · источник: ${esc(minutes)}, ${esc(megabytes)}.`;
  } catch { runtimeStatus.textContent = 'Локальный режим'; }
  refresh();
})();
