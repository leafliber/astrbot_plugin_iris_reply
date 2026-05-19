const bridge = window.AstrBotPluginPage;

const GROUP_PAGE_SIZE = 20;
const LOG_PAGE_SIZE = 20;

let logPage = 0;
let selectedGroup = "";

await bridge.ready();

async function apiGet(endpoint, params) {
  return bridge.apiGet(endpoint, params);
}

function formatTime(ts) {
  if (!ts) return "-";
  const d = new Date(ts * 1000);
  return d.toLocaleString("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function formatDuration(ms) {
  if (!ms) return "-";
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

function stateBadge(state) {
  const map = {
    idle: ["空闲", "badge-idle"],
    cooldown: ["冷却", "badge-cooldown"],
    following: ["跟进", "badge-following"],
  };
  const [text, cls] = map[state] || [state, "badge-idle"];
  return `<span class="badge ${cls}">${text}</span>`;
}

function willingnessSpan(level) {
  const map = { low: "低", medium: "中", high: "高" };
  return `<span class="willingness-${level}">${map[level] || level}</span>`;
}

function resultBadge(log) {
  if (log.topic_drifted) return `<span class="badge badge-drift">话题偏移</span>`;
  if (log.should_reply) return `<span class="badge badge-reply">回复</span>`;
  return `<span class="badge badge-skip">跳过</span>`;
}

function escapeHtml(str) {
  if (!str) return "";
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

async function loadStatus() {
  try {
    return await apiGet("stats/status");
  } catch {
    return { enabled: false };
  }
}

async function loadGroups() {
  try {
    return await apiGet("stats/groups");
  } catch {
    return [];
  }
}

async function loadLogs(groupId, page) {
  try {
    const params = { limit: LOG_PAGE_SIZE, offset: page * LOG_PAGE_SIZE };
    if (groupId) params.group_id = groupId;
    return await apiGet("stats/logs", params);
  } catch {
    return [];
  }
}

async function render() {
  const content = document.getElementById("content");
  const status = await loadStatus();

  if (!status.enabled) {
    content.innerHTML = `
      <div class="stats-disabled">
        <h2>统计监控未启用</h2>
        <p>请在插件配置中开启「启用统计监控」选项后刷新此页面</p>
      </div>`;
    return;
  }

  const groups = await loadGroups();

  let groupOptions = '<option value="">全部群聊</option>';
  for (const g of groups) {
    groupOptions += `<option value="${escapeHtml(g.group_id)}">${escapeHtml(g.group_id)}</option>`;
  }

  content.innerHTML = `
    <div class="toolbar">
      <select id="groupFilter">${groupOptions}</select>
      <button id="refreshBtn">刷新</button>
      <button id="clearBtn">清除记录</button>
    </div>

    <h2>群聊概览</h2>
    <div id="groupTable"></div>

    <h2>LLM 调用日志</h2>
    <div id="logList"></div>
    <div id="logPagination" class="pagination"></div>
  `;

  document.getElementById("groupFilter").value = selectedGroup;
  document.getElementById("groupFilter").addEventListener("change", (e) => {
    selectedGroup = e.target.value;
    logPage = 0;
    renderGroups(groups);
    renderLogs();
  });
  document.getElementById("refreshBtn").addEventListener("click", () => {
    render();
  });
  document.getElementById("clearBtn").addEventListener("click", async () => {
    if (!confirm("确定要清除所有统计数据吗？")) return;
    await bridge.apiPost("stats/clear");
    render();
  });

  renderGroups(groups);
  renderLogs();
}

function renderGroups(groups) {
  const container = document.getElementById("groupTable");
  if (!groups.length) {
    container.innerHTML = '<div class="empty">暂无群聊数据</div>';
    return;
  }

  let html = `<table>
    <thead>
      <tr>
        <th>群ID</th>
        <th>状态</th>
        <th>意愿</th>
        <th>触发</th>
        <th>回复</th>
        <th>跳过</th>
        <th>偏移</th>
        <th>被动</th>
        <th>消息</th>
        <th>阈值 N/T</th>
        <th>退避</th>
        <th>连续</th>
        <th>最近触发</th>
      </tr>
    </thead>
    <tbody>`;

  for (const g of groups) {
    html += `<tr>
      <td>${escapeHtml(g.group_id)}</td>
      <td>${stateBadge(g.current_state)}</td>
      <td>${willingnessSpan(g.willingness)}</td>
      <td>${g.total_triggers}</td>
      <td>${g.total_replies}</td>
      <td>${g.total_skips}</td>
      <td>${g.total_drifts}</td>
      <td>${g.total_passive_replies}</td>
      <td>${g.msg_count}</td>
      <td>${g.effective_n}/${g.effective_t}m</td>
      <td>${g.backoff_level}</td>
      <td>${g.consecutive_replies}</td>
      <td>${formatTime(g.last_trigger_time)}</td>
    </tr>`;
  }

  html += "</tbody></table>";
  container.innerHTML = html;
}

async function renderLogs() {
  const container = document.getElementById("logList");
  const pagination = document.getElementById("logPagination");
  container.innerHTML = '<div class="loading">加载中...</div>';

  const logs = await loadLogs(selectedGroup, logPage);

  if (!logs.length) {
    container.innerHTML = '<div class="empty">暂无调用日志</div>';
    pagination.innerHTML = "";
    return;
  }

  let html = "";
  for (const log of logs) {
    const id = `log-${log.timestamp}-${log.group_id}`.replace(/[^a-zA-Z0-9-]/g, "");
    html += `
      <div class="log-entry">
        <div class="log-header" data-target="${id}">
          <div class="log-meta">
            ${resultBadge(log)}
            <span>${escapeHtml(log.group_id)}</span>
            <span class="badge badge-passive">${log.trigger_reason}</span>
            <span class="duration">${formatDuration(log.duration_ms)}</span>
          </div>
          <span class="log-time">${formatTime(log.timestamp)}</span>
        </div>
        <div class="log-detail" id="${id}">
          <label>观察摘要</label>
          <pre>${escapeHtml(log.observation) || "-"}</pre>
          ${log.follow_up_users && log.follow_up_users.length ? `<label>关注用户</label><pre>${escapeHtml(log.follow_up_users.join(", "))}</pre>` : ""}
          ${log.interest_reason ? `<label>关注原因</label><pre>${escapeHtml(log.interest_reason)}</pre>` : ""}
          <label>LLM 原始响应</label>
          <pre>${escapeHtml(log.response_text)}</pre>
          <label>System Prompt</label>
          <pre>${escapeHtml(log.system_prompt)}</pre>
          <label>User Prompt</label>
          <pre>${escapeHtml(log.user_prompt)}</pre>
        </div>
      </div>`;
  }

  container.innerHTML = html;

  container.querySelectorAll(".log-header").forEach((header) => {
    header.addEventListener("click", () => {
      const targetId = header.getAttribute("data-target");
      const detail = document.getElementById(targetId);
      if (detail) detail.classList.toggle("open");
    });
  });

  pagination.innerHTML = `
    <button id="prevPage" ${logPage === 0 ? "disabled" : ""}>上一页</button>
    <span>第 ${logPage + 1} 页</span>
    <button id="nextPage" ${logs.length < LOG_PAGE_SIZE ? "disabled" : ""}>下一页</button>
  `;

  document.getElementById("prevPage").addEventListener("click", () => {
    if (logPage > 0) { logPage--; renderLogs(); }
  });
  document.getElementById("nextPage").addEventListener("click", () => {
    logPage++; renderLogs();
  });
}

render();
