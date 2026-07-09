const bridge = window.AstrBotPluginPage;

const LOG_PAGE_SIZE = 20;

let logPage = 0;
let selectedGroup = "";
let currentTab = "manage";

await bridge.ready();

function escapeHtml(str) {
  if (!str) return "";
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
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

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach((c) => c.classList.remove("active"));
    tab.classList.add("active");
    const name = tab.getAttribute("data-tab");
    document.getElementById(`tab-${name}`).classList.add("active");
    currentTab = name;
    if (name === "manage") renderManage();
    else if (name === "stats") renderStats();
    else if (name === "settings") renderSettings();
  });
});

async function loadWhitelist() {
  try {
    return await bridge.apiGet("whitelist/list");
  } catch {
    return [];
  }
}

async function loadStatsStatus() {
  try {
    return await bridge.apiGet("stats/status");
  } catch {
    return { enabled: false };
  }
}

async function loadStatsGroups() {
  try {
    return await bridge.apiGet("stats/groups");
  } catch {
    return [];
  }
}

async function loadLogs(groupId, page) {
  try {
    const params = { limit: LOG_PAGE_SIZE, offset: page * LOG_PAGE_SIZE };
    if (groupId) params.group_id = groupId;
    return await bridge.apiGet("stats/logs", params);
  } catch {
    return [];
  }
}

async function renderManage() {
  const container = document.getElementById("tab-manage");
  const groups = await loadWhitelist();

  let rows = "";
  for (const g of groups) {
    const followUsers = (g.follow_up_users && g.follow_up_users.length)
      ? g.follow_up_users.map(u => `<span class="badge badge-following">${escapeHtml(u)}</span>`).join(" ")
      : '<span class="muted">-</span>';
    const followKeywords = (g.follow_up_keywords && g.follow_up_keywords.length)
      ? g.follow_up_keywords.map(k => `<span class="badge badge-drift">${escapeHtml(k)}</span>`).join(" ")
      : '<span class="muted">-</span>';
    const followReason = g.follow_up_reason ? escapeHtml(g.follow_up_reason) : '<span class="muted">-</span>';

    rows += `<tr>
      <td>${escapeHtml(g.group_id)}</td>
      <td>
        <label class="toggle-switch">
          <input type="checkbox" checked data-group="${escapeHtml(g.group_id)}" class="group-toggle" />
          <span class="toggle-slider"></span>
        </label>
        <span class="badge badge-on">已启用</span>
      </td>
      <td>${stateBadge(g.state)}</td>
      <td>
        <select class="willingness-select" data-group="${escapeHtml(g.group_id)}">
          <option value="low" ${g.willingness === "low" ? "selected" : ""}>低</option>
          <option value="medium" ${g.willingness === "medium" ? "selected" : ""}>中</option>
          <option value="high" ${g.willingness === "high" ? "selected" : ""}>高</option>
        </select>
      </td>
      <td>${g.msg_count}</td>
      <td>${g.effective_n}/${g.effective_t}m</td>
      <td>${g.backoff_level}</td>
      <td>${g.consecutive_replies}</td>
      <td>${followUsers}</td>
      <td>${followKeywords}</td>
      <td>${followReason}</td>
      <td>
        <button class="action-btn" data-action="reset" data-group="${escapeHtml(g.group_id)}">重置</button>
        <button class="action-btn danger" data-action="disable" data-group="${escapeHtml(g.group_id)}">禁用</button>
      </td>
    </tr>`;
  }

  const emptyHtml = !groups.length ? '<div class="empty">暂无已启用的群聊</div>' : "";

  container.innerHTML = `
    <div class="add-group-row">
      <input type="text" id="newGroupId" placeholder="输入群ID以添加到白名单" />
      <button class="btn-success" id="addGroupBtn">启用群聊</button>
    </div>
    ${emptyHtml}
    ${groups.length ? `<table>
      <thead>
        <tr>
          <th>群ID</th>
          <th>状态</th>
          <th>状态机</th>
          <th>意愿</th>
          <th>消息</th>
          <th>阈值 N/T</th>
          <th>退避</th>
          <th>连续</th>
          <th>跟进用户</th>
          <th>跟进关键词</th>
          <th>跟进原因</th>
          <th>操作</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>` : ""}
  `;

  document.getElementById("addGroupBtn").addEventListener("click", async () => {
    const input = document.getElementById("newGroupId");
    const gid = input.value.trim();
    if (!gid) return;
    await bridge.apiPost("whitelist/enable", { group_id: gid });
    input.value = "";
    renderManage();
  });

  document.getElementById("newGroupId").addEventListener("keydown", (e) => {
    if (e.key === "Enter") document.getElementById("addGroupBtn").click();
  });

  container.querySelectorAll(".willingness-select").forEach((sel) => {
    sel.addEventListener("change", async () => {
      const gid = sel.getAttribute("data-group");
      await bridge.apiPost("group/set_willingness", {
        group_id: gid,
        willingness: sel.value,
      });
    });
  });

  container.querySelectorAll(".action-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const action = btn.getAttribute("data-action");
      const gid = btn.getAttribute("data-group");
      if (action === "disable") {
        if (!confirm(`确定要禁用群 ${gid} 的主动回复吗？`)) return;
        await bridge.apiPost("whitelist/disable", { group_id: gid });
        renderManage();
      } else if (action === "reset") {
        if (!confirm(`确定要重置群 ${gid} 的状态吗？`)) return;
        await bridge.apiPost("group/reset", { group_id: gid });
        renderManage();
      }
    });
  });
}

async function renderStats() {
  const container = document.getElementById("tab-stats");
  const status = await loadStatsStatus();

  if (!status.enabled) {
    container.innerHTML = `
      <div class="stats-disabled">
        <h2>统计监控未启用</h2>
        <p>请在插件配置中开启「启用统计监控」选项后刷新此页面</p>
      </div>`;
    return;
  }

  const groups = await loadStatsGroups();

  let groupOptions = '<option value="">全部群聊</option>';
  for (const g of groups) {
    groupOptions += `<option value="${escapeHtml(g.group_id)}">${escapeHtml(g.group_id)}</option>`;
  }

  container.innerHTML = `
    <div class="toolbar">
      <select id="groupFilter">${groupOptions}</select>
      <button id="refreshBtn">刷新</button>
      <button id="clearBtn" class="btn-danger">清除记录</button>
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
    renderStatsGroups(groups);
    renderStatsLogs();
  });
  document.getElementById("refreshBtn").addEventListener("click", () => {
    renderStats();
  });
  document.getElementById("clearBtn").addEventListener("click", async () => {
    if (!confirm("确定要清除所有统计数据吗？")) return;
    await bridge.apiPost("stats/clear");
    renderStats();
  });

  renderStatsGroups(groups);
  renderStatsLogs();
}

function renderStatsGroups(groups) {
  const container = document.getElementById("groupTable");
  if (!container) return;
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
        <th>错误</th>
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
      <td>${g.total_errors}</td>
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

async function renderStatsLogs() {
  const container = document.getElementById("logList");
  const pagination = document.getElementById("logPagination");
  if (!container) return;
  container.innerHTML = '<div class="loading">加载中...</div>';

  const logs = await loadLogs(selectedGroup, logPage);

  if (!logs.length) {
    container.innerHTML = '<div class="empty">暂无调用日志</div>';
    if (pagination) pagination.innerHTML = "";
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
          ${log.follow_up_keywords && log.follow_up_keywords.length ? `<label>关注关键词</label><pre>${escapeHtml(log.follow_up_keywords.join(", "))}</pre>` : ""}
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

  if (pagination) {
    pagination.innerHTML = `
      <button id="prevPage" ${logPage === 0 ? "disabled" : ""}>上一页</button>
      <span>第 ${logPage + 1} 页</span>
      <button id="nextPage" ${logs.length < LOG_PAGE_SIZE ? "disabled" : ""}>下一页</button>
    `;

    document.getElementById("prevPage").addEventListener("click", () => {
      if (logPage > 0) { logPage--; renderStatsLogs(); }
    });
    document.getElementById("nextPage").addEventListener("click", () => {
      logPage++; renderStatsLogs();
    });
  }
}

async function renderSettings() {
  const container = document.getElementById("tab-settings");
  container.innerHTML = '<div class="loading">加载中...</div>';

  let data;
  try {
    data = await bridge.apiGet("config/get");
  } catch {
    container.innerHTML = '<div class="empty">加载配置失败</div>';
    return;
  }

  const { values, meta } = data;
  let html = "";

  const orderedKeys = [
    "mute_period", "window_size", "default_n", "default_t", "max_token",
    "quality_threshold", "follow_up_ttl", "follow_up_aggregate_window",
    "trigger_min_interval", "boost_factor", "boost_duration", "max_boosted_replies",
  ];

  html += '<div class="config-section">';
  html += '<h3>基本参数</h3>';

  for (const key of orderedKeys) {
    const m = meta[key];
    if (!m) continue;
    const val = values[key];
    const hint = m.hint ? `<div class="config-hint">${escapeHtml(m.hint)}</div>` : "";

    if (m.type === "object") {
      let subItems = "";
      for (const [subKey, subMeta] of Object.entries(m.items)) {
        const subVal = val ? val[subKey] : subMeta.min;
        subItems += `<div class="config-sub-item">
          <span>${escapeHtml(subMeta.label)}</span>
          <input type="number" data-config="${key}.${subKey}" value="${subVal}" min="${subMeta.min}" max="${subMeta.max}" />
        </div>`;
      }
      html += `<div class="config-row">
        <label>${escapeHtml(m.label)}</label>
        <div class="config-sub-group">${subItems}</div>
        ${hint}
      </div>`;
    } else {
      const step = m.step || (m.type === "float" ? "0.01" : "1");
      html += `<div class="config-row">
        <label>${escapeHtml(m.label)}</label>
        <input type="number" data-config="${key}" value="${val}" min="${m.min}" max="${m.max}" step="${step}" />
        ${hint}
      </div>`;
    }
  }

  html += '</div>';

  html += `<div class="config-save-bar">
    <button class="btn-primary" id="saveConfigBtn">保存配置</button>
    <span class="config-msg" id="configMsg"></span>
  </div>`;

  container.innerHTML = html;

  document.getElementById("saveConfigBtn").addEventListener("click", async () => {
    const msg = document.getElementById("configMsg");
    msg.className = "config-msg";
    msg.textContent = "保存中...";

    const payload = {};
    container.querySelectorAll("[data-config]").forEach((input) => {
      const path = input.getAttribute("data-config").split(".");
      if (path.length === 2) {
        if (!payload[path[0]]) payload[path[0]] = {};
        payload[path[0]][path[1]] = parseFloat(input.value);
      } else {
        payload[path[0]] = parseFloat(input.value);
      }
    });

    try {
      await bridge.apiPost("config/set", payload);
      msg.textContent = "保存成功";
    } catch {
      msg.className = "config-msg error";
      msg.textContent = "保存失败";
    }
  });
}

renderManage();
