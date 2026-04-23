const REFRESH_MS = 60 * 1000;
const WINDOW_HOURS = 12;
const FORECAST_DELTA_WINDOW_HOURS = 2;

async function fetchJson(url) {
  const response = await fetch(`${url}?ts=${Date.now()}`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Falha ao carregar ${url}`);
  }
  return response.json();
}

function formatNumber(value, digits = 1) {
  return Number(value).toFixed(digits).replace(".", ",");
}

function formatDate(value) {
  if (!value) return "--";
  return new Date(value).toLocaleString("pt-BR");
}

function formatHour(value) {
  if (!value) return "--";
  return new Date(value).toLocaleTimeString("pt-BR", { hour: "2-digit", minute: "2-digit" });
}

function updateHeader(metrics, state) {
  const simple2h = metrics.forecast?.simple_glucose_in_2h_mg_dl ?? metrics.forecast?.glucose_in_2h_mg_dl;
  const tsMl1h = metrics.forecast?.takagi_sugeno_ml_glucose_in_1h_mg_dl ?? metrics.forecast?.glucose_in_2h_mg_dl;
  const source2h = metrics.forecast?.forecast_primary_source || "--";
  const exerciseCount = metrics.inputs.exercise_count ?? 0;
  document.getElementById("generated-at").textContent = formatDate(metrics.generated_at || state.generated_at);
  document.getElementById("rmse").textContent = `${formatNumber(metrics.metrics.rmse_mg_dl, 2)} mg/dL`;
  document.getElementById("mae").textContent = `${formatNumber(metrics.metrics.mae_mg_dl, 2)} mg/dL`;
  document.getElementById("latest-glucose").textContent = `${formatNumber(state.latest_glucose_mg_dl, 0)} mg/dL`;
  document.getElementById("forecast-glucose").textContent = `${formatNumber(simple2h, 0)} mg/dL`;
  document.getElementById("hybrid-forecast-glucose").textContent = `${formatNumber(tsMl1h, 0)} mg/dL`;
  document.getElementById("forecast-delta-now").textContent = source2h;
  document.getElementById("inputs-count").textContent = `${metrics.inputs.meals_count} refeições / ${metrics.inputs.insulin_count} insulinas / ${exerciseCount} exercícios`;
  document.getElementById("status-dot").classList.add("ok");
}

function createSummaryRow(label, value) {
  const wrapper = document.createElement("div");
  const dt = document.createElement("dt");
  dt.textContent = label;
  const dd = document.createElement("dd");
  dd.textContent = value;
  wrapper.append(dt, dd);
  return wrapper;
}

function updateStateSummary(state) {
  const target = document.getElementById("state-summary");
  target.innerHTML = "";
  const initialState = state.inverse?.estimated_initial_state || {};
  const params = state.params || {};
  const regime = state.regime || {};
  const rows = [
    ["Último timestamp", formatDate(state.latest_timestamp)],
    ["Regime", regime.name ? `${regime.name} (${regime.reason || "contexto"})` : "--"],
    ["G0", `${formatNumber(initialState.G0 || 0, 0)} mg`],
    ["Ip0", formatNumber(initialState.Ip0 || 0, 2)],
    ["Ii0", formatNumber(initialState.Ii0 || 0, 2)],
    ["k_a", formatNumber(params.ka || 0, 3)],
    ["Meal tau", `${formatNumber(params.meal_tau || 0, 0)} min`],
    ["Bioavailability", formatNumber(params.insulin_bioavailability || 0, 2)],
    ["Inclinação recente", `${formatNumber(regime.recent_slope_mgdl_per_min || 0, 2)} mg/dL/min`],
    ["Input inferido", `${formatNumber(state.inverse?.mean_inferred_glucose_input_mg_min || 0, 1)} mg/min`],
  ];
  rows.forEach(([label, value]) => target.appendChild(createSummaryRow(label, value)));
}

function renderEvents(state) {
  const target = document.getElementById("events-summary");
  target.innerHTML = "";
  const meals = (state.meals || []).slice(-3).reverse();
  const insulin = (state.insulin || []).slice(-3).reverse();
  const exercise = (state.exercise || []).slice(-3).reverse();

  if (!meals.length && !insulin.length && !exercise.length) {
    target.innerHTML = '<div class="event-item"><strong>Sem eventos recentes</strong><p>As planilhas ainda não trouxeram refeições, insulina ou exercício dentro da janela atual.</p></div>';
    return;
  }

  meals.forEach((event) => {
    const div = document.createElement("div");
    div.className = "event-item";
    div.innerHTML = `<strong>Refeição: ${formatNumber(event.carbs_g, 0)} g CHO</strong><p>${formatDate(event.timestamp)}</p>`;
    target.appendChild(div);
  });

  insulin.forEach((event) => {
    const div = document.createElement("div");
    div.className = "event-item";
    const insulinType = event.insulin_type && event.insulin_type !== "desconhecido" ? ` (${event.insulin_type})` : "";
    div.innerHTML = `<strong>Insulina: ${formatNumber(event.units, 1)} U${insulinType}</strong><p>${formatDate(event.timestamp)}</p>`;
    target.appendChild(div);
  });

  exercise.forEach((event) => {
    const div = document.createElement("div");
    div.className = "event-item";
    div.innerHTML = `<strong>Exercício: ${formatNumber(event.duration_minutes, 0)} min @ ${formatNumber(event.heart_rate_bpm, 0)} bpm</strong><p>${formatDate(event.timestamp)}</p>`;
    target.appendChild(div);
  });
}

function buildPath(valuesX, valuesY, width, height, padding, minX, maxX, yMin, yMax) {
  return valuesX.map((x, index) => {
    const scaledX = padding + ((x - minX) / (maxX - minX || 1)) * (width - padding * 2);
    const scaledY = height - padding - ((valuesY[index] - yMin) / (yMax - yMin || 1)) * (height - padding * 2);
    return `${index === 0 ? "M" : "L"} ${scaledX.toFixed(2)} ${scaledY.toFixed(2)}`;
  }).join(" ");
}

function scaleX(x, minX, maxX, width, padding) {
  return padding + ((x - minX) / (maxX - minX || 1)) * (width - padding * 2);
}

function scaleY(y, minY, maxY, height, padding) {
  return height - padding - ((y - minY) / (maxY - minY || 1)) * (height - padding * 2);
}

function buildValueLabels(points, color, minX, maxX, minY, maxY, width, height, padding) {
  return points.map((point) => {
    const x = scaleX(point.time, minX, maxX, width, padding);
    const y = scaleY(point.value, minY, maxY, height, padding);
    return `
      <circle cx="${x.toFixed(2)}" cy="${y.toFixed(2)}" r="3.5" fill="${color}"></circle>
      <text x="${x.toFixed(2)}" y="${(y - 10).toFixed(2)}" text-anchor="middle" fill="${color}" font-size="11" font-weight="700">${Math.round(point.value)}</text>
    `;
  }).join("");
}

function renderChart(series) {
  const svg = document.getElementById("glucose-chart");
  const width = 900;
  const height = 320;
  const padding = 28;
  const observedTimes = (series.observed_timestamps || []).map((value) => new Date(value).getTime());
  const simulatedTimes = (series.timestamps || []).map((value) => new Date(value).getTime());
  const observedY = series.observed_glucose_mg_dl || [];
  const simulatedY = series.glucose_mg_dl || [];
  const simpleHorizons = (series.simple_forecast_horizons || []).map((item) => ({
    time: new Date(item.forecast_target_timestamp).getTime(),
    value: Number(item.predicted_glucose_mg_dl),
    status: item.prediction_status,
  }));
  const tsMlHorizons = (series.takagi_sugeno_ml_forecast_horizons || []).map((item) => ({
    time: new Date(item.forecast_target_timestamp).getTime(),
    value: Number(item.predicted_glucose_mg_dl),
    status: item.prediction_status,
  }));
  if (!observedTimes.length || !simulatedTimes.length) {
    svg.innerHTML = "";
    return;
  }

  const endTime = Math.max(observedTimes[observedTimes.length - 1], simulatedTimes[simulatedTimes.length - 1]);
  const startTime = endTime - WINDOW_HOURS * 60 * 60 * 1000;

  const observedPairs = observedTimes.map((time, index) => ({ time, value: observedY[index] })).filter((point) => point.time >= startTime);
  const simulatedPairs = simulatedTimes.map((time, index) => ({ time, value: simulatedY[index] })).filter((point) => point.time >= startTime);
  const forecastStartTime = new Date(series.forecast_start_timestamp).getTime();

  if (!observedPairs.length || !simulatedPairs.length) {
    svg.innerHTML = "";
    return;
  }

  const observedX = observedPairs.map((point) => point.time);
  const observedFilteredY = observedPairs.map((point) => point.value);
  const historicalPairs = simulatedPairs.filter((point) => point.time <= forecastStartTime);
  const forecastPairs = simulatedPairs.filter((point) => point.time >= forecastStartTime);

  if (!historicalPairs.length || !forecastPairs.length) {
    svg.innerHTML = "";
    return;
  }

  const historicalX = historicalPairs.map((point) => point.time);
  const historicalY = historicalPairs.map((point) => point.value);
  const futureSimplePoints = simpleHorizons.filter((point) => point.time >= forecastStartTime && point.time <= endTime);
  const simpleCurvePoints = [{ time: forecastStartTime, value: historicalY[historicalY.length - 1] }, ...futureSimplePoints];
  const simpleX = simpleCurvePoints.map((point) => point.time);
  const simpleY = simpleCurvePoints.map((point) => point.value);

  const tsMlVisible = tsMlHorizons.filter((point) => point.time >= forecastStartTime && point.time <= endTime && point.status === "trained");
  const minY = Math.min(...observedFilteredY, ...historicalY, ...simpleY, ...(tsMlVisible.length ? tsMlVisible.map((point) => point.value) : simpleY)) - 10;
  const maxY = Math.max(...observedFilteredY, ...historicalY, ...simpleY, ...(tsMlVisible.length ? tsMlVisible.map((point) => point.value) : simpleY)) + 10;
  const minX = Math.min(observedX[0], historicalX[0], simpleX[0]);
  const maxX = Math.max(observedX[observedX.length - 1], historicalX[historicalX.length - 1], simpleX[simpleX.length - 1]);
  const gridLines = [0.2, 0.4, 0.6, 0.8].map((ratio) => {
    const y = padding + ratio * (height - padding * 2);
    return `<line x1="${padding}" y1="${y}" x2="${width - padding}" y2="${y}" stroke="rgba(31,41,51,0.10)" stroke-width="1" />`;
  }).join("");

  const observedPath = buildPath(observedX, observedFilteredY, width, height, padding, minX, maxX, minY, maxY);
  const historicalPath = buildPath(historicalX, historicalY, width, height, padding, minX, maxX, minY, maxY);
  const simplePath = buildPath(simpleX, simpleY, width, height, padding, minX, maxX, minY, maxY);
  const forecastStartX = padding + ((forecastStartTime - minX) / (maxX - minX || 1)) * (width - padding * 2);
  const observedLabels = [observedPairs[0], observedPairs[observedPairs.length - 1]];
  const historicalLabels = [historicalPairs[0], historicalPairs[historicalPairs.length - 1]];
  const simpleVisible = futureSimplePoints.filter((point) => point.time >= minX && point.time <= maxX);
  const simpleLabelPoints = simpleVisible.length ? [simpleVisible[simpleVisible.length - 1]] : [];
  const simpleMarkers = simpleVisible.map((point) => {
    const x = scaleX(point.time, minX, maxX, width, padding);
    const y = scaleY(point.value, minY, maxY, height, padding);
    return `
      <circle cx="${x.toFixed(2)}" cy="${y.toFixed(2)}" r="4.5" fill="#dc2626"></circle>
      <text x="${x.toFixed(2)}" y="${(y - 12).toFixed(2)}" text-anchor="middle" fill="#dc2626" font-size="11" font-weight="700">${Math.round(point.value)}</text>
    `;
  }).join("");
  const tsMlMarkers = tsMlVisible.map((point) => {
    const x = scaleX(point.time, minX, maxX, width, padding);
    const y = scaleY(point.value, minY, maxY, height, padding);
    return `
      <circle cx="${x.toFixed(2)}" cy="${y.toFixed(2)}" r="4.5" fill="#7c3aed"></circle>
      <text x="${x.toFixed(2)}" y="${(y - 12).toFixed(2)}" text-anchor="middle" fill="#7c3aed" font-size="11" font-weight="700">${Math.round(point.value)}</text>
    `;
  }).join("");
  const tsMlPath = tsMlVisible.length
    ? buildPath(
        [forecastStartTime, ...tsMlVisible.map((point) => point.time)],
        [historicalY[historicalY.length - 1], ...tsMlVisible.map((point) => point.value)],
        width,
        height,
        padding,
        minX,
        maxX,
        minY,
        maxY,
      )
    : "";

  svg.innerHTML = `
    <rect x="0" y="0" width="${width}" height="${height}" fill="transparent"></rect>
    <rect x="${forecastStartX.toFixed(2)}" y="${padding}" width="${Math.max(width - padding - forecastStartX, 0).toFixed(2)}" height="${height - padding * 2}" fill="rgba(15,118,110,0.08)"></rect>
    ${gridLines}
    <path d="${historicalPath}" fill="none" stroke="#1d4ed8" stroke-width="3" stroke-linecap="round"></path>
    <path d="${simplePath}" fill="none" stroke="#dc2626" stroke-width="3" stroke-linecap="round" stroke-dasharray="7 5"></path>
    ${tsMlVisible.length ? `<path d="${tsMlPath}" fill="none" stroke="#7c3aed" stroke-width="3" stroke-linecap="round" stroke-dasharray="5 4"></path>` : ""}
    <path d="${observedPath}" fill="none" stroke="#b45309" stroke-width="2.5" stroke-linecap="round" stroke-dasharray="2 0"></path>
    <line x1="${forecastStartX.toFixed(2)}" y1="${padding}" x2="${forecastStartX.toFixed(2)}" y2="${height - padding}" stroke="#0f766e" stroke-width="2" stroke-dasharray="6 6"></line>
    ${simpleMarkers}
    ${tsMlMarkers}
    ${buildValueLabels(observedLabels, "#b45309", minX, maxX, minY, maxY, width, height, padding)}
    ${buildValueLabels(historicalLabels, "#1d4ed8", minX, maxX, minY, maxY, width, height, padding)}
    ${buildValueLabels(simpleLabelPoints, "#dc2626", minX, maxX, minY, maxY, width, height, padding)}
  `;

  const axis = document.getElementById("chart-axis");
  const steps = 5;
  const labels = [];
  for (let index = 0; index < steps; index += 1) {
    const ratio = index / (steps - 1);
    labels.push(formatHour(new Date(startTime + ratio * (endTime - startTime)).toISOString()));
  }
  axis.innerHTML = labels.map((label) => `<span>${label}</span>`).join("");
}

function renderForecastDeltaChart(series) {
  const svg = document.getElementById("forecast-delta-chart");
  const axis = document.getElementById("forecast-delta-axis");
  const width = 900;
  const height = 260;
  const padding = 30;
  const rawHistory = (series.forecast_history || []).map((item) => ({
    time: new Date(item.generated_at).getTime(),
    value: Number(item.delta_mg_dl),
  }));
  const referenceTime = Number.isFinite(new Date(series.generated_at || "").getTime())
    ? new Date(series.generated_at).getTime()
    : (rawHistory.length ? rawHistory[rawHistory.length - 1].time : Date.now());
  const cutoff = referenceTime - FORECAST_DELTA_WINDOW_HOURS * 60 * 60 * 1000;
  const history = rawHistory.filter((item) => item.time >= cutoff);

  if (!history.length) {
    svg.innerHTML = `
      <rect x="0" y="0" width="${width}" height="${height}" fill="transparent"></rect>
      <text x="${width / 2}" y="${height / 2}" text-anchor="middle" fill="#52606d" font-size="16">
        Ainda não há projeções resolvidas nas últimas 2 horas.
      </text>
    `;
    axis.innerHTML = "";
    return;
  }

  const minX = history[0].time;
  const maxX = history[history.length - 1].time;
  const minY = Math.min(...history.map((point) => point.value), 0) - 5;
  const maxY = Math.max(...history.map((point) => point.value), 0) + 5;
  const zeroY = scaleY(0, minY, maxY, height, padding);
  const bars = history.map((point) => {
    const x = scaleX(point.time, minX, maxX, width, padding);
    const y = scaleY(point.value, minY, maxY, height, padding);
    const barWidth = Math.max((width - padding * 2) / Math.max(history.length * 1.8, 8), 8);
    const barHeight = Math.abs(zeroY - y);
    const topY = Math.min(y, zeroY);
    const color = point.value >= 0 ? "#be123c" : "#0f766e";
    return `
      <rect x="${(x - barWidth / 2).toFixed(2)}" y="${topY.toFixed(2)}" width="${barWidth.toFixed(2)}" height="${barHeight.toFixed(2)}" fill="${color}" rx="4"></rect>
      <text x="${x.toFixed(2)}" y="${(topY - 8).toFixed(2)}" text-anchor="middle" fill="${color}" font-size="11" font-weight="700">${Math.round(point.value)}</text>
    `;
  }).join("");

  const gridLines = [0.2, 0.4, 0.6, 0.8].map((ratio) => {
    const y = padding + ratio * (height - padding * 2);
    return `<line x1="${padding}" y1="${y}" x2="${width - padding}" y2="${y}" stroke="rgba(31,41,51,0.10)" stroke-width="1" />`;
  }).join("");

  svg.innerHTML = `
    <rect x="0" y="0" width="${width}" height="${height}" fill="transparent"></rect>
    ${gridLines}
    <line x1="${padding}" y1="${zeroY.toFixed(2)}" x2="${width - padding}" y2="${zeroY.toFixed(2)}" stroke="#1f2933" stroke-width="1.5" stroke-dasharray="4 4"></line>
    ${bars}
  `;

  const steps = Math.min(history.length, 5);
  const labels = [];
  for (let index = 0; index < steps; index += 1) {
    const ratio = steps === 1 ? 0 : index / (steps - 1);
    labels.push(formatHour(new Date(minX + ratio * (maxX - minX)).toISOString()));
  }
  axis.innerHTML = labels.map((label) => `<span>${label}</span>`).join("");
}

function renderTsDynamicChart(series, benchmarkPayload) {
  const svg = document.getElementById("ts-dynamic-chart");
  const axis = document.getElementById("ts-dynamic-axis");
  const summary = document.getElementById("ts-dynamic-summary");
  const width = 900;
  const height = 280;
  const padding = 28;
  const rollout = series?.takagi_sugeno_ml_rollout || {};
  const hasRealtimeCurve = rollout
    && rollout.start_timestamp
    && Array.isArray(rollout.grid_minutes)
    && rollout.grid_minutes.length
    && Array.isArray(rollout.observed_glucose)
    && rollout.observed_glucose.length;
  const fallbackExample = (benchmarkPayload?.examples || []).slice(-1)[0] || null;
  const curveSource = hasRealtimeCurve
    ? {
        start_timestamp: rollout.start_timestamp,
        grid_minutes: rollout.grid_minutes,
        observed_glucose: rollout.observed_glucose,
        takagi_sugeno_glucose: rollout.takagi_sugeno_glucose || [],
        takagi_sugeno_ml_glucose: rollout.takagi_sugeno_ml_glucose || [],
      }
    : fallbackExample;

  if (!curveSource) {
    svg.innerHTML = "";
    axis.innerHTML = "";
    summary.innerHTML = `
      <div class="prototype-card">
        <span class="label">Takagi-Sugeno</span>
        <strong>Sem curva</strong>
        <p>Aguardando atualização automática do benchmark TS+ML.</p>
      </div>
    `;
    return;
  }

  const startTime = new Date(curveSource.start_timestamp).getTime();
  const times = (curveSource.grid_minutes || []).map((minute) => startTime + Number(minute) * 60 * 1000);
  const observed = (curveSource.observed_glucose || []).map(Number);
  const predicted = (curveSource.takagi_sugeno_glucose || []).map(Number);
  const predictedMl = (curveSource.takagi_sugeno_ml_glucose || []).map(Number);
  if (!times.length || !observed.length || !predicted.length) {
    svg.innerHTML = "";
    axis.innerHTML = "";
    return;
  }
  const minX = Math.min(...times);
  const maxX = Math.max(...times);
  const minY = Math.min(...observed, ...predicted, ...(predictedMl.length ? predictedMl : predicted)) - 10;
  const maxY = Math.max(...observed, ...predicted, ...(predictedMl.length ? predictedMl : predicted)) + 10;
  const gridLines = [0.2, 0.4, 0.6, 0.8].map((ratio) => {
    const y = padding + ratio * (height - padding * 2);
    return `<line x1="${padding}" y1="${y}" x2="${width - padding}" y2="${y}" stroke="rgba(31,41,51,0.10)" stroke-width="1" />`;
  }).join("");
  const observedPath = buildPath(times, observed, width, height, padding, minX, maxX, minY, maxY);
  const predictedPath = buildPath(times, predicted, width, height, padding, minX, maxX, minY, maxY);
  const predictedMlPath = predictedMl.length ? buildPath(times, predictedMl, width, height, padding, minX, maxX, minY, maxY) : "";
  const observedLabels = [
    { time: times[0], value: observed[0] },
    { time: times[times.length - 1], value: observed[observed.length - 1] },
  ];
  const predictedLabels = [
    { time: times[0], value: predicted[0] },
    { time: times[times.length - 1], value: predicted[predicted.length - 1] },
  ];
  const predictedMlLabels = predictedMl.length ? [
    { time: times[0], value: predictedMl[0] },
    { time: times[times.length - 1], value: predictedMl[predictedMl.length - 1] },
  ] : [];

  svg.innerHTML = `
    <rect x="0" y="0" width="${width}" height="${height}" fill="transparent"></rect>
    ${gridLines}
    <path d="${observedPath}" fill="none" stroke="#b45309" stroke-width="3" stroke-linecap="round"></path>
    <path d="${predictedPath}" fill="none" stroke="#7c3aed" stroke-width="3" stroke-linecap="round" stroke-dasharray="7 5"></path>
    ${predictedMl.length ? `<path d="${predictedMlPath}" fill="none" stroke="#dc2626" stroke-width="3" stroke-linecap="round" stroke-dasharray="4 4"></path>` : ""}
    ${buildValueLabels(observedLabels, "#b45309", minX, maxX, minY, maxY, width, height, padding)}
    ${buildValueLabels(predictedLabels, "#7c3aed", minX, maxX, minY, maxY, width, height, padding)}
    ${predictedMl.length ? buildValueLabels(predictedMlLabels, "#dc2626", minX, maxX, minY, maxY, width, height, padding) : ""}
  `;

  axis.innerHTML = times.map((value, index) => {
    if (index === 0 || index === times.length - 1 || index === Math.floor(times.length / 2)) {
      return `<span>${formatHour(new Date(value).toISOString())}</span>`;
    }
    return "";
  }).join("");

  const metricsSource = benchmarkPayload || {};
  summary.innerHTML = `
    <div class="prototype-card">
      <span class="label">TS one-step</span>
      <strong>${metricsSource.one_step_valid?.takagi_sugeno?.rmse_mg_dl != null ? `${formatNumber(metricsSource.one_step_valid.takagi_sugeno.rmse_mg_dl, 2)} mg/dL` : "--"}</strong>
      <p>MAE ${metricsSource.one_step_valid?.takagi_sugeno?.mae_mg_dl != null ? formatNumber(metricsSource.one_step_valid.takagi_sugeno.mae_mg_dl, 2) : "--"}</p>
    </div>
    <div class="prototype-card">
      <span class="label">TS+ML one-step</span>
      <strong>${metricsSource.one_step_valid?.takagi_sugeno_ml?.rmse_mg_dl != null ? `${formatNumber(metricsSource.one_step_valid.takagi_sugeno_ml.rmse_mg_dl, 2)} mg/dL` : "--"}</strong>
      <p>MAE ${metricsSource.one_step_valid?.takagi_sugeno_ml?.mae_mg_dl != null ? formatNumber(metricsSource.one_step_valid.takagi_sugeno_ml.mae_mg_dl, 2) : "--"}</p>
    </div>
    <div class="prototype-card">
      <span class="label">TS rollout</span>
      <strong>${metricsSource.rollout_valid?.takagi_sugeno?.rmse_mg_dl != null ? `${formatNumber(metricsSource.rollout_valid.takagi_sugeno.rmse_mg_dl, 2)} mg/dL` : "--"}</strong>
      <p>MAE ${metricsSource.rollout_valid?.takagi_sugeno?.mae_mg_dl != null ? formatNumber(metricsSource.rollout_valid.takagi_sugeno.mae_mg_dl, 2) : "--"}</p>
    </div>
    <div class="prototype-card">
      <span class="label">TS+ML rollout</span>
      <strong>${metricsSource.rollout_valid?.takagi_sugeno_ml?.rmse_mg_dl != null ? `${formatNumber(metricsSource.rollout_valid.takagi_sugeno_ml.rmse_mg_dl, 2)} mg/dL` : "--"}</strong>
      <p>MAE ${metricsSource.rollout_valid?.takagi_sugeno_ml?.mae_mg_dl != null ? formatNumber(metricsSource.rollout_valid.takagi_sugeno_ml.mae_mg_dl, 2) : "--"}</p>
    </div>
  `;
}

function updatePlot() {
  document.getElementById("pipeline-plot").src = `/outputs/plots/latest_comparison.png?ts=${Date.now()}`;
}

function estimatePkActiveFromState(state) {
  const insulin = Array.isArray(state.insulin) ? state.insulin : [];
  if (!insulin.length) return 0;
  const now = new Date(state.latest_timestamp || Date.now()).getTime();
  let active = 0;
  insulin.forEach((event) => {
    const t = new Date(event.timestamp).getTime();
    const elapsedMin = Math.max((now - t) / 60000, 0);
    const units = Number(event.units || 0);
    const insulinType = String(event.insulin_type || "desconhecido").toLowerCase();
    const tau = insulinType === "basal" ? 360 : 75;
    const remain = Math.exp(-elapsedMin / tau);
    active += units * remain;
  });
  return active;
}

function renderDoseCalculator(state) {
  const choInput = document.getElementById("dose-cho");
  const glucoseInput = document.getElementById("dose-glucose");
  const sensitivityInput = document.getElementById("dose-sensitivity");
  const target = document.getElementById("dose-summary");
  if (!choInput || !glucoseInput || !sensitivityInput || !target) return;

  const latestGlucose = Number(state.latest_glucose_mg_dl || 140);
  const scale = Number(state.params?.insulin_sensitivity_scale || 1.0);
  const adaptiveIsf = 45 * scale;
  const adaptiveSensitivity = Math.min(0.2, Math.max(0.02, 3.4 / Math.max(adaptiveIsf, 1e-6)));
  if (!choInput.dataset.initialized) {
    glucoseInput.value = String(Math.round(latestGlucose));
    sensitivityInput.value = adaptiveSensitivity.toFixed(3);
    choInput.dataset.initialized = "1";
  }

  const compute = () => {
    const choRaw = Number(choInput.value || 0);
    const cho = Math.max(0, Math.round(choRaw / 10) * 10);
    choInput.value = String(cho);
    const glucose = Number(glucoseInput.value || 140);
    const sensitivity = Number(sensitivityInput.value || 0.1);

    const choDose = sensitivity * cho;
    const correction = (glucose - 140) / 30;
    const rawDose = Math.max(0, choDose + correction);

    const pkActiveU = estimatePkActiveFromState(state);
    const pkAttenuation = Math.min(0.65, Math.max(0, pkActiveU / 6.0));
    const doseAfterPk = Math.max(0, rawDose * (1 - pkAttenuation));
    const maxBolus = 4.0;
    const finalDose = Math.min(maxBolus, doseAfterPk);

    target.innerHTML = `
      <strong>Dose sugerida: ${formatNumber(finalDose, 2)} U</strong>
      <p>CHO: ${formatNumber(choDose, 2)} U | Correção: ${formatNumber(correction, 2)} U</p>
      <p>Dose bruta: ${formatNumber(rawDose, 2)} U | Pós-PK: ${formatNumber(doseAfterPk, 2)} U | Limite: ${formatNumber(maxBolus, 1)} U</p>
      <p>IOB PK estimado: ${formatNumber(pkActiveU, 2)} U | Atenuação PK: ${formatNumber(pkAttenuation * 100, 0)}%</p>
      <p>Fórmula: dose = sensibilidade*CHO + (glicemia-140)/30</p>
    `;
  };

  if (!choInput.dataset.bound) {
    ["input", "change"].forEach((eventName) => {
      choInput.addEventListener(eventName, compute);
      glucoseInput.addEventListener(eventName, compute);
      sensitivityInput.addEventListener(eventName, compute);
    });
    choInput.dataset.bound = "1";
  }
  compute();
}

async function loadDashboard() {
  try {
    const [metrics, state, series] = await Promise.all([
      fetchJson("/api/metrics"),
      fetchJson("/api/state"),
      fetchJson("/api/series"),
    ]);

    updateHeader(metrics, state);
    updateStateSummary(state);
    renderEvents(state);
    renderChart(series);
    renderForecastDeltaChart(series);
    renderTsDynamicChart(series, metrics.takagi_sugeno_ml_model || {});
    renderDoseCalculator(state);
    updatePlot();
  } catch (error) {
    document.getElementById("generated-at").textContent = "falha ao carregar";
    document.getElementById("status-dot").classList.remove("ok");
    console.error(error);
  }
}

document.getElementById("refresh-btn").addEventListener("click", loadDashboard);
loadDashboard();
setInterval(loadDashboard, REFRESH_MS);
