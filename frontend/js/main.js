const MODEL_COLORS = {
    'Baseline': '#555568',
    'HAR-RV': '#5b8fff',
    'XGBoost': '#f5a623',
    'XGBoost-NLP': '#3ecf8e',
};

const MODEL_BADGES = {
    'Baseline': {label: 'naive', cls: 'badge-baseline'},
    'HAR-RV': {label: 'linear', cls: 'badge-good'},
    'XGBoost': {label: 'ml boosting', cls: 'badge-good'},
    'XGBoost-NLP': {label: 'ml boosting nlp', cls: 'badge-good'},
};

const LAYOUT_BASE = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: {family: "'IBM Plex Mono', monospace", color: '#8888a0', size: 10},
    margin: {t: 10, b: 40, l: 65, r: 20},
    xaxis: {
        gridcolor: 'rgba(255,255,255,0.04)',
        linecolor: 'rgba(255,255,255,0.07)',
        tickcolor: 'rgba(255,255,255,0.07)',
        tickfont: {size: 9},
    },
    yaxis: {
        gridcolor: 'rgba(255,255,255,0.04)',
        linecolor: 'rgba(255,255,255,0.07)',
        tickcolor: 'rgba(255,255,255,0.07)',
        tickfont: {size: 9},
        tickformat: '.2e',
    },
    hovermode: 'x unified',
    hoverlabel: {
        bgcolor: '#1a1a24',
        bordercolor: 'rgba(255,255,255,0.1)',
        font: {family: "'IBM Plex Mono', monospace", size: 10, color: '#e8e8f0'},
    },
};

const CONFIG = {displayModeBar: false, responsive: true};

async function init() {
    const res = await fetch('/api/results');
    const data = await res.json();
    const {period, models} = data;

    document.getElementById('period-badge').textContent =
        `${period.start} → ${period.end}`;

    renderMetrics(models);
    renderMainChart(models);
    renderErrorChart(models);
    renderBarChart(models);

    const count = Object.values(models)[0].predictions.length;
    document.getElementById('pred-count').textContent =
        `${count.toLocaleString()} trading days · ${Object.keys(models).length} models`;
}

function renderMetrics(models) {
    const grid = document.getElementById('metrics-grid');
    const entries = Object.entries(models);
    const mses = entries.map(([, m]) => m.metrics.MSE);
    const minMSE = Math.min(...mses);

    grid.innerHTML = entries.map(([name, m]) => {
        const badge = MODEL_BADGES[name] || {label: 'model', cls: 'badge-good'};
        const isBest = m.metrics.MSE === minMSE;
        const improvement = ((mses[0] - m.metrics.MSE) / mses[0] * 100).toFixed(1);
        return `
      <div class="metric-card">
        <div class="metric-label">model</div>
        <div class="metric-model" style="color:${MODEL_COLORS[name] || '#e8e8f0'}">${name}</div>
        <div class="metric-mse" style="color:${isBest ? '#3ecf8e' : '#e8e8f0'}">
          ${m.metrics.MSE.toExponential(3)}
        </div>
        <div class="metric-qlike">QLIKE ${m.metrics.QLIKE.toFixed(3)}</div>
        <div style="display:flex;gap:0.4rem;flex-wrap:wrap;margin-top:0.25rem;">
          <span class="metric-badge ${isBest ? 'badge-best' : badge.cls}">
            ${isBest ? 'best mse' : badge.label}
          </span>
          ${parseFloat(improvement) > 0
            ? `<span class="metric-badge badge-best">−${improvement}% mse</span>`
            : ''}
        </div>
      </div>`;
    }).join('');
}

function renderMainChart(models) {
    const el = document.getElementById('chart-main');
    el.style.height = '360px';
    el.className = '';
    el.innerHTML = '';

    const entries = Object.entries(models);
    const firstModel = entries[0][1];

    const traces = [];

    traces.push({
        x: firstModel.dates,
        y: firstModel.actuals,
        name: 'Actual RV',
        type: 'scatter',
        mode: 'lines',
        line: {color: 'rgba(255,255,255,0.15)', width: 1},
        fill: 'tozeroy',
        fillcolor: 'rgba(255,255,255,0.02)',
        hovertemplate: '%{y:.2e}<extra>Actual</extra>',
    });

    entries.forEach(([name, m]) => {
        traces.push({
            x: m.dates,
            y: m.predictions,
            name,
            type: 'scatter',
            mode: 'lines',
            line: {
                color: MODEL_COLORS[name] || '#888',
                width: name === 'Baseline' ? 1 : 1.5,
                dash: name === 'Baseline' ? 'dot' : 'solid',
            },
            hovertemplate: `%{y:.2e}<extra>${name}</extra>`,
        });
    });

    Plotly.newPlot(el, traces, {
        ...LAYOUT_BASE,
        margin: {t: 10, b: 50, l: 65, r: 20},
        shapes: [{
            type: 'line',
            x0: '2020-03-16', x1: '2020-03-16',
            y0: 0, y1: 1, yref: 'paper',
            line: {color: 'rgba(255,107,107,0.25)', width: 1, dash: 'dot'},
        }],
        annotations: [{
            x: '2020-03-16', y: 1, xref: 'x', yref: 'paper',
            text: 'COVID crash',
            showarrow: false,
            font: {size: 8, color: 'rgba(255,107,107,0.5)', family: "'IBM Plex Mono', monospace"},
            xanchor: 'left',
            xshift: 6,
        }],
    }, CONFIG);

    const legend = document.getElementById('main-legend');
    const items = [['Actual RV', 'rgba(255,255,255,0.3)'], ...Object.entries(MODEL_COLORS)];
    legend.innerHTML = items.map(([n, c]) => `
    <div class="legend-item">
      <div class="legend-dot" style="background:${c}"></div>
      ${n}
    </div>`).join('');
}

function renderErrorChart(models) {
    const el = document.getElementById('chart-error');
    el.style.height = '260px';
    el.className = '';
    el.innerHTML = '';

    const traces = Object.entries(models)
        .map(([name, m]) => ({
            x: m.dates,
            y: m.predictions.map((p, i) => p - m.actuals[i]),
            name,
            type: 'scatter',
            mode: 'lines',
            line: {
                color: MODEL_COLORS[name], width: name === 'Baseline' ? 1 : 1.5,
                dash: name === 'Baseline' ? 'dot' : 'solid'
            },
            hovertemplate: `%{y:.2e}<extra>${name}</extra>`,
        }));

    Plotly.newPlot(el, traces, {
        ...LAYOUT_BASE,
        yaxis: {
            ...LAYOUT_BASE.yaxis,
            title: {text: 'pred − actual', font: {size: 9}},
            zeroline: true,
            zerolinecolor: 'rgba(255,255,255,0.15)',
            zerolinewidth: 1,
        },
        showlegend: false,
    }, CONFIG);
}

function renderBarChart(models) {
    const el = document.getElementById('chart-bar');
    el.style.height = '260px';
    el.className = '';
    el.innerHTML = '';

    const entries = Object.entries(models);

    const ys = entries.map(([, m]) => m.metrics.MSE);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const padding = (maxY - minY) * 0.3;
    const range = [minY * 0.99, maxY * 1.01];

    Plotly.newPlot(el, [{
        x: entries.map(([n]) => n),
        y: ys,
        type: 'bar',
        marker: {
            color: entries.map(([n]) => MODEL_COLORS[n] || '#888'),
            opacity: 0.85,
        },
        hovertemplate: '%{y:.3e}<extra></extra>',
    }], {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: {family: "'IBM Plex Mono', monospace", color: '#8888a0', size: 10},
        margin: {t: 10, b: 40, l: 65, r: 20},
        yaxis: {
            gridcolor: 'rgba(255,255,255,0.04)',
            tickformat: '.2e',
            title: {text: 'MSE', font: {size: 9}},
            range: [minY * 0.99, maxY * 1.01],
        },
        xaxis: {
            gridcolor: 'rgba(255,255,255,0.04)',
            tickfont: {size: 9},
        },
        showlegend: false,
        bargap: 0.35,
    }, CONFIG);
}

init().catch(err => {
    document.body.innerHTML = `
    <div class="error-screen">
      error: ${err.message} — run python run_pipeline.py first
    </div>`;
});