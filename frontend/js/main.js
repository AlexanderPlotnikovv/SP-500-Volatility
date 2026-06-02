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
    margin: {t: 10, b: 55, l: 80, r: 20},
    xaxis: {
        gridcolor: 'rgba(255,255,255,0.04)',
        linecolor: 'rgba(255,255,255,0.07)',
        tickcolor: 'rgba(255,255,255,0.07)',
        tickfont: {size: 9},
        title: {text: 'Date', font: {size: 9, color: '#555568'}},
    },
    yaxis: {
        gridcolor: 'rgba(255,255,255,0.04)',
        linecolor: 'rgba(255,255,255,0.07)',
        tickcolor: 'rgba(255,255,255,0.07)',
        tickfont: {size: 9},
        tickformat: '.2e',
        title: {text: 'RV = r²', font: {size: 9, color: '#555568'}},
    },
    hovermode: 'x unified',
    hoverlabel: {
        bgcolor: '#1a1a24',
        bordercolor: 'rgba(255,255,255,0.1)',
        font: {family: "'IBM Plex Mono', monospace", size: 10, color: '#e8e8f0'},
    },
};

const CONFIG = {
    displayModeBar: true,
    responsive: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['select2d', 'lasso2d', 'autoScale2d', 'toImage'],
};

let GLOBAL_DATA = null;

async function init() {
    const res = await fetch('/api/results');
    GLOBAL_DATA = await res.json();
    const {period, models} = GLOBAL_DATA;

    document.getElementById('period-badge').textContent =
        `${period.start} → ${period.end}`;

    const fromInput = document.getElementById('date-from');
    const toInput = document.getElementById('date-to');

    fromInput.value = period.start;
    toInput.value = period.end;
    fromInput.min = period.start;
    fromInput.max = period.end;
    toInput.min = period.start;
    toInput.max = period.end;

    // attach listeners (reliable, not inline)
    fromInput.addEventListener('change', renderAll);
    toInput.addEventListener('change', renderAll);

    document.getElementById('reset-btn').addEventListener('click', () => {
        fromInput.value = period.start;
        toInput.value = period.end;
        renderAll();
    });

    renderAll();

    const count = Object.values(models)[0].predictions.length;
    document.getElementById('pred-count').textContent =
        `${count.toLocaleString()} trading days · ${Object.keys(models).length} models`;
}

function getFiltered() {
    const from = document.getElementById('date-from').value;
    const to = document.getElementById('date-to').value;
    const models = GLOBAL_DATA.models;

    const result = {};
    for (const [name, m] of Object.entries(models)) {
        const idx = [];
        for (let i = 0; i < m.dates.length; i++) {
            if (m.dates[i] >= from && m.dates[i] <= to) idx.push(i);
        }
        result[name] = {
            ...m,
            dates: idx.map(i => m.dates[i]),
            predictions: idx.map(i => m.predictions[i]),
            actuals: idx.map(i => m.actuals[i]),
        };
    }
    return result;
}

function renderAll() {
    const filtered = getFiltered();

    // update hint
    const from = document.getElementById('date-from').value;
    const to = document.getElementById('date-to').value;
    const n = Object.values(filtered)[0].dates.length;
    document.getElementById('range-hint').textContent =
        `${n.toLocaleString()} days selected`;

    renderMetrics(filtered);
    renderMainChart(filtered);
    renderErrorChart(filtered);
    renderBarChart(filtered);
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
    el.style.height = '380px';
    el.className = '';
    el.innerHTML = '';

    const entries = Object.entries(models);
    const firstModel = entries[0][1];
    const traces = [];

    // clip top 1% of actuals → always informative outside COVID spike
    const sorted = firstModel.actuals.slice().sort((a, b) => a - b);
    const p99 = sorted[Math.floor(sorted.length * 0.99)];
    const yRange = [0, p99 * 1.15];

    traces.push({
        x: firstModel.dates,
        y: firstModel.actuals,
        name: 'Actual RV',
        type: 'scatter',
        mode: 'lines',
        line: {color: 'rgba(255,255,255,0.18)', width: 1},
        fill: 'tozeroy',
        fillcolor: 'rgba(255,255,255,0.025)',
        hovertemplate: '<b>%{x}</b><br>Actual RV: %{y:.4e}<extra></extra>',
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
                width: name === 'Baseline' ? 1 : 1.6,
                dash: name === 'Baseline' ? 'dot' : 'solid',
            },
            hovertemplate: `<b>%{x}</b><br>${name}: %{y:.4e}<extra></extra>`,
        });
    });

    Plotly.react(el, traces, {
        ...LAYOUT_BASE,
        yaxis: {...LAYOUT_BASE.yaxis, range: yRange},
        legend: {
            bgcolor: 'rgba(17,17,24,0.85)',
            bordercolor: 'rgba(255,255,255,0.07)',
            borderwidth: 1,
            font: {size: 9},
            x: 0.01, y: 0.99,
            xanchor: 'left', yanchor: 'top',
        },
        shapes: [{
            type: 'line',
            x0: '2020-03-16', x1: '2020-03-16',
            y0: 0, y1: 1, yref: 'paper',
            line: {color: 'rgba(255,107,107,0.25)', width: 1, dash: 'dot'},
        }],
        annotations: [{
            x: '2020-03-16', y: 0.97, xref: 'x', yref: 'paper',
            text: 'COVID-19',
            showarrow: false,
            font: {
                size: 8, color: 'rgba(255,107,107,0.6)',
                family: "'IBM Plex Mono', monospace"
            },
            xanchor: 'left', xshift: 5,
        }],
    }, CONFIG);

    const legend = document.getElementById('main-legend');
    const items = [
        ['Actual RV', 'rgba(255,255,255,0.3)'],
        ...Object.entries(MODEL_COLORS),
    ];
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

    const traces = Object.entries(models).map(([name, m]) => ({
        x: m.dates,
        y: m.predictions.map((p, i) => p - m.actuals[i]),
        name,
        type: 'scatter',
        mode: 'lines',
        line: {
            color: MODEL_COLORS[name],
            width: name === 'Baseline' ? 1 : 1.5,
            dash: name === 'Baseline' ? 'dot' : 'solid',
        },
        hovertemplate: `<b>%{x}</b><br>${name}: %{y:.4e}<extra></extra>`,
    }));

    Plotly.react(el, traces, {
        ...LAYOUT_BASE,
        yaxis: {
            ...LAYOUT_BASE.yaxis,
            title: {text: 'pred − actual', font: {size: 9, color: '#555568'}},
            zeroline: true,
            zerolinecolor: 'rgba(255,255,255,0.2)',
            zerolinewidth: 1,
        },
        legend: {bgcolor: 'transparent', font: {size: 9}},
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
    const minIdx = ys.indexOf(minY);

    // best model full color, others muted
    const colors = entries.map(([n], i) =>
        i === minIdx ? MODEL_COLORS[n] : MODEL_COLORS[n] + '88'
    );

    Plotly.react(el, [{
        x: entries.map(([n]) => n),
        y: ys,
        type: 'bar',
        width: 0.55,
        marker: {
            color: colors,
            cornerradius: 6,
            line: {width: 0},
        },
        hovertemplate: '<b>%{x}</b><br>MSE: %{y:.4e}<extra></extra>',
    }], {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: {family: "'IBM Plex Mono', monospace", color: '#8888a0', size: 10},
        margin: {t: 25, b: 45, l: 80, r: 20},
        yaxis: {
            gridcolor: 'rgba(255,255,255,0.04)',
            tickformat: '.2e',
            title: {text: 'MSE ↓ lower is better', font: {size: 9, color: '#555568'}},
            range: [minY * 0.985, maxY * 1.02],
            zeroline: false,
        },
        xaxis: {
            linecolor: 'rgba(255,255,255,0.07)',
            tickfont: {size: 9},
        },
        showlegend: false,
        annotations: entries.map(([, m], i) => ({
            x: i,
            y: m.metrics.MSE,
            text: m.metrics.MSE.toExponential(2),
            showarrow: false,
            yanchor: 'bottom',
            yshift: 5,
            font: {
                size: 8.5, color: i === minIdx ? '#3ecf8e' : '#8888a0',
                family: "'IBM Plex Mono', monospace"
            },
        })),
    }, CONFIG);
}

init().catch(err => {
    document.body.innerHTML = `
    <div class="error-screen">
      error: ${err.message} — run python run_pipeline.py first
    </div>`;
});