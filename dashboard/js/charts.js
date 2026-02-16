const ChartConfigs = {
  default: {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'bottom',
        labels: {
          padding: 20,
          usePointStyle: true
        }
      }
    }
  },

  bar: {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false }
    },
    scales: {
      y: {
        beginAtZero: true,
        grid: {
          color: 'rgba(0,0,0,0.05)'
        }
      },
      x: {
        grid: { display: false }
      }
    }
  },

  pie: {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'right',
        labels: {
          padding: 15,
          usePointStyle: true
        }
      }
    }
  },

  line: {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false }
    },
    scales: {
      y: {
        beginAtZero: true,
        grid: {
          color: 'rgba(0,0,0,0.05)'
        }
      },
      x: {
        grid: { display: false }
      }
    },
    elements: {
      line: {
        tension: 0.4
      },
      point: {
        radius: 4,
        hoverRadius: 6
      }
    }
  },

  radar: {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'bottom'
      }
    },
    scales: {
      r: {
        beginAtZero: true,
        grid: {
          color: 'rgba(0,0,0,0.1)'
        }
      }
    }
  },

  doughnut: {
    responsive: true,
    maintainAspectRatio: false,
    cutout: '60%',
    plugins: {
      legend: {
        position: 'right',
        labels: {
          padding: 15,
          usePointStyle: true
        }
      }
    }
  }
};

const ChartColors = [
  '#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6',
  '#06b6d4', '#ec4899', '#14b8a6', '#f97316', '#6366f1'
];

function createYearDistributionChart(canvasId, data) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return null;

  return new Chart(ctx, {
    type: 'bar',
    data: {
      labels: Object.keys(data),
      datasets: [{
        label: '论文数',
        data: Object.values(data),
        backgroundColor: ChartColors[0],
        borderRadius: 6
      }]
    },
    options: ChartConfigs.bar
  });
}

function createFieldDistributionChart(canvasId, papers) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return null;

  const fieldCounts = {};
  papers.forEach(p => {
    fieldCounts[p.field] = (fieldCounts[p.field] || 0) + 1;
  });

  return new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: Object.keys(fieldCounts),
      datasets: [{
        data: Object.values(fieldCounts),
        backgroundColor: Object.keys(fieldCounts).map(f => getFieldColor(f))
      }]
    },
    options: ChartConfigs.doughnut
  });
}

function createCitationsTrendChart(canvasId, data) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return null;

  return new Chart(ctx, {
    type: 'line',
    data: {
      labels: Object.keys(data),
      datasets: [{
        label: '引用数',
        data: Object.values(data),
        borderColor: ChartColors[0],
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        fill: true
      }]
    },
    options: ChartConfigs.line
  });
}

function createJournalDistributionChart(canvasId, papers) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return null;

  const journalCounts = {};
  papers.forEach(p => {
    journalCounts[p.journal] = (journalCounts[p.journal] || 0) + 1;
  });

  const sorted = Object.entries(journalCounts).sort((a, b) => b[1] - a[1]).slice(0, 10);

  return new Chart(ctx, {
    type: 'bar',
    data: {
      labels: sorted.map(s => s[0]),
      datasets: [{
        label: '论文数',
        data: sorted.map(s => s[1]),
        backgroundColor: ChartColors,
        borderRadius: 6
      }]
    },
    options: {
      ...ChartConfigs.bar,
      indexAxis: 'y'
    }
  });
}

function createComparisonRadarChart(canvasId, data) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return null;

  const labels = data.map(d => d.name);
  const papersData = data.map(d => d.papers);
  const citationsData = data.map(d => d.citations / 100);
  const hIndexData = data.map(d => d.hIndex);

  return new Chart(ctx, {
    type: 'radar',
    data: {
      labels: labels,
      datasets: [
        {
          label: '论文数',
          data: papersData,
          borderColor: ChartColors[0],
          backgroundColor: 'rgba(59, 130, 246, 0.2)'
        },
        {
          label: '引用数(×100)',
          data: citationsData,
          borderColor: ChartColors[1],
          backgroundColor: 'rgba(16, 185, 129, 0.2)'
        },
        {
          label: 'H-index',
          data: hIndexData,
          borderColor: ChartColors[2],
          backgroundColor: 'rgba(245, 158, 11, 0.2)'
        }
      ]
    },
    options: ChartConfigs.radar
  });
}

function createCollaborationChart(canvasId, data) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return null;

  return new Chart(ctx, {
    type: 'bar',
    data: {
      labels: data.map(c => c.name),
      datasets: [{
        label: '合作次数',
        data: data.map(c => c.collaborations),
        backgroundColor: ChartColors,
        borderRadius: 6
      }]
    },
    options: {
      ...ChartConfigs.bar,
      indexAxis: 'y'
    }
  });
}
