const Utils = {
  formatNumber(num) {
    if (num >= 1000) {
      return (num / 1000).toFixed(1) + 'k';
    }
    return num.toString();
  },

  debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
      const later = () => {
        clearTimeout(timeout);
        func(...args);
      };
      clearTimeout(timeout);
      timeout = setTimeout(later, wait);
    };
  },

  exportToCSV(data, filename) {
    const csv = data.map(row => row.join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = filename;
    link.click();
  },

  exportToJSON(data, filename) {
    const json = JSON.stringify(data, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = filename;
    link.click();
  },

  exportChartAsImage(chartId, filename) {
    const canvas = document.getElementById(chartId);
    if (canvas) {
      const link = document.createElement('a');
      link.download = filename + '.png';
      link.href = canvas.toDataURL();
      link.click();
    }
  }
};

const ThemeManager = {
  init() {
    const savedTheme = localStorage.getItem('theme') || 'light';
    this.setTheme(savedTheme);
    
    document.querySelector('.theme-toggle')?.addEventListener('click', () => {
      const current = document.documentElement.getAttribute('data-theme');
      this.setTheme(current === 'dark' ? 'light' : 'dark');
    });
  },

  setTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
    
    const toggle = document.querySelector('.theme-toggle');
    if (toggle) {
      toggle.textContent = theme === 'dark' ? '☀️' : '🌙';
    }
  }
};

const FilterManager = {
  filters: {
    year: 'all',
    field: 'all',
    method: 'all'
  },

  init() {
    document.querySelectorAll('.filter-group select').forEach(select => {
      select.addEventListener('change', (e) => {
        this.filters[e.target.name] = e.target.value;
        this.applyFilters();
      });
    });
  },

  applyFilters() {
    window.dispatchEvent(new CustomEvent('filtersChanged', { detail: this.filters }));
  },

  getFilters() {
    return this.filters;
  },

  filterPapers(papers) {
    return papers.filter(paper => {
      if (this.filters.year !== 'all') {
        const yearRange = this.filters.year.split('-').map(Number);
        if (yearRange.length === 2) {
          if (paper.year < yearRange[0] || paper.year > yearRange[1]) return false;
        } else {
          if (paper.year !== yearRange[0]) return false;
        }
      }
      if (this.filters.field !== 'all' && paper.field !== this.filters.field) return false;
      return true;
    });
  }
};

const SearchManager = {
  init() {
    const searchInput = document.querySelector('.search-box input');
    if (searchInput) {
      searchInput.addEventListener('input', Utils.debounce((e) => {
        this.search(e.target.value);
      }, 300));
    }
  },

  search(query) {
    window.dispatchEvent(new CustomEvent('search', { detail: query }));
  }
};

const ExportManager = {
  init() {
    document.querySelectorAll('.export-dropdown').forEach(dropdown => {
      const btn = dropdown.querySelector('.btn');
      const menu = dropdown.querySelector('.export-menu');
      
      btn?.addEventListener('click', (e) => {
        e.stopPropagation();
        menu.classList.toggle('active');
      });
    });

    document.addEventListener('click', () => {
      document.querySelectorAll('.export-menu').forEach(menu => {
        menu.classList.remove('active');
      });
    });

    document.querySelectorAll('[data-export]').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const type = e.target.dataset.export;
        this.export(type);
      });
    });
  },

  export(type) {
    window.dispatchEvent(new CustomEvent('export', { detail: type }));
  }
};

function initCommon() {
  ThemeManager.init();
  FilterManager.init();
  SearchManager.init();
  ExportManager.init();

  document.querySelectorAll('.sidebar-nav a').forEach(link => {
    if (link.getAttribute('href') === location.pathname.split('/').pop()) {
      link.classList.add('active');
    }
  });
}

function createStatCard(title, value, change, icon, iconBg) {
  return `
    <div class="stat-card">
      <div class="stat-card-header">
        <span class="stat-card-title">${title}</span>
        <div class="stat-card-icon" style="background: ${iconBg}">${icon}</div>
      </div>
      <div class="stat-card-value">${value}</div>
      <div class="stat-card-change ${change >= 0 ? 'positive' : 'negative'}">
        ${change >= 0 ? '↑' : '↓'} ${Math.abs(change)}% 较去年
      </div>
    </div>
  `;
}

function createPaperRow(paper) {
  return `
    <tr>
      <td>${paper.title}</td>
      <td>${paper.year}</td>
      <td><span class="tag tag-blue">${paper.field}</span></td>
      <td>${paper.journal}</td>
      <td>${paper.citations}</td>
      <td>${paper.impact}</td>
    </tr>
  `;
}

document.addEventListener('DOMContentLoaded', initCommon);
