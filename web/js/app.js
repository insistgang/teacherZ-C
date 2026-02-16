const Utils = {
  $(selector) {
    return document.querySelector(selector);
  },
  $$(selector) {
    return document.querySelectorAll(selector);
  },
  debounce(fn, delay) {
    let timer;
    return (...args) => {
      clearTimeout(timer);
      timer = setTimeout(() => fn.apply(this, args), delay);
    };
  },
  formatNumber(num) {
    if (num >= 1000) return (num / 1000).toFixed(1) + 'k';
    return num.toString();
  },
  getDomainColor(domainId) {
    const domain = RESEARCH_DATA.domains.find(d => d.id === domainId);
    return domain ? domain.color : '#64748b';
  },
  getMethodColor(methodId) {
    const method = RESEARCH_DATA.methods.find(m => m.id === methodId);
    if (!method) return '#64748b';
    const cat = METHOD_CATEGORIES[method.category];
    return cat ? cat.color : '#64748b';
  }
};

class Navigation {
  constructor() {
    this.currentPage = window.location.pathname.split('/').pop() || 'index.html';
    this.highlightCurrent();
  }
  highlightCurrent() {
    Utils.$$('.nav-links a').forEach(link => {
      const href = link.getAttribute('href');
      if (href === this.currentPage) {
        link.classList.add('active');
      }
    });
  }
}

class Modal {
  constructor() {
    this.modal = Utils.$('#paper-modal');
    if (this.modal) {
      this.modal.querySelector('.modal-close').addEventListener('click', () => this.close());
      this.modal.addEventListener('click', (e) => {
        if (e.target === this.modal) this.close();
      });
    }
  }
  open(content) {
    if (!this.modal) return;
    this.modal.querySelector('.modal-body').innerHTML = content;
    this.modal.classList.add('active');
    document.body.style.overflow = 'hidden';
  }
  close() {
    if (!this.modal) return;
    this.modal.classList.remove('active');
    document.body.style.overflow = '';
  }
}

class PapersFilter {
  constructor() {
    this.papers = [...RESEARCH_DATA.papers];
    this.filteredPapers = [...this.papers];
    this.filters = { year: '', domain: '', method: '', search: '' };
    this.init();
  }
  init() {
    const yearSelect = Utils.$('#filter-year');
    const domainSelect = Utils.$('#filter-domain');
    const methodSelect = Utils.$('#filter-method');
    const searchInput = Utils.$('#search-input');
    if (yearSelect) {
      const years = [...new Set(this.papers.map(p => p.year))].sort((a, b) => b - a);
      years.forEach(year => {
        const opt = document.createElement('option');
        opt.value = year;
        opt.textContent = year;
        yearSelect.appendChild(opt);
      });
      yearSelect.addEventListener('change', (e) => {
        this.filters.year = e.target.value;
        this.applyFilters();
      });
    }
    if (domainSelect) {
      RESEARCH_DATA.domains.forEach(domain => {
        const opt = document.createElement('option');
        opt.value = domain.id;
        opt.textContent = domain.name;
        domainSelect.appendChild(opt);
      });
      domainSelect.addEventListener('change', (e) => {
        this.filters.domain = e.target.value;
        this.applyFilters();
      });
    }
    if (methodSelect) {
      RESEARCH_DATA.methods.forEach(method => {
        const opt = document.createElement('option');
        opt.value = method.id;
        opt.textContent = method.name;
        methodSelect.appendChild(opt);
      });
      methodSelect.addEventListener('change', (e) => {
        this.filters.method = e.target.value;
        this.applyFilters();
      });
    }
    if (searchInput) {
      searchInput.addEventListener('input', Utils.debounce((e) => {
        this.filters.search = e.target.value.toLowerCase();
        this.applyFilters();
      }, 300));
    }
    this.render();
  }
  applyFilters() {
    this.filteredPapers = this.papers.filter(paper => {
      if (this.filters.year && paper.year.toString() !== this.filters.year) return false;
      if (this.filters.domain && paper.domain !== this.filters.domain) return false;
      if (this.filters.method && paper.method !== this.filters.method) return false;
      if (this.filters.search) {
        const searchStr = `${paper.title} ${paper.venue}`.toLowerCase();
        if (!searchStr.includes(this.filters.search)) return false;
      }
      return true;
    });
    this.render();
  }
  render() {
    const container = Utils.$('#papers-container');
    if (!container) return;
    container.innerHTML = this.filteredPapers.map(paper => this.renderPaperCard(paper)).join('');
    Utils.$$('.paper-card').forEach(card => {
      card.addEventListener('click', () => {
        const paperId = card.dataset.id;
        this.showPaperDetail(paperId);
      });
    });
  }
  renderPaperCard(paper) {
    const domain = RESEARCH_DATA.domains.find(d => d.id === paper.domain);
    const domainColor = domain ? domain.color : '#64748b';
    const arxivLink = paper.arxiv ? `<a href="https://arxiv.org/abs/${paper.arxiv}" target="_blank" class="arxiv-link" onclick="event.stopPropagation()">arXiv</a>` : '';
    return `
      <div class="paper-card" data-id="${paper.id}" style="border-left-color: ${domainColor}">
        <h3>${paper.title}</h3>
        <div class="meta">
          <span class="year">${paper.year}</span>
          <span class="venue">${paper.venue}</span>
          ${arxivLink}
        </div>
        <span class="domain-tag" style="background: ${domainColor}20; color: ${domainColor}">${domain ? domain.name : 'Other'}</span>
      </div>
    `;
  }
  showPaperDetail(paperId) {
    const paper = this.papers.find(p => p.id === paperId);
    if (!paper) return;
    const domain = RESEARCH_DATA.domains.find(d => d.id === paper.domain);
    const method = RESEARCH_DATA.methods.find(m => m.id === paper.method);
    const content = `
      <div class="paper-detail">
        <h2>${paper.title}</h2>
        <div class="meta-row">
          <div class="meta-item"><span>📅</span> ${paper.year}</div>
          <div class="meta-item"><span>📚</span> ${paper.venue}</div>
          <div class="meta-item"><span>📝</span> ${paper.citations} citations</div>
        </div>
        ${paper.arxiv ? `<p><a href="https://arxiv.org/abs/${paper.arxiv}" target="_blank">View on arXiv: ${paper.arxiv}</a></p>` : ''}
        <div style="margin-top: 20px;">
          <h4>Domain</h4>
          <p style="color: ${domain ? domain.color : '#64748b'}">${domain ? domain.name : 'Other'}</p>
        </div>
        ${method ? `
        <div style="margin-top: 16px;">
          <h4>Method</h4>
          <p><strong>${method.fullName}</strong> (${method.category})</p>
          <p style="color: var(--text-light); margin-top: 8px;">${method.description}</p>
        </div>
        ` : ''}
      </div>
    `;
    new Modal().open(content);
  }
}

class CitationNetwork {
  constructor() {
    this.width = 0;
    this.height = 600;
    this.svg = null;
    this.simulation = null;
  }
  init() {
    const container = Utils.$('#network-svg');
    if (!container) return;
    this.width = container.clientWidth || 800;
    this.height = 600;
    this.svg = d3.select('#network-svg')
      .append('svg')
      .attr('width', '100%')
      .attr('height', this.height)
      .attr('viewBox', `0 0 ${this.width} ${this.height}`);
    this.render();
  }
  render() {
    const data = RESEARCH_DATA.citationNetwork;
    const colorScale = d3.scaleOrdinal()
      .domain([1, 2, 3, 4, 5])
      .range(['#2196F3', '#4CAF50', '#9C27B0', '#FF9800', '#F44336']);
    this.simulation = d3.forceSimulation(data.nodes)
      .force('link', d3.forceLink(data.links).id(d => d.id).distance(120))
      .force('charge', d3.forceManyBody().strength(-400))
      .force('center', d3.forceCenter(this.width / 2, this.height / 2))
      .force('collision', d3.forceCollide().radius(50));
    const link = this.svg.append('g')
      .selectAll('line')
      .data(data.links)
      .enter().append('line')
      .attr('stroke', '#cbd5e1')
      .attr('stroke-width', 2)
      .attr('marker-end', 'url(#arrowhead)');
    this.svg.append('defs').append('marker')
      .attr('id', 'arrowhead')
      .attr('viewBox', '-0 -5 10 10')
      .attr('refX', 20)
      .attr('refY', 0)
      .attr('orient', 'auto')
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .append('path')
      .attr('d', 'M 0,-5 L 10,0 L 0,5')
      .attr('fill', '#94a3b8');
    const node = this.svg.append('g')
      .selectAll('g')
      .data(data.nodes)
      .enter().append('g')
      .call(d3.drag()
        .on('start', (event, d) => {
          if (!event.active) this.simulation.alphaTarget(0.3).restart();
          d.fx = d.x;
          d.fy = d.y;
        })
        .on('drag', (event, d) => {
          d.fx = event.x;
          d.fy = event.y;
        })
        .on('end', (event, d) => {
          if (!event.active) this.simulation.alphaTarget(0);
          d.fx = null;
          d.fy = null;
        }));
    node.append('rect')
      .attr('width', 80)
      .attr('height', 40)
      .attr('x', -40)
      .attr('y', -20)
      .attr('rx', 6)
      .attr('fill', d => colorScale(d.group))
      .attr('stroke', '#fff')
      .attr('stroke-width', 2);
    node.append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', '0.35em')
      .attr('fill', '#fff')
      .attr('font-size', '10px')
      .attr('font-weight', '600')
      .text(d => d.label.split('\n')[0]);
    node.append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', '1.5em')
      .attr('fill', '#fff')
      .attr('font-size', '9px')
      .text(d => d.label.split('\n')[1] || '');
    node.append('title')
      .text(d => d.label);
    this.simulation.on('tick', () => {
      link
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y);
      node.attr('transform', d => `translate(${d.x},${d.y})`);
    });
    const zoom = d3.zoom()
      .scaleExtent([0.5, 3])
      .on('zoom', (event) => {
        this.svg.select('g').attr('transform', event.transform);
      });
    this.svg.call(zoom);
  }
}

class MethodsTimeline {
  constructor() {
    this.methods = RESEARCH_DATA.methods.sort((a, b) => a.year - b.year);
  }
  init() {
    const container = Utils.$('#methods-timeline');
    if (!container) return;
    container.innerHTML = this.methods.map(method => this.renderMethod(method)).join('');
  }
  renderMethod(method) {
    const cat = METHOD_CATEGORIES[method.category];
    const color = cat ? cat.color : '#64748b';
    const icon = cat ? cat.icon : '?';
    return `
      <div class="method-node">
        <div class="method-icon" style="background: ${color}">${icon}</div>
        <div class="method-info">
          <h3>${method.fullName} (${method.name})</h3>
          <p>${method.description}</p>
          <div class="method-tags">
            <span class="badge" style="background: ${color}20; color: ${color}">${method.category}</span>
            <span class="badge badge-primary">Since ${method.year}</span>
          </div>
        </div>
      </div>
    `;
  }
}

class SLATDemo {
  constructor() {
    this.canvas = null;
    this.ctx = null;
    this.params = {
      lambda: 2.0,
      iterations: 50,
      k: 4
    };
    this.imageData = null;
    this.noisyData = null;
    this.smoothedData = null;
    this.liftedData = null;
    this.segmentedData = null;
    this.stage = 0;
  }
  init() {
    this.canvas = Utils.$('#demo-canvas');
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext('2d');
    this.canvas.width = this.canvas.clientWidth;
    this.canvas.height = this.canvas.clientHeight;
    this.setupControls();
    this.generateSyntheticImage();
    this.render();
  }
  setupControls() {
    const lambdaSlider = Utils.$('#lambda-slider');
    const iterSlider = Utils.$('#iter-slider');
    const kSlider = Utils.$('#k-slider');
    if (lambdaSlider) {
      lambdaSlider.addEventListener('input', (e) => {
        this.params.lambda = parseFloat(e.target.value);
        Utils.$('#lambda-value').textContent = this.params.lambda.toFixed(1);
      });
    }
    if (iterSlider) {
      iterSlider.addEventListener('input', (e) => {
        this.params.iterations = parseInt(e.target.value);
        Utils.$('#iter-value').textContent = this.params.iterations;
      });
    }
    if (kSlider) {
      kSlider.addEventListener('input', (e) => {
        this.params.k = parseInt(e.target.value);
        Utils.$('#k-value').textContent = this.params.k;
      });
    }
    Utils.$('#btn-run')?.addEventListener('click', () => this.runDemo());
    Utils.$('#btn-reset')?.addEventListener('click', () => this.reset());
  }
  generateSyntheticImage() {
    const w = this.canvas.width;
    const h = this.canvas.height;
    this.imageData = this.ctx.createImageData(w, h);
    const regions = [
      { x: 0, y: 0, w: w/2, h: h/2, color: [220, 60, 60] },
      { x: w/2, y: 0, w: w/2, h: h/2, color: [60, 180, 60] },
      { x: 0, y: h/2, w: w/2, h: h/2, color: [60, 60, 220] },
      { x: w/2, y: h/2, w: w/2, h: h/2, color: [220, 180, 60] }
    ];
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = (y * w + x) * 4;
        for (const r of regions) {
          if (x >= r.x && x < r.x + r.w && y >= r.y && y < r.y + r.h) {
            this.imageData.data[i] = r.color[0];
            this.imageData.data[i + 1] = r.color[1];
            this.imageData.data[i + 2] = r.color[2];
            this.imageData.data[i + 3] = 255;
            break;
          }
        }
      }
    }
    this.noisyData = this.addNoise(this.imageData, 30);
  }
  addNoise(data, sigma) {
    const noisy = new ImageData(data.width, data.height);
    for (let i = 0; i < data.data.length; i += 4) {
      noisy.data[i] = Math.min(255, Math.max(0, data.data[i] + (Math.random() - 0.5) * 2 * sigma));
      noisy.data[i + 1] = Math.min(255, Math.max(0, data.data[i + 1] + (Math.random() - 0.5) * 2 * sigma));
      noisy.data[i + 2] = Math.min(255, Math.max(0, data.data[i + 2] + (Math.random() - 0.5) * 2 * sigma));
      noisy.data[i + 3] = 255;
    }
    return noisy;
  }
  rofDenoise(data, lambda, iterations) {
    const w = data.width;
    const h = data.height;
    const result = new ImageData(w, h);
    for (let c = 0; c < 3; c++) {
      const channel = new Float32Array(w * h);
      const p = new Float32Array(w * h * 2);
      for (let i = 0; i < w * h; i++) {
        channel[i] = data.data[i * 4 + c];
      }
      const tau = 0.25;
      for (let iter = 0; iter < iterations; iter++) {
        for (let y = 0; y < h; y++) {
          for (let x = 0; x < w; x++) {
            const idx = y * w + x;
            let divPx = 0, divPy = 0;
            if (y > 0) divPx += p[(y * w + x) * 2] - p[((y - 1) * w + x) * 2];
            if (x > 0) divPy += p[(y * w + x) * 2 + 1] - p[(y * w + x - 1) * 2 + 1];
            const u = channel[idx] - lambda * (divPx + divPy);
            let gradX = 0, gradY = 0;
            if (x < w - 1) gradX = u - (channel[idx] - lambda * 0);
            if (y < h - 1) gradY = u - (channel[idx] - lambda * 0);
            const idx2 = (y * w + x) * 2;
            p[idx2] = (p[idx2] + tau * gradX) / (1 + tau);
            p[idx2 + 1] = (p[idx2 + 1] + tau * gradY) / (1 + tau);
            channel[idx] = u;
          }
        }
      }
      for (let i = 0; i < w * h; i++) {
        result.data[i * 4 + c] = Math.min(255, Math.max(0, channel[i]));
      }
    }
    for (let i = 0; i < w * h; i++) {
      result.data[i * 4 + 3] = 255;
    }
    return result;
  }
  kMeansSegment(data, k) {
    const w = data.width;
    const h = data.height;
    const pixels = [];
    for (let i = 0; i < data.data.length; i += 4) {
      pixels.push([data.data[i], data.data[i + 1], data.data[i + 2]]);
    }
    const centers = [];
    for (let i = 0; i < k; i++) {
      centers.push(pixels[Math.floor(Math.random() * pixels.length)].slice());
    }
    const labels = new Array(pixels.length);
    for (let iter = 0; iter < 20; iter++) {
      for (let i = 0; i < pixels.length; i++) {
        let minDist = Infinity, minIdx = 0;
        for (let j = 0; j < k; j++) {
          const d = this.colorDist(pixels[i], centers[j]);
          if (d < minDist) { minDist = d; minIdx = j; }
        }
        labels[i] = minIdx;
      }
      const sums = Array(k).fill(null).map(() => [0, 0, 0]);
      const counts = Array(k).fill(0);
      for (let i = 0; i < pixels.length; i++) {
        const l = labels[i];
        sums[l][0] += pixels[i][0];
        sums[l][1] += pixels[i][1];
        sums[l][2] += pixels[i][2];
        counts[l]++;
      }
      for (let j = 0; j < k; j++) {
        if (counts[j] > 0) {
          centers[j][0] = sums[j][0] / counts[j];
          centers[j][1] = sums[j][1] / counts[j];
          centers[j][2] = sums[j][2] / counts[j];
        }
      }
    }
    const colors = [
      [231, 76, 60], [46, 204, 113], [52, 152, 219],
      [155, 89, 182], [241, 196, 15], [26, 188, 156],
      [230, 126, 34], [149, 165, 166]
    ];
    const result = new ImageData(w, h);
    for (let i = 0; i < pixels.length; i++) {
      const c = colors[labels[i] % colors.length];
      result.data[i * 4] = c[0];
      result.data[i * 4 + 1] = c[1];
      result.data[i * 4 + 2] = c[2];
      result.data[i * 4 + 3] = 255;
    }
    return result;
  }
  colorDist(a, b) {
    return Math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2);
  }
  async runDemo() {
    this.stage = 1;
    this.render();
    await this.delay(300);
    this.smoothedData = this.rofDenoise(this.noisyData, this.params.lambda, this.params.iterations);
    this.stage = 2;
    this.render();
    await this.delay(300);
    this.segmentedData = this.kMeansSegment(this.smoothedData, this.params.k);
    this.stage = 3;
    this.render();
    this.updateResults();
  }
  delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
  reset() {
    this.stage = 0;
    this.smoothedData = null;
    this.segmentedData = null;
    this.generateSyntheticImage();
    this.render();
    Utils.$('.demo-results')?.remove();
  }
  render() {
    if (!this.ctx) return;
    this.ctx.fillStyle = '#1e293b';
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    let data = this.noisyData;
    if (this.stage === 0) {
      this.ctx.fillStyle = '#94a3b8';
      this.ctx.font = '16px Inter, sans-serif';
      this.ctx.textAlign = 'center';
      this.ctx.fillText('Click "Run SLaT" to start', this.canvas.width / 2, this.canvas.height / 2);
      return;
    } else if (this.stage === 1) {
      data = this.noisyData;
    } else if (this.stage === 2) {
      data = this.smoothedData;
    } else {
      data = this.segmentedData;
    }
    if (data) {
      this.ctx.putImageData(data, 0, 0);
    }
    this.ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    this.ctx.fillRect(0, this.canvas.height - 30, this.canvas.width, 30);
    this.ctx.fillStyle = '#fff';
    this.ctx.font = '12px Inter, sans-serif';
    this.ctx.textAlign = 'left';
    const stageText = ['Original', 'Stage 1: Smoothing (ROF)', 'Stage 2: Lifting (RGB→Lab)', 'Stage 3: Thresholding (K-means)'];
    this.ctx.fillText(stageText[this.stage], 10, this.canvas.height - 10);
  }
  updateResults() {
    let resultsDiv = Utils.$('.demo-results');
    if (!resultsDiv) {
      resultsDiv = document.createElement('div');
      resultsDiv.className = 'demo-results';
      Utils.$('.controls')?.appendChild(resultsDiv);
    }
    resultsDiv.innerHTML = `
      <h4>Results</h4>
      <div class="result-item"><span>Lambda (smoothing)</span><span>${this.params.lambda.toFixed(1)}</span></div>
      <div class="result-item"><span>ROF Iterations</span><span>${this.params.iterations}</span></div>
      <div class="result-item"><span>Segments (K)</span><span>${this.params.k}</span></div>
      <div class="result-item"><span>Image Size</span><span>${this.canvas.width}x${this.canvas.height}</span></div>
    `;
  }
}

class HomePage {
  constructor() {
    this.init();
  }
  init() {
    this.renderResearcherCard();
    this.renderStats();
    this.renderDomainChart();
    this.renderTimeline();
    this.renderLatestPapers();
  }
  renderResearcherCard() {
    const container = Utils.$('#researcher-card');
    if (!container) return;
    const r = RESEARCH_DATA.researcher;
    container.innerHTML = `
      <div class="researcher-avatar">XC</div>
      <div class="researcher-info">
        <h2>${r.name}</h2>
        <p>${r.position}<br>${r.affiliation}</p>
        <p>${r.bio}</p>
      </div>
    `;
  }
  renderStats() {
    const container = Utils.$('#stats-grid');
    if (!container) return;
    const stats = RESEARCH_DATA.researcher.stats;
    container.innerHTML = `
      <div class="stat-item"><div class="stat-value">${stats.papers}+</div><div class="stat-label">Publications</div></div>
      <div class="stat-item"><div class="stat-value">${Utils.formatNumber(stats.citations)}+</div><div class="stat-label">Citations</div></div>
      <div class="stat-item"><div class="stat-value">${stats.hIndex}</div><div class="stat-label">h-Index</div></div>
      <div class="stat-item"><div class="stat-value">${stats.years}</div><div class="stat-label">Active Years</div></div>
    `;
  }
  renderDomainChart() {
    const ctx = Utils.$('#pie-chart')?.getContext('2d');
    if (!ctx) return;
    const domains = RESEARCH_DATA.domains;
    new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: domains.map(d => d.name),
        datasets: [{
          data: domains.map(d => d.count),
          backgroundColor: domains.map(d => d.color),
          borderWidth: 2,
          borderColor: '#fff'
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        plugins: {
          legend: {
            position: 'right',
            labels: { font: { size: 11 }, padding: 12 }
          }
        }
      }
    });
  }
  renderTimeline() {
    const container = Utils.$('#timeline-container');
    if (!container) return;
    const years = [2011, 2015, 2017, 2023, 2025];
    const events = {
      2011: 'First SIAM paper on Tight-Frame vessel segmentation',
      2015: 'SLaT three-stage segmentation in IEEE TIP',
      2017: 'Radio interferometric imaging series in MNRAS',
      2023: 'Tensor decomposition methods for ML',
      2025: 'HiFi-Mamba for MRI reconstruction'
    };
    container.innerHTML = years.map(year => `
      <div class="timeline-item">
        <div class="timeline-dot"></div>
        <div class="timeline-content">
          <div class="timeline-year">${year}</div>
          <p>${events[year]}</p>
        </div>
      </div>
    `).join('');
  }
  renderLatestPapers() {
    const container = Utils.$('#latest-papers');
    if (!container) return;
    const latest = [...RESEARCH_DATA.papers].sort((a, b) => b.year - a.year).slice(0, 6);
    container.innerHTML = latest.map(paper => {
      const domain = RESEARCH_DATA.domains.find(d => d.id === paper.domain);
      const domainColor = domain ? domain.color : '#64748b';
      return `
        <div class="paper-card" style="border-left-color: ${domainColor}">
          <h3>${paper.title}</h3>
          <div class="meta">
            <span class="year">${paper.year}</span>
            <span class="venue">${paper.venue}</span>
          </div>
        </div>
      `;
    }).join('');
  }
}

document.addEventListener('DOMContentLoaded', () => {
  new Navigation();
  const page = window.location.pathname.split('/').pop() || 'index.html';
  switch (page) {
    case 'index.html':
    case '':
      new HomePage();
      break;
    case 'papers.html':
      new PapersFilter();
      new Modal();
      break;
    case 'network.html':
      new CitationNetwork();
      break;
    case 'methods.html':
      new MethodsTimeline();
      break;
    case 'demo.html':
      new SLATDemo();
      break;
  }
});
