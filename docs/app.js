// 蔡晓昊论文研究可视化系统 - 主应用
// 生成时间: 2026-02-20

let currentPage = 'overview';
let currentPaper = null;
let charts = {};

// 初始化
document.addEventListener('DOMContentLoaded', function() {
    console.log('初始化论文可视化系统...');
    initNavigation();
    initOverviewPage();
    initPapersPage();
    initModalEvents();
});

// ===== 导航 =====
function initNavigation() {
    // 侧边栏导航
    const sidebarNav = document.querySelector('.sidebar-nav');
    if (sidebarNav) {
        sidebarNav.addEventListener('click', function(e) {
            const navItem = e.target.closest('.nav-item');
            if (navItem) {
                const page = navItem.getAttribute('data-page');
                if (page) {
                    e.preventDefault();
                    switchPage(page);
                }
            }
        });
    }

    // 移动端菜单切换
    const menuToggle = document.getElementById('menuToggle');
    if (menuToggle) {
        menuToggle.addEventListener('click', function() {
            const sidebar = document.getElementById('sidebar');
            if (sidebar) sidebar.classList.toggle('open');
        });
    }

    // 全局搜索
    const globalSearch = document.getElementById('globalSearch');
    if (globalSearch) {
        globalSearch.addEventListener('input', function(e) {
            const query = e.target.value.toLowerCase();
            if (query.length > 1) {
                searchPapers(query);
            } else if (query.length === 0) {
                // 清空搜索时重置
                filterPapers();
            }
        });
    }
}

function switchPage(page) {
    console.log('切换到页面:', page);
    currentPage = page;

    // 更新导航激活状态
    const navItems = document.querySelectorAll('.nav-item');
    navItems.forEach(item => {
        if (item.getAttribute('data-page') === page) {
            item.classList.add('active');
        } else {
            item.classList.remove('active');
        }
    });

    // 更新页面标题
    const titles = {
        overview: '研究概览',
        papers: '论文列表',
        timeline: '研究时间线',
        methods: '方法演进',
        domains: '研究领域',
        network: '引用网络',
        pdfs: 'PDF原文'
    };
    const pageTitle = document.getElementById('pageTitle');
    if (pageTitle && titles[page]) {
        pageTitle.textContent = titles[page];
    }

    // 显示/隐藏页面
    const pages = document.querySelectorAll('.page');
    pages.forEach(p => {
        if (p.id === page) {
            p.classList.add('active');
        } else {
            p.classList.remove('active');
        }
    });

    // 关闭移动端菜单
    const sidebar = document.getElementById('sidebar');
    if (sidebar) sidebar.classList.remove('open');

    // 初始化特定页面
    setTimeout(() => {
        if (page === 'timeline') initTimelinePage();
        if (page === 'methods') initMethodsPage();
        if (page === 'network') initNetworkPage();
        if (page === 'pdfs') initPdfsPage();
        if (page === 'domains') initDomainsPage();
    }, 100);
}

// ===== 概览页面 =====
function initOverviewPage() {
    // 更新统计数据
    updateOverviewStats();

    // 领域分布饼图
    const domainCtx = document.getElementById('domainPieChart');
    if (domainCtx && typeof Chart !== 'undefined') {
        const labels = [];
        const data = [];
        const colors = [];

        for (const cat in PAPERS_DATA.categories) {
            labels.push(cat);
            data.push(PAPERS_DATA.categories[cat].count);
            colors.push(PAPERS_DATA.categories[cat].color);
        }

        charts.domainPie = new Chart(domainCtx, {
            type: 'doughnut',
            data: {
                labels: labels,
                datasets: [{
                    data: data,
                    backgroundColor: colors,
                    borderWidth: 2,
                    borderColor: '#fff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            padding: 15,
                            usePointStyle: true,
                            font: { size: 11 }
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const label = context.label || '';
                                const value = context.parsed || 0;
                                const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                const percentage = ((value / total) * 100).toFixed(1);
                                return `${label}: ${value}篇 (${percentage}%)`;
                            }
                        }
                    }
                }
            }
        });
    }

    // 年度趋势图
    const yearCtx = document.getElementById('yearTrendChart');
    if (yearCtx && typeof Chart !== 'undefined') {
        const yearCounts = {};
        PAPERS_DATA.papers.forEach(p => {
            if (p.year) {
                yearCounts[p.year] = (yearCounts[p.year] || 0) + 1;
            }
        });

        const years = Object.keys(yearCounts).map(Number).sort((a, b) => a - b);
        const counts = years.map(y => yearCounts[y]);

        charts.yearTrend = new Chart(yearCtx, {
            type: 'line',
            data: {
                labels: years,
                datasets: [{
                    label: '论文数量',
                    data: counts,
                    borderColor: '#2563eb',
                    backgroundColor: 'rgba(37, 99, 235, 0.1)',
                    fill: true,
                    tension: 0.4,
                    pointRadius: 5,
                    pointHoverRadius: 7
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `${context.parsed.y} 篇论文`;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            stepSize: 1
                        }
                    }
                }
            }
        });
    }
}

// 更新概览页面统计数据
function updateOverviewStats() {
    // 更新论文总数
    const paperCountBadge = document.getElementById('paperCount');
    if (paperCountBadge) {
        paperCountBadge.textContent = PAPERS_DATA.summary.total;
    }

    // 更新进度信息
    const progressPercent = document.querySelector('.progress-percent');
    const progressText = document.querySelector('.progress-text');
    const progressFill = document.querySelector('.progress-fill');

    if (progressPercent && progressText && progressFill) {
        const percentage = Math.round((PAPERS_DATA.summary.filled / PAPERS_DATA.summary.total) * 100);
        progressPercent.textContent = percentage + '%';
        progressText.textContent = `${PAPERS_DATA.summary.filled}/${PAPERS_DATA.summary.total} 篇已完成`;
        progressFill.style.width = percentage + '%';
    }

    // 更新统计卡片数字
    updateStatCards();
}

function updateStatCards() {
    const categoryCounts = {};
    PAPERS_DATA.papers.forEach(p => {
        categoryCounts[p.category] = (categoryCounts[p.category] || 0) + 1;
    });

    // 更新各领域统计卡片
    const statCards = document.querySelectorAll('.stat-card');
    statCards.forEach(card => {
        const title = card.querySelector('h3');
        if (title) {
            const titleText = title.textContent;
            let category = null;
            if (titleText.includes('基础理论')) category = '基础理论';
            else if (titleText.includes('变分分割')) category = '变分分割';
            else if (titleText.includes('深度学习')) category = '深度学习';
            else if (titleText.includes('雷达与无线电')) category = '雷达与无线电';
            else if (titleText.includes('医学图像')) category = '医学图像';
            else if (titleText.includes('张量分解')) category = '张量分解';
            else if (titleText.includes('3D视觉') || titleText.includes('其他')) {
                // 3D视觉与点云归类到"其他"
                category = '3D视觉与点云';
            }

            if (category && categoryCounts[category]) {
                const valueEl = card.querySelector('.stat-value');
                if (valueEl) valueEl.textContent = categoryCounts[category];
            }
        }
    });
}

// ===== 论文列表页面 =====
let filteredPapers = [];
let currentPageNum = 1;
const itemsPerPage = 12;

function initPapersPage() {
    filteredPapers = PAPERS_DATA.papers.slice();

    // 更新筛选选项以包含新分类
    updateCategoryFilter();

    const categoryFilter = document.getElementById('categoryFilter');
    const statusFilter = document.getElementById('statusFilter');
    const sortFilter = document.getElementById('sortFilter');

    if (categoryFilter) categoryFilter.addEventListener('change', filterPapers);
    if (statusFilter) statusFilter.addEventListener('change', filterPapers);
    if (sortFilter) sortFilter.addEventListener('change', filterPapers);

    filterPapers();
}

function updateCategoryFilter() {
    const categoryFilter = document.getElementById('categoryFilter');
    if (!categoryFilter) return;

    // 保留"全部"选项
    const allOption = categoryFilter.querySelector('option[value="all"]');
    categoryFilter.innerHTML = '';
    if (allOption) categoryFilter.appendChild(allOption);
    else categoryFilter.innerHTML = '<option value="all">全部</option>';

    // 添加所有分类
    Object.keys(PAPERS_DATA.categories).forEach(cat => {
        const option = document.createElement('option');
        option.value = cat;
        option.textContent = cat;
        categoryFilter.appendChild(option);
    });
}

function filterPapers() {
    let papers = PAPERS_DATA.papers.slice();

    const categoryFilter = document.getElementById('categoryFilter');
    if (categoryFilter && categoryFilter.value !== 'all') {
        papers = papers.filter(p => p.category === categoryFilter.value);
    }

    const statusFilter = document.getElementById('statusFilter');
    if (statusFilter && statusFilter.value !== 'all') {
        papers = papers.filter(p => p.status === statusFilter.value);
    }

    const sortFilter = document.getElementById('sortFilter');
    const sortBy = sortFilter ? sortFilter.value : 'id';

    papers.sort((a, b) => {
        if (sortBy === 'year') return (b.year || 0) - (a.year || 0);
        if (sortBy === 'category') return a.category.localeCompare(b.category);
        if (sortBy === 'id') return (a.id || 0) - (b.id || 0);
        // Sort ID as number for proper numeric ordering
        return (parseInt(a.id) || 0) - (parseInt(b.id) || 0);
    });

    filteredPapers = papers;
    currentPageNum = 1;
    renderPapers();
}

function renderPapers() {
    const grid = document.getElementById('papersGrid');
    if (!grid) return;

    const start = (currentPageNum - 1) * itemsPerPage;
    const end = start + itemsPerPage;
    const pagePapers = filteredPapers.slice(start, end);

    if (pagePapers.length === 0) {
        grid.innerHTML = '<div style="grid-column: 1/-1; text-align: center; padding: 3rem; color: #6b7280;">没有找到匹配的论文</div>';
        renderPagination();
        return;
    }

    let html = '';
    pagePapers.forEach(paper => {
        const categoryClass = Utils.getCategoryClass(paper.category);
        const categoryColor = Utils.getCategoryColor(paper.category);
        html += `
            <div class="paper-card ${categoryClass}" onclick="openPaperModalById('${paper.id}')">
                <div class="paper-card-header">
                    <span class="paper-id-badge" style="background-color: ${categoryColor}20; color: ${categoryColor}">[${paper.id}]</span>
                    <span class="paper-status ${paper.status}">${paper.status === 'filled' ? '✓' : '○'}</span>
                </div>
                <div class="paper-title">${paper.title}</div>
                <div class="paper-footer">
                    <span class="badge ${categoryClass}" style="background-color: ${categoryColor}20; color: ${categoryColor}">${paper.category}</span>
                    <span class="paper-year">${paper.year}</span>
                </div>
            </div>
        `;
    });

    grid.innerHTML = html;
    renderPagination();
}

function renderPagination() {
    const container = document.getElementById('pagination');
    if (!container) return;

    const totalPages = Math.ceil(filteredPapers.length / itemsPerPage);
    if (totalPages <= 1) {
        container.innerHTML = '';
        return;
    }

    let html = '';
    // 上一页
    html += `<button class="page-btn" ${currentPageNum === 1 ? 'disabled' : ''} onclick="goToPage(${currentPageNum - 1})">‹</button>`;

    // 页码
    for (let i = 1; i <= totalPages; i++) {
        if (i === 1 || i === totalPages || (i >= currentPageNum - 1 && i <= currentPageNum + 1)) {
            html += `<button class="page-btn ${i === currentPageNum ? 'active' : ''}" onclick="goToPage(${i})">${i}</button>`;
        } else if (i === currentPageNum - 2 || i === currentPageNum + 2) {
            html += `<span class="page-ellipsis">...</span>`;
        }
    }

    // 下一页
    html += `<button class="page-btn" ${currentPageNum === totalPages ? 'disabled' : ''} onclick="goToPage(${currentPageNum + 1})">›</button>`;

    container.innerHTML = html;
}

function goToPage(n) {
    if (n < 1 || n > Math.ceil(filteredPapers.length / itemsPerPage)) return;
    currentPageNum = n;
    renderPapers();
    // 滚动到顶部
    document.getElementById('papers').scrollIntoView({ behavior: 'smooth' });
}

function openPaperModalById(id) {
    const paper = PAPERS_DATA.papers.find(p => p.id === id);
    if (paper) openPaperModal(paper);
}

function searchPapers(query) {
    switchPage('papers');
    filteredPapers = PAPERS_DATA.papers.filter(p => {
        const searchFields = [
            p.title,
            p.titleEn,
            p.id,
            p.summary,
            ...(p.methods || []),
            ...(p.authors || [])
        ].join(' ').toLowerCase();
        return searchFields.includes(query);
    });
    currentPageNum = 1;
    renderPapers();
}

// ===== 时间线页面 =====
let timelineInited = false;
function initTimelinePage() {
    if (timelineInited) {
        charts.timeline && charts.timeline.resize();
        return;
    }

    const chartDom = document.getElementById('timelineChart');
    if (!chartDom) return;

    if (typeof echarts === 'undefined') {
        chartDom.innerHTML = '<div style="padding:40px;text-align:center"><h3>图表库加载失败</h3></div>';
        return;
    }

    try {
        charts.timeline = echarts.init(chartDom);
        timelineInited = true;

        const sorted = PAPERS_DATA.papers.filter(p => p.year && !isNaN(p.year));
        sorted.sort((a, b) => a.year - b.year);

        const years = [...new Set(sorted.map(p => p.year))].sort((a, b) => a - b);
        const yearGroups = {};
        sorted.forEach(p => {
            if (!yearGroups[p.year]) yearGroups[p.year] = [];
            yearGroups[p.year].push(p);
        });

        const data = sorted.map(paper => {
            const yearIdx = years.indexOf(paper.year);
            const idxInYear = yearGroups[paper.year].indexOf(paper);
            return {
                value: [yearIdx, idxInYear],
                paper: paper,
                itemStyle: { color: Utils.getCategoryColor(paper.category) },
                symbolSize: 18,
                label: {
                    show: idxInYear === 0, // 只显示每年的第一篇
                    position: 'top',
                    formatter: () => paper.year,
                    fontSize: 12,
                    fontWeight: 'bold'
                }
            };
        });

        const maxInYear = Math.max(...Object.values(yearGroups).map(arr => arr.length));

        charts.timeline.setOption({
            tooltip: {
                trigger: 'item',
                backgroundColor: 'rgba(255, 255, 255, 0.95)',
                borderColor: '#2563eb',
                borderWidth: 1,
                textStyle: {
                    color: '#1f2937',
                    fontSize: 13
                },
                extraCssText: 'box-shadow: 0 4px 12px rgba(0,0,0,0.15); border-radius: 8px; padding: 12px;',
                formatter: params => {
                    if (params.data && params.data.paper) {
                        const p = params.data.paper;
                        const categoryColors = {
                            '基础理论': '#ef4444',
                            '变分分割': '#3b82f6',
                            '深度学习': '#10b981',
                            '雷达与无线电': '#8b5cf6',
                            '医学图像': '#f59e0b',
                            '张量分解': '#ec4899',
                            '3D视觉与点云': '#06b6d4'
                        };
                        const catColor = categoryColors[p.category] || '#6b7280';
                        return `<div style="margin-bottom: 6px;"><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${catColor};margin-right:6px;"></span><strong style="color:${catColor};">${p.category}</strong></div>` +
                               `<div style="font-size:14px;font-weight:600;margin-bottom:4px;">${p.title}</div>` +
                               `<div style="color:#6b7280;font-size:12px;">[${p.id}] 年份: ${p.year}</div>`;
                    }
                    return '';
                }
            },
            grid: { left: '5%', right: '10%', bottom: '10%', top: '5%' },
            xAxis: {
                type: 'category',
                data: years,
                name: '年份',
                nameLocation: 'middle',
                nameGap: 30,
                nameTextStyle: {
                    fontSize: 13,
                    fontWeight: 600
                },
                axisLabel: { rotate: 45, fontSize: 11, color: '#4b5563' },
                axisLine: { lineStyle: { color: '#d1d5db' } }
            },
            yAxis: {
                type: 'value',
                show: false,
                min: -0.5,
                max: maxInYear + 0.5
            },
            series: [{
                type: 'scatter',
                data: data,
                symbolSize: function(val) {
                    return 16;
                },
                itemStyle: {
                    opacity: 0.85,
                    borderWidth: 2,
                    borderColor: '#fff'
                },
                label: {
                    show: false  // 使用tooltip替代标签
                },
                labelLayout: {
                    hideOverlap: true
                },
                emphasis: {
                    label: {
                        show: false
                    },
                    itemStyle: {
                        opacity: 1,
                        borderColor: '#1f2937',
                        borderWidth: 2,
                        shadowBlur: 10,
                        shadowColor: 'rgba(0,0,0,0.3)'
                    },
                    scale: 1.3
                }
            }]
        });

        charts.timeline.on('click', params => {
            if (params.data && params.data.paper) openPaperModal(params.data.paper);
        });

        // 绑定过滤按钮
        const filterButtons = document.querySelectorAll('.timeline-filters .filter-btn');
        filterButtons.forEach(btn => {
            btn.addEventListener('click', function() {
                filterButtons.forEach(b => b.classList.remove('active'));
                this.classList.add('active');

                const filter = this.getAttribute('data-filter');
                let filteredData = data;

                if (filter !== 'all') {
                    filteredData = data.filter(d => d.paper && d.paper.category === filter);
                }

                charts.timeline.setOption({
                    series: [{ data: filteredData }]
                });
            });
        });

    } catch (e) {
        console.error('时间线图表错误:', e);
        chartDom.innerHTML = '<div style="padding:40px;text-align:center;color:red"><h3>图表加载失败</h3></div>';
    }
}

// ===== 方法演进页面 =====
let methodsInited = false;
function initMethodsPage() {
    if (methodsInited) {
        charts.methods && charts.methods.resize();
        return;
    }

    const chartDom = document.getElementById('methodsChart');
    if (!chartDom) return;

    if (typeof echarts === 'undefined') {
        chartDom.innerHTML = '<div style="padding:40px;text-align:center"><h3>图表库加载失败</h3></div>';
        return;
    }

    try {
        charts.methods = echarts.init(chartDom);
        methodsInited = true;

        // 从citations数据构建节点和链接
        const nodes = [];
        const links = [];
        const nodeSet = new Set();

        // 添加所有被引用的论文节点
        PAPERS_DATA.citations.forEach(c => {
            nodeSet.add(c.source);
            nodeSet.add(c.target);
            links.push({
                source: c.source,
                target: c.target,
                name: c.description || '',
                lineStyle: {
                    color: getLinkColor(c.type),
                    curveness: 0.3
                }
            });
        });

        // 创建节点
        nodeSet.forEach(id => {
            const paper = PAPERS_DATA.papers.find(p => p.id === id);
            if (paper) {
                nodes.push({
                    id: id,
                    name: `${paper.id}\n${paper.year}`,
                    value: paper.id,
                    symbolSize: 50,
                    itemStyle: { color: Utils.getCategoryColor(paper.category) },
                    paper: paper
                });
            }
        });

        charts.methods.setOption({
            tooltip: {
                formatter: params => {
                    if (params.dataType === 'node') {
                        const paper = params.data.paper;
                        return `<strong>[${paper.id}] ${paper.year}</strong><br/>${paper.title}`;
                    } else if (params.dataType === 'edge') {
                        return params.data.name;
                    }
                    return '';
                }
            },
            series: [{
                type: 'graph',
                layout: 'force',
                data: nodes,
                links: links,
                roam: true,
                label: {
                    show: true,
                    fontSize: 10,
                    formatter: params => {
                        return params.data.id || '';
                    }
                },
                edgeSymbol: ['circle', 'arrow'],
                edgeSymbolSize: [4, 10],
                force: {
                    repulsion: 500,
                    edgeLength: 150,
                    gravity: 0.1
                },
                lineStyle: {
                    width: 2,
                    opacity: 0.6
                }
            }]
        });

        charts.methods.on('click', params => {
            if (params.dataType === 'node' && params.data.paper) {
                openPaperModal(params.data.paper);
            }
        });

    } catch (e) {
        console.error('方法演进图表错误:', e);
    }
}

function getLinkColor(type) {
    const colors = {
        '方法扩展': '#10b981',
        '方法升级': '#ef4444',
        '理论基础': '#3b82f6',
        '方法发展': '#8b5cf6',
        '方法应用': '#f59e0b',
        '方法比较': '#ec4899'
    };
    return colors[type] || '#6b7280';
}

// ===== 研究领域页面 =====
let domainsInited = false;
function initDomainsPage() {
    if (domainsInited) return;
    domainsInited = true;

    const grid = document.querySelector('.domains-grid');
    if (!grid) return;

    let html = '';
    for (const cat in PAPERS_DATA.categories) {
        const categoryData = PAPERS_DATA.categories[cat];
        const papers = PAPERS_DATA.papers.filter(p => p.category === cat);
        const categoryClass = Utils.getCategoryClass(cat);
        const categoryColor = Utils.getCategoryColor(cat);

        html += `
            <div class="domain-card">
                <div class="domain-header">
                    <span class="domain-icon">${getDomainIcon(cat)}</span>
                    <h3>${cat}</h3>
                    <span class="domain-count">${categoryData.count}篇</span>
                </div>
                <p class="domain-desc">${categoryData.description}</p>
                <ul class="domain-papers">
        `;

        // 显示前6篇论文
        papers.slice(0, 6).forEach(paper => {
            html += `<li onclick="openPaperModalById('${paper.id}')" style="cursor:pointer">${paper.title} (${paper.year})</li>`;
        });

        if (papers.length > 6) {
            html += `<li style="color:var(--primary);cursor:pointer" onclick="filterAndShowPapers('${cat}')">+ 还有${papers.length - 6}篇...</li>`;
        }

        html += `
                </ul>
            </div>
        `;
    }

    grid.innerHTML = html;
}

function getDomainIcon(category) {
    const icons = {
        '基础理论': '📘',
        '变分分割': '✂️',
        '深度学习': '🤖',
        '雷达与无线电': '📡',
        '医学图像': '🏥',
        '张量分解': '🔷',
        '3D视觉与点云': '🎲'
    };
    return icons[category] || '🔬';
}

function filterAndShowPapers(category) {
    switchPage('papers');
    const categoryFilter = document.getElementById('categoryFilter');
    if (categoryFilter) {
        categoryFilter.value = category;
        filterPapers();
    }
}

// ===== 引用网络页面 =====
let networkInited = false;
function initNetworkPage() {
    if (networkInited) {
        charts.network && charts.network.resize();
        return;
    }

    const chartDom = document.getElementById('networkChart');
    if (!chartDom) return;

    if (typeof echarts === 'undefined') {
        chartDom.innerHTML = '<div style="padding:40px;text-align:center"><h3>图表库加载失败</h3></div>';
        return;
    }

    try {
        charts.network = echarts.init(chartDom);
        networkInited = true;

        const nodeMap = {};
        const nodes = [];

        // 按分类组织节点
        PAPERS_DATA.papers.forEach(p => {
            if (p.id && !nodeMap[p.id]) {
                nodeMap[p.id] = true;
                nodes.push({
                    id: p.id,
                    name: p.id,
                    value: p.category,
                    symbolSize: 30,
                    itemStyle: { color: Utils.getCategoryColor(p.category) },
                    paper: p,
                    category: p.category
                });
            }
        });

        const links = [];
        PAPERS_DATA.citations.forEach(c => {
            if (nodeMap[c.source] && nodeMap[c.target]) {
                links.push({
                    source: c.source,
                    target: c.target,
                    value: c.strength || 1,
                    lineStyle: {
                        width: (c.strength || 1) * 0.5,
                        opacity: 0.6
                    }
                });
            }
        });

        charts.network.setOption({
            tooltip: {
                trigger: 'item',
                formatter: params => {
                    if (params.dataType === 'node') {
                        const p = params.data.paper;
                        return `<strong>[${p.id}] ${p.year}</strong><br/>${p.title}<br/>分类: ${p.category}`;
                    } else if (params.dataType === 'edge') {
                        return `${params.data.source} → ${params.data.target}`;
                    }
                    return '';
                }
            },
            legend: {
                data: Object.keys(PAPERS_DATA.categories),
                orient: 'vertical',
                right: 10,
                top: 'center',
                textStyle: { fontSize: 11 }
            },
            series: [{
                type: 'graph',
                layout: 'force',
                data: nodes,
                links: links,
                categories: Object.keys(PAPERS_DATA.categories).map(cat => ({ name: cat })),
                roam: true,
                label: {
                    show: true,
                    position: 'right',
                    formatter: '{b}',
                    fontSize: 10
                },
                labelLayout: {
                    hideOverlap: true
                },
                force: {
                    repulsion: 400,
                    edgeLength: [100, 200],
                    gravity: 0.1,
                    friction: 0.6
                },
                edgeSymbol: ['none', 'arrow'],
                edgeSymbolSize: [0, 8],
                lineStyle: {
                    color: '#source',
                    curveness: 0.1
                },
                emphasis: {
                    focus: 'adjacency',
                    lineStyle: {
                        width: 3
                    }
                }
            }]
        });

        charts.network.on('click', params => {
            if (params.dataType === 'node' && params.data.paper) {
                openPaperModal(params.data.paper);
            }
        });

    } catch (e) {
        console.error('网络图表错误:', e);
        chartDom.innerHTML = '<div style="padding:40px;text-align:center;color:red"><h3>图表加载失败</h3></div>';
    }
}

// ===== 模态框事件 =====
function initModalEvents() {
    const modalClose = document.querySelector('.modal-close');
    if (modalClose) modalClose.addEventListener('click', closeModal);

    const modalOverlay = document.querySelector('.modal-overlay');
    if (modalOverlay) modalOverlay.addEventListener('click', closeModal);

    const btnReadNote = document.getElementById('btnReadNote');
    if (btnReadNote) btnReadNote.addEventListener('click', showNoteContent);

    const btnViewPDF = document.getElementById('btnViewPDF');
    if (btnViewPDF) btnViewPDF.addEventListener('click', openPDF);

    // ESC键关闭
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') closeModal();
    });
}

function openPaperModal(paper) {
    currentPaper = paper;

    document.getElementById('modalPaperId').textContent = '[' + paper.id + ']';
    document.getElementById('modalPaperTitle').textContent = paper.title;

    const catBadge = document.getElementById('modalPaperCategory');
    catBadge.textContent = paper.category;
    catBadge.className = 'badge ' + Utils.getCategoryClass(paper.category);
    catBadge.style.backgroundColor = Utils.getCategoryColor(paper.category) + '20';
    catBadge.style.color = Utils.getCategoryColor(paper.category);

    document.getElementById('modalPaperYear').textContent = paper.year + '年';

    const statusBadge = document.getElementById('modalPaperStatus');
    statusBadge.textContent = paper.status === 'filled' ? '✓ 已完成' : '○ 待填充';
    statusBadge.className = 'badge ' + (paper.status === 'filled' ? 'success' : 'warning');

    // 显示论文信息
    showPaperInfo(paper);

    // 更新PDF按钮状态
    const hasPDF = paper.pdfFile && paper.pdfFile.trim() !== '';
    const btnViewPDF = document.getElementById('btnViewPDF');
    btnViewPDF.disabled = !hasPDF;
    btnViewPDF.style.opacity = hasPDF ? '1' : '0.5';

    document.getElementById('paperModal').classList.add('active');
}

function showPaperInfo(paper) {
    const notePreview = document.getElementById('notePreview');

    let html = `
        <div class="paper-detail">
            <div class="detail-section">
                <h4>基本信息</h4>
                <table class="info-table">
                    <tr><td>编号</td><td>${paper.id}</td></tr>
                    <tr><td>年份</td><td>${paper.year}</td></tr>
                    <tr><td>分类</td><td>${paper.category}</td></tr>
                    <tr><td>状态</td><td>${paper.status === 'filled' ? '✓ 已完成' : '○ 待填充'}</td></tr>
                </table>
            </div>
    `;

    if (paper.titleEn) {
        html += `
            <div class="detail-section">
                <h4>英文标题</h4>
                <p>${paper.titleEn}</p>
            </div>
        `;
    }

    if (paper.authors && paper.authors.length > 0) {
        html += `
            <div class="detail-section">
                <h4>作者</h4>
                <p>${paper.authors.join(', ')}</p>
            </div>
        `;
    }

    if (paper.summary) {
        html += `
            <div class="detail-section">
                <h4>摘要</h4>
                <p>${paper.summary}</p>
            </div>
        `;
    }

    if (paper.methods && paper.methods.length > 0) {
        html += `
            <div class="detail-section">
                <h4>方法</h4>
                <div class="tags">
                    ${paper.methods.map(m => `<span class="tag">${m}</span>`).join('')}
                </div>
            </div>
        `;
    }

    if (paper.innovations && paper.innovations.length > 0) {
        html += `
            <div class="detail-section">
                <h4>创新点</h4>
                <ul>
                    ${paper.innovations.map(i => `<li>${i}</li>`).join('')}
                </ul>
            </div>
        `;
    }

    html += `
            <div class="detail-section">
                <h4>相关文件</h4>
                <div class="file-links">
    `;

    if (paper.pdfFile) {
        const pdfPath = Utils.getPDFPath(paper.pdfFile);
        html += `<a href="${pdfPath}" target="_blank" class="file-link">📄 查看PDF原文</a>`;
    }

    if (paper.noteFile) {
        const notePath = Utils.getNotePath(paper.noteFile);
        html += `<a href="${notePath}" target="_blank" class="file-link">📖 查看精读笔记</a>`;
    }

    if (paper.arxiv) {
        html += `<a href="https://arxiv.org/abs/${paper.arxiv}" target="_blank" class="file-link">🔗 arXiv链接</a>`;
    }

    html += `
                </div>
            </div>
        </div>
    `;

    notePreview.innerHTML = html;
}

function closeModal() {
    document.getElementById('paperModal').classList.remove('active');
    currentPaper = null;
}

function showNoteContent() {
    if (!currentPaper) return;
    // 切换到笔记内容显示
    showPaperInfo(currentPaper);
}

function openPDF() {
    if (!currentPaper || !currentPaper.pdfFile) {
        alert('该论文暂无PDF文件');
        return;
    }
    const path = Utils.getPDFPath(currentPaper.pdfFile);
    if (path) {
        window.open(path, '_blank');
    }
}

// 窗口大小改变时重绘图表
window.addEventListener('resize', () => {
    if (charts.timeline) charts.timeline.resize();
    if (charts.network) charts.network.resize();
    if (charts.methods) charts.methods.resize();
    if (charts.domainPie) charts.domainPie.resize();
    if (charts.yearTrend) charts.yearTrend.resize();
});

// ===== PDF列表页面 =====
let pdfsInited = false;
function initPdfsPage() {
    if (pdfsInited) return;
    pdfsInited = true;

    const container = document.getElementById('pdfsContainer');
    if (!container) return;

    // 按分类组织PDF
    const categoryGroups = {};
    Object.keys(PAPERS_DATA.categories).forEach(cat => {
        categoryGroups[cat] = [];
    });

    PAPERS_DATA.papers.forEach(paper => {
        if (paper.pdfFile && paper.pdfFile.trim() !== '') {
            categoryGroups[paper.category].push(paper);
        }
    });

    let html = '';
    let index = 1;

    for (const category in categoryGroups) {
        const papers = categoryGroups[category];
        if (papers.length === 0) continue;

        const categoryColor = Utils.getCategoryColor(category);

        html += `
            <div class="pdf-category">
                <div class="pdf-category-title" style="background: ${categoryColor}">
                    ${category} (${papers.length}篇)
                </div>
                <div class="pdf-list">
        `;

        papers.forEach(paper => {
            const pdfPath = Utils.getPDFPath(paper.pdfFile);
            html += `
                <a class="pdf-item" href="${pdfPath}" target="_blank" style="border-left-color: ${categoryColor}">
                    <span class="pdf-number" style="background-color: ${categoryColor}">${index++}</span>
                    <span class="pdf-name">${paper.title}</span>
                    <span class="pdf-id">[${paper.id}]</span>
                </a>
            `;
        });

        html += `
                </div>
            </div>
        `;
    }

    if (html === '') {
        html = '<div style="text-align: center; padding: 3rem; color: #6b7280;">暂无PDF文件</div>';
    }

    container.innerHTML = html;
}
