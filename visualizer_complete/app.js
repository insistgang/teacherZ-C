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

// ===== 暴露全局函数（供内联事件处理器使用）=====
window.openPaperModalById = openPaperModalById;
window.goToPage = goToPage;
window.filterPapers = filterPapers;
window.filterAndShowPapers = filterAndShowPapers;
window.switchPage = switchPage;
window.showNoteContent = showNoteContent;
window.showPaperInfo = showPaperInfo;
window.markdownToHtml = markdownToHtml;

function goToPage(n) {
    if (n < 1 || n > Math.ceil(filteredPapers.length / itemsPerPage)) return;
    currentPageNum = n;
    renderPapers();
    // 滚动到顶部
    document.getElementById('papers').scrollIntoView({ behavior: 'smooth' });
}

function openPaperModalById(id) {
    // 将字符串 id 转换为数字，因为 data.js 中的 id 是数字类型
    const numId = parseInt(id, 10);
    const paper = PAPERS_DATA.papers.find(p => p.id === numId);
    if (paper) {
        openPaperModal(paper);
    } else {
        console.error('找不到论文 id:', id);
    }
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
                backgroundColor: 'rgba(255, 255, 255, 0.98)',
                borderColor: '#2563eb',
                borderWidth: 1,
                textStyle: {
                    color: '#1f2937',
                    fontSize: 13
                },
                extraCssText: 'box-shadow: 0 8px 24px rgba(0,0,0,0.12); border-radius: 12px; padding: 16px; max-width: 400px;',
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
                        return `<div style="margin-bottom: 10px; display: flex; align-items: center; gap: 8px;">` +
                               `<span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:${catColor};box-shadow: 0 0 0 3px ${catColor}33;"></span>` +
                               `<strong style="color:${catColor}; font-size: 13px;">${p.category}</strong>` +
                               `<span style="margin-left: auto; color: #9ca3af; font-size: 11px;">#${p.id}</span></div>` +
                               `<div style="font-size: 14px; font-weight: 600; margin-bottom: 8px; line-height: 1.5; color: #111827;">${p.title}</div>` +
                               `<div style="color: #6b7280; font-size: 12px; display: flex; gap: 12px;">` +
                               `<span>📅 ${p.year}</span>` +
                               `<span>📄 ${p.pdfFile ? 'PDF 可用' : '暂无 PDF'}</span></div>`;
                    }
                    return '';
                }
            },
            grid: { left: '3%', right: '5%', bottom: '15%', top: '10%', containLabel: true },
            dataZoom: [
                {
                    type: 'inside',
                    xAxisIndex: 0,
                    start: 0,
                    end: 100
                },
                {
                    type: 'slider',
                    xAxisIndex: 0,
                    start: 0,
                    end: 100,
                    height: 20,
                    bottom: 10,
                    borderColor: '#e5e7eb',
                    fillerColor: 'rgba(37, 99, 235, 0.1)',
                    handleStyle: { color: '#2563eb' },
                    textStyle: { color: '#6b7280' }
                }
            ],
            xAxis: {
                type: 'category',
                data: years,
                name: '年份',
                nameLocation: 'middle',
                nameGap: 35,
                nameTextStyle: {
                    fontSize: 14,
                    fontWeight: 600,
                    color: '#374151'
                },
                axisLabel: { 
                    rotate: 0, 
                    fontSize: 12, 
                    color: '#4b5563',
                    fontWeight: 500,
                    interval: 0
                },
                axisLine: { lineStyle: { color: '#d1d5db', width: 2 } },
                axisTick: { show: true, alignWithLabel: true },
                splitLine: { 
                    show: true, 
                    lineStyle: { color: '#f3f4f6', type: 'dashed' } 
                }
            },
            yAxis: {
                type: 'value',
                show: true,
                name: '论文分布',
                nameTextStyle: {
                    fontSize: 12,
                    color: '#6b7280'
                },
                min: -0.5,
                max: maxInYear + 0.5,
                axisLine: { show: false },
                axisTick: { show: false },
                axisLabel: { show: false },
                splitLine: { show: false }
            },
            series: [{
                type: 'scatter',
                data: data,
                symbolSize: function(val) {
                    return 22;
                },
                itemStyle: {
                    opacity: 0.9,
                    borderWidth: 3,
                    borderColor: '#fff',
                    shadowBlur: 8,
                    shadowColor: 'rgba(0,0,0,0.15)'
                },
                label: {
                    show: false
                },
                labelLayout: {
                    hideOverlap: true
                },
                emphasis: {
                    scale: 1.5,
                    itemStyle: {
                        opacity: 1,
                        borderColor: '#fff',
                        borderWidth: 4,
                        shadowBlur: 20,
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
    if (!paper) {
        console.error('openPaperModal: paper 为空');
        return;
    }

    currentPaper = paper;
    console.log('打开论文模态框:', paper.id, paper.title);

    const modalPaperId = document.getElementById('modalPaperId');
    const modalPaperTitle = document.getElementById('modalPaperTitle');

    if (!modalPaperId || !modalPaperTitle) {
        console.error('找不到模态框元素');
        return;
    }

    modalPaperId.textContent = '[' + paper.id + ']';
    modalPaperTitle.textContent = paper.title;

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

    // 隐藏返回按钮、TOC侧边栏和进度条
    const btnBackToInfo = document.getElementById('btnBackToInfo');
    const tocSidebar = document.getElementById('tocSidebar');
    const progressContainer = document.getElementById('readingProgressContainer');
    const backToTop = document.getElementById('backToTop');

    if (btnBackToInfo) btnBackToInfo.style.display = 'none';
    if (tocSidebar) tocSidebar.classList.add('hidden');
    if (progressContainer) progressContainer.style.display = 'none';
    if (backToTop) backToTop.classList.remove('visible');

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
        html += `<button onclick="showNoteContent()" class="file-link" style="cursor: pointer; border: none; background: none; font-size: inherit;">📖 查看精读笔记</button>`;
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

    // 重置TOC和进度条
    const tocSidebar = document.getElementById('tocSidebar');
    const tocNav = document.getElementById('tocNav');
    const progressContainer = document.getElementById('readingProgressContainer');
    const backToTop = document.getElementById('backToTop');
    const btnBackToInfo = document.getElementById('btnBackToInfo');

    if (tocSidebar) tocSidebar.classList.add('hidden');
    if (tocNav) tocNav.innerHTML = '';
    if (progressContainer) progressContainer.style.display = 'none';
    if (backToTop) backToTop.classList.remove('visible');
    if (btnBackToInfo) btnBackToInfo.style.display = 'none';
}

function showNoteContent() {
    if (!currentPaper) {
        console.error('showNoteContent: 没有选中论文');
        return;
    }

    const notePreview = document.getElementById('notePreview');
    const tocSidebar = document.getElementById('tocSidebar');
    const tocNav = document.getElementById('tocNav');
    const progressContainer = document.getElementById('readingProgressContainer');
    const btnBackToInfo = document.getElementById('btnBackToInfo');

    if (!notePreview) {
        console.error('找不到 notePreview 元素');
        return;
    }

    // 显示返回按钮
    if (btnBackToInfo) {
        btnBackToInfo.style.display = 'inline-flex';
        btnBackToInfo.onclick = function() {
            showPaperInfo(currentPaper);
            if (tocSidebar) tocSidebar.classList.add('hidden');
            if (progressContainer) progressContainer.style.display = 'none';
            btnBackToInfo.style.display = 'none';
        };
    }

    // 如果没有笔记文件
    if (!currentPaper.noteFile) {
        notePreview.innerHTML = `
            <div class="paper-detail">
                <div class="detail-section" style="text-align: center; padding: 3rem;">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">📝</div>
                    <h3>暂无精读笔记</h3>
                    <p style="color: #6b7280;">该论文暂时没有精读笔记</p>
                </div>
            </div>
        `;
        if (tocSidebar) tocSidebar.classList.add('hidden');
        if (progressContainer) progressContainer.style.display = 'none';
        return;
    }

    const notePath = Utils.getNotePath(currentPaper.noteFile);
    console.log('加载笔记:', notePath);

    // 显示加载中
    notePreview.innerHTML = `
        <div class="paper-detail">
            <div class="detail-section" style="text-align: center; padding: 3rem;">
                <div style="font-size: 2rem; margin-bottom: 1rem;">⏳</div>
                <p>正在加载笔记...</p>
            </div>
        </div>
    `;

    // 加载笔记文件 - 确保使用 UTF-8 编码
    fetch(notePath)
        .then(response => {
            if (!response.ok) {
                throw new Error('笔记文件不存在: ' + notePath);
            }
            // 强制使用 UTF-8 解码
            return response.arrayBuffer().then(buffer => {
                const decoder = new TextDecoder('utf-8');
                return decoder.decode(buffer);
            });
        })
        .then(markdown => {
            // 将 Markdown 转换为 HTML（支持 LaTeX）
            const html = markdownToHtml(markdown);

            // 创建临时容器解析HTML
            const tempDiv = document.createElement('div');
            tempDiv.innerHTML = `<div class="markdown-content">${html}</div>`;

            // 生成目录
            const tocData = generateTOC(tempDiv);

            // 更新内容
            notePreview.innerHTML = tempDiv.innerHTML;

            // 更新目录
            if (tocNav && tocData.length > 0) {
                renderTOC(tocData, tocNav);
                if (tocSidebar) tocSidebar.classList.remove('hidden');
            } else {
                if (tocSidebar) tocSidebar.classList.add('hidden');
            }

            // 显示进度条
            if (progressContainer) progressContainer.style.display = 'block';

            // 等待 KaTeX 加载完成后再渲染公式
            function renderMath() {
                if (typeof renderMathInElement !== 'undefined' && typeof katex !== 'undefined') {
                    try {
                        console.log('KaTeX 渲染开始...');
                        renderMathInElement(notePreview, {
                            delimiters: [
                                {left: '$$', right: '$$', display: true},
                                {left: '$', right: '$', display: false},
                                {left: '\\[', right: '\\]', display: true},
                                {left: '\\(', right: '\\)', display: false}
                            ],
                            throwOnError: false,
                            errorColor: '#cc0000',
                            strict: false,
                            trust: true
                        });
                        console.log('KaTeX 渲染完成');
                    } catch (e) {
                        console.warn('KaTeX 渲染失败:', e);
                    }
                } else {
                    // KaTeX 还没加载，稍后再试
                    console.log('等待 KaTeX 加载...');
                    setTimeout(renderMath, 100);
                }
            }
            renderMath();

            // 代码高亮
            notePreview.querySelectorAll('pre').forEach((pre) => {
                const code = pre.querySelector('code');
                if (code && window.hljs) {
                    // 添加语言标识
                    const langClass = Array.from(code.classList).find(c => c.startsWith('language-'));
                    if (langClass) {
                        const lang = langClass.replace('language-', '');
                        pre.setAttribute('data-lang', lang);
                    }
                    try {
                        hljs.highlightElement(code);
                    } catch (e) {
                        console.warn('代码高亮失败:', e);
                    }
                }
            });

            // 添加折叠功能
            addCollapsibleSections(notePreview);

            // 设置滚动监听
            setupScrollListener(notePreview, tocNav);

            // 平滑滚动到顶部
            notePreview.scrollTop = 0;
        })
        .catch(error => {
            console.error('加载笔记失败:', error);
            notePreview.innerHTML = `
                <div class="paper-detail">
                    <div class="detail-section" style="text-align: center; padding: 3rem;">
                        <div style="font-size: 3rem; margin-bottom: 1rem;">❌</div>
                        <h3>加载失败</h3>
                        <p style="color: #6b7280;">${error.message}</p>
                        <p style="color: #9ca3af; font-size: 0.9rem; margin-top: 1rem;">路径: ${notePath}</p>
                    </div>
                </div>
            `;
            if (tocSidebar) tocSidebar.classList.add('hidden');
            if (progressContainer) progressContainer.style.display = 'none';
        });
}

// ===== 目录生成 =====
function generateTOC(container) {
    const headings = container.querySelectorAll('.markdown-content h1, .markdown-content h2, .markdown-content h3, .markdown-content h4');
    const toc = [];

    headings.forEach((heading, index) => {
        // 生成唯一ID
        const id = `section-${index}`;
        heading.id = id;

        const level = parseInt(heading.tagName[1]);
        toc.push({
            level: level,
            title: heading.textContent.trim(),
            id: id
        });
    });

    return toc;
}

// ===== 渲染目录 =====
function renderTOC(tocData, tocNav) {
    let html = '';

    tocData.forEach(item => {
        // 截断过长的标题
        let title = item.title;
        if (title.length > 35) {
            title = title.substring(0, 35) + '...';
        }

        html += `
            <a class="toc-item level-${item.level}"
               href="#${item.id}"
               data-target="${item.id}"
               title="${item.title}">
                ${title}
            </a>
        `;
    });

    tocNav.innerHTML = html;

    // 绑定点击事件
    tocNav.querySelectorAll('.toc-item').forEach(item => {
        item.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('data-target');
            const notePreview = document.getElementById('notePreview');
            const targetElement = notePreview.querySelector(`#${targetId}`);

            if (targetElement) {
                // 平滑滚动
                notePreview.scrollTo({
                    top: targetElement.offsetTop - 20,
                    behavior: 'smooth'
                });

                // 更新活动状态
                tocNav.querySelectorAll('.toc-item').forEach(i => i.classList.remove('active'));
                this.classList.add('active');
            }
        });
    });
}

// ===== 滚动监听 =====
function setupScrollListener(container, tocNav) {
    const progressBar = document.getElementById('readingProgressBar');
    const backToTop = document.getElementById('backToTop');

    // 滚动事件处理
    const handleScroll = () => {
        // 更新进度条
        if (progressBar) {
            const scrollTop = container.scrollTop;
            const scrollHeight = container.scrollHeight - container.clientHeight;
            const progress = scrollHeight > 0 ? (scrollTop / scrollHeight) * 100 : 0;
            progressBar.style.width = `${progress}%`;
        }

        // 显示/隐藏返回顶部按钮
        if (backToTop) {
            if (container.scrollTop > 300) {
                backToTop.classList.add('visible');
            } else {
                backToTop.classList.remove('visible');
            }
        }

        // 更新目录高亮
        if (tocNav) {
            highlightActiveSection(container, tocNav);
        }
    };

    // 绑定滚动事件
    container.addEventListener('scroll', handleScroll);

    // 返回顶部按钮
    if (backToTop) {
        backToTop.onclick = () => {
            container.scrollTo({
                top: 0,
                behavior: 'smooth'
            });
        };
    }

    // 初始调用一次
    handleScroll();
}

// ===== 高亮当前章节 =====
function highlightActiveSection(container, tocNav) {
    const headings = container.querySelectorAll('.markdown-content h1[id], .markdown-content h2[id], .markdown-content h3[id], .markdown-content h4[id]');

    let activeId = null;
    const scrollTop = container.scrollTop + 50; // 偏移量

    // 找到当前可见的标题
    for (let i = headings.length - 1; i >= 0; i--) {
        const heading = headings[i];
        if (heading.offsetTop <= scrollTop) {
            activeId = heading.id;
            break;
        }
    }

    // 更新目录高亮
    tocNav.querySelectorAll('.toc-item').forEach(item => {
        if (item.getAttribute('data-target') === activeId) {
            item.classList.add('active');
            // 确保活动项可见
            item.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
        } else {
            item.classList.remove('active');
        }
    });
}

// ===== 添加折叠功能 =====
function addCollapsibleSections(container) {
    const h2Elements = container.querySelectorAll('.markdown-content h2');

    h2Elements.forEach(h2 => {
        // 收集h2下面的所有内容直到下一个h2
        const contentElements = [];
        let nextElement = h2.nextElementSibling;

        while (nextElement && !nextElement.matches('h2')) {
            contentElements.push(nextElement);
            nextElement = nextElement.nextElementSibling;
        }

        if (contentElements.length > 0) {
            // 创建包装器
            const wrapper = document.createElement('div');
            wrapper.className = 'section-content';
            wrapper.style.maxHeight = 'none';

            // 将内容移动到包装器中
            contentElements.forEach(el => {
                wrapper.appendChild(el);
            });

            // 插入包装器
            h2.insertAdjacentElement('afterend', wrapper);

            // 添加点击事件
            h2.addEventListener('click', function() {
                const isCollapsed = wrapper.classList.contains('collapsed');

                if (isCollapsed) {
                    // 展开
                    wrapper.style.maxHeight = wrapper.scrollHeight + 'px';
                    wrapper.classList.remove('collapsed');
                    h2.classList.remove('collapsed');
                    // 延迟移除maxHeight以允许自适应
                    setTimeout(() => {
                        if (!wrapper.classList.contains('collapsed')) {
                            wrapper.style.maxHeight = 'none';
                        }
                    }, 300);
                } else {
                    // 收起
                    wrapper.style.maxHeight = wrapper.scrollHeight + 'px';
                    // 触发重排
                    wrapper.offsetHeight;
                    wrapper.style.maxHeight = '0';
                    wrapper.classList.add('collapsed');
                    h2.classList.add('collapsed');
                }
            });
        }
    });
}

// 增强版 Markdown 转 HTML 函数（支持 LaTeX）
// 增强版 Markdown 转 HTML 函数
function markdownToHtml(markdown) {
    if (!markdown) return '';

    // 使用唯一占位符保护特殊内容
    const placeholders = {
        latexBlocks: [],
        latexInline: []
    };

    let text = markdown;

    // ===== 第1步：保护 LaTeX 公式 =====
    // 使用特殊的占位符格式，避免被 marked.js 处理
    const PLACEHOLDER_PREFIX = '___MATH_PLACEHOLDER_';
    const PLACEHOLDER_SUFFIX = '___';

    // 块级公式 $$...$$ - 必须独占一行
    text = text.replace(/^\$\$([\s\S]*?)\$\$$/gm, (match, formula) => {
        const idx = placeholders.latexBlocks.length;
        placeholders.latexBlocks.push(formula.trim());
        return `\n${PLACEHOLDER_PREFIX}BLOCK_${idx}${PLACEHOLDER_SUFFIX}\n`;
    });

    // 行内公式 $...$ (排除货币符号 - 要求$后不是空格或数字，且不含换行)
    text = text.replace(/\$([^\$\s\n][^\$\n]*?)\$/g, (match, formula) => {
        const idx = placeholders.latexInline.length;
        placeholders.latexInline.push(formula.trim());
        return `${PLACEHOLDER_PREFIX}INLINE_${idx}${PLACEHOLDER_SUFFIX}`;
    });

    // \( ... \) 格式
    text = text.replace(/\\\(([\s\S]*?)\\\)/g, (match, formula) => {
        const idx = placeholders.latexInline.length;
        placeholders.latexInline.push(formula.trim());
        return `${PLACEHOLDER_PREFIX}INLINE_${idx}${PLACEHOLDER_SUFFIX}`;
    });

    // \[ ... \] 格式
    text = text.replace(/\\\[([\s\S]*?)\\\]/g, (match, formula) => {
        const idx = placeholders.latexBlocks.length;
        placeholders.latexBlocks.push(formula.trim());
        return `\n${PLACEHOLDER_PREFIX}BLOCK_${idx}${PLACEHOLDER_SUFFIX}\n`;
    });

    console.log('LaTeX 保护完成，块级:', placeholders.latexBlocks.length, '行内:', placeholders.latexInline.length);

    let html;

    // ===== 第2步：使用 marked.js 或自定义解析器 =====
    if (typeof marked !== 'undefined' && marked.parse) {
        try {
            console.log('使用 marked.js 解析 Markdown');

            // 配置 marked.js
            const renderer = new marked.Renderer();

            // 自定义表格渲染
            renderer.table = function(header, body) {
                if (!body) body = '';
                return '<div class="table-wrapper"><table><thead>' + header + '</thead><tbody>' + body + '</tbody></table></div>';
            };

            // 自定义代码块渲染
            renderer.code = function(code, language) {
                const lang = language || 'plaintext';
                const escapedCode = code
                    .replace(/&/g, '&amp;')
                    .replace(/</g, '&lt;')
                    .replace(/>/g, '&gt;');
                return `<pre data-lang="${lang}"><code class="language-${lang}">${escapedCode}</code></pre>`;
            };

            // 自定义链接渲染
            renderer.link = function(href, title, text) {
                const titleAttr = title ? ` title="${title}"` : '';
                return `<a href="${href}" target="_blank" rel="noopener"${titleAttr}>${text}</a>`;
            };

            // 段落渲染 - 不包装占位符
            renderer.paragraph = function(text) {
                if (text.includes(PLACEHOLDER_PREFIX)) {
                    return text + '\n';
                }
                return '<p>' + text + '</p>\n';
            };

            // 解析 Markdown
            html = marked.parse(text, {
                breaks: true,
                gfm: true,
                renderer: renderer
            });

            console.log('marked.js 解析完成');
        } catch (e) {
            console.warn('marked.js 解析失败，使用自定义解析器:', e);
            html = parseMarkdownCustom(text);
        }
    } else {
        console.log('marked.js 不可用，使用自定义解析器');
        html = parseMarkdownCustom(text);
    }

    // ===== 第3步：恢复 LaTeX 公式 =====
    html = html.replace(new RegExp(PLACEHOLDER_PREFIX + 'BLOCK_(\\d+)' + PLACEHOLDER_SUFFIX, 'g'), (match, idx) => {
        const formula = placeholders.latexBlocks[parseInt(idx)];
        if (!formula) return match;
        return `$$${formula}$$`;
    });

    html = html.replace(new RegExp(PLACEHOLDER_PREFIX + 'INLINE_(\\d+)' + PLACEHOLDER_SUFFIX, 'g'), (match, idx) => {
        const formula = placeholders.latexInline[parseInt(idx)];
        if (!formula) return match;
        return `$${formula}$`;
    });

    // 清理可能的多余段落标签
    html = html.replace(/<p>\s*<div class="table-wrapper">/g, '<div class="table-wrapper">');
    html = html.replace(/<\/div>\s*<\/p>/g, '</div>');
    html = html.replace(/<p>\s*\$\$/g, '$$');
    html = html.replace(/\$\$\s*<\/p>/g, '$$');

    console.log('LaTeX 恢复完成');
    return html;
}

// 自定义 Markdown 解析器（当 marked.js 不可用时使用）
function parseMarkdownCustom(text) {
    console.log('使用自定义 Markdown 解析器');
    const placeholders = {
        codeBlocks: [],
        tables: []
    };

    const PLACEHOLDER_PREFIX = '___CUSTOM_PLACEHOLDER_';
    const PLACEHOLDER_SUFFIX = '___';

    // 保护代码块
    text = text.replace(/```(\w*)\n([\s\S]*?)```/g, (match, lang, code) => {
        const idx = placeholders.codeBlocks.length;
        placeholders.codeBlocks.push({ lang: lang || 'plaintext', code: code.trim() });
        return `${PLACEHOLDER_PREFIX}CODE_${idx}${PLACEHOLDER_SUFFIX}`;
    });

    // 保护表格 - 改进的正则表达式
    text = text.replace(/^\|(.+)\|\s*\n\|([:\-|\s]+)\|\s*\n((?:^\|.+\|\s*\n?)+)/gm, (match, headerLine, separatorLine, bodyLines) => {
        const idx = placeholders.tables.length;
        placeholders.tables.push({ header: headerLine, separator: separatorLine, body: bodyLines });
        return `${PLACEHOLDER_PREFIX}TABLE_${idx}${PLACEHOLDER_SUFFIX}`;
    });

    // HTML 转义
    text = text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');

    // 标题
    text = text.replace(/^###### (.+)$/gm, '<h6>$1</h6>');
    text = text.replace(/^##### (.+)$/gm, '<h5>$1</h5>');
    text = text.replace(/^#### (.+)$/gm, '<h4>$1</h4>');
    text = text.replace(/^### (.+)$/gm, '<h3>$1</h3>');
    text = text.replace(/^## (.+)$/gm, '<h2>$1</h2>');
    text = text.replace(/^# (.+)$/gm, '<h1>$1</h1>');

    // 分割线
    text = text.replace(/^(---|\*\*\*|___)$/gm, '<hr>');

    // 图片和链接
    text = text.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, '<img src="$2" alt="$1" style="max-width:100%;border-radius:8px;">');
    text = text.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');

    // 格式化 (注意顺序：先处理更复杂的模式)
    text = text.replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>');
    text = text.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    text = text.replace(/(?<!\*)\*([^*\n]+?)\*(?!\*)/g, '<em>$1</em>');
    text = text.replace(/~~(.+?)~~/g, '<del>$1</del>');
    text = text.replace(/`([^`\n]+)`/g, '<code>$1</code>');

    // 引用
    text = text.replace(/^&gt; (.+)$/gm, '<blockquote>$1</blockquote>');

    // 列表
    text = text.replace(/^[\t ]*[-*+]\s+(.+)$/gm, '<li>$1</li>');
    text = text.replace(/^[\t ]*\d+\.\s+(.+)$/gm, '<li>$1</li>');

    // 包裹列表
    text = text.replace(/(<li>.*?<\/li>\n?)+/g, '<ul>$&</ul>');

    // 段落
    let paragraphs = text.split(/\n\n+/);
    text = paragraphs.map(p => {
        p = p.trim();
        if (!p) return '';
        if (p.match(/^<(h[1-6]|ul|ol|blockquote|hr|pre|table|div|___)/)) return p;
        if (p.includes(PLACEHOLDER_PREFIX)) return p;
        p = p.replace(/\n/g, '<br>');
        return '<p>' + p + '</p>';
    }).join('\n');

    // 恢复代码块
    text = text.replace(new RegExp(PLACEHOLDER_PREFIX + 'CODE_(\\d+)' + PLACEHOLDER_SUFFIX, 'g'), (match, idx) => {
        const block = placeholders.codeBlocks[parseInt(idx)];
        if (!block) return match;
        const escapedCode = block.code
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');
        return `<pre data-lang="${block.lang}"><code class="language-${block.lang}">${escapedCode}</code></pre>`;
    });

    // 恢复表格
    text = text.replace(new RegExp(PLACEHOLDER_PREFIX + 'TABLE_(\\d+)' + PLACEHOLDER_SUFFIX, 'g'), (match, idx) => {
        const table = placeholders.tables[parseInt(idx)];
        if (!table) return match;

        // 解析表头
        const headers = table.header.split('|').map(h => h.trim()).filter(h => h);

        // 解析表体
        const rows = table.body.trim().split('\n').map(row => {
            return row.replace(/^\|/, '').replace(/\|$/, '').split('|').map(c => c.trim());
        }).filter(r => r.length > 0 && r.some(c => c));

        // 生成表格 HTML
        let html = '<div class="table-wrapper"><table><thead><tr>';
        headers.forEach(h => {
            let content = processInlineFormatting(h);
            html += `<th>${content}</th>`;
        });
        html += '</tr></thead><tbody>';

        rows.forEach(row => {
            html += '<tr>';
            for (let i = 0; i < headers.length; i++) {
                let cell = row[i] || '';
                let content = processInlineFormatting(cell);
                html += `<td>${content}</td>`;
            }
            html += '</tr>';
        });
        html += '</tbody></table></div>';
        return html;
    });

    return text;
}

// 处理行内格式化
function processInlineFormatting(text) {
    if (!text) return '';
    return text
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.+?)\*/g, '<em>$1</em>')
        .replace(/`(.+?)`/g, '<code>$1</code>');
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
