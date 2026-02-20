/**
 * Xiaohao Cai 论文精读系统 - 优化版
 * 主要改进：
 * 1. ES6+ 现代化语法
 * 2. 模块化架构
 * 3. 真正的 Markdown 渲染
 * 4. 完善的错误处理
 * 5. 性能优化（防抖、懒加载）
 * 6. 智能笔记路径匹配
 */

// ===== 配置常量 =====
const CONFIG = {
    ITEMS_PER_PAGE: 12,
    DEBOUNCE_DELAY: 300,
    ANIMATION_DURATION: 300,
    NOTE_BASE_PATH: '../xiaohao_cai_ultimate_notes_final/',
    PDF_BASE_PATH: '00_papers/'
};

// ===== 状态管理 =====
const state = {
    currentPage: 'dashboard',
    currentPaper: null,
    filteredPapers: [],
    currentPageNum: 1,
    charts: {},
    isInitialized: {
        network: false,
        timeline: false
    },
    searchQuery: ''
};

// ===== 工具函数 =====
const Utils = {
    /**
     * 防抖函数
     */
    debounce: (fn, delay) => {
        let timer = null;
        return (...args) => {
            clearTimeout(timer);
            timer = setTimeout(() => fn.apply(this, args), delay);
        };
    },

    /**
     * 智能获取笔记路径 - 支持多种文件名格式匹配
     */
    getNotePath: (paper) => {
        if (!paper) return null;
        
        // 优先使用 data.js 中定义的 noteFile
        if (paper.noteFile) {
            return CONFIG.NOTE_BASE_PATH + paper.noteFile;
        }
        
        // 智能匹配：根据论文标题尝试查找可能的笔记文件
        const possibleNames = [
            // 超精读笔记格式
            `${paper.title.replace(/[\s\[\]]/g, '_')}_超精读笔记_已填充.md`,
            `${paper.title.replace(/[\s\[\]]/g, '_')}_超精读笔记.md`,
            // 英文标题格式（如果有）
            paper.pdfFile?.replace('.pdf', '_超精读笔记_已填充.md'),
            // 简化版
            `${paper.title}.md`
        ].filter(Boolean);
        
        return possibleNames[0] || null;
    },

    getPDFPath: (filename) => {
        if (!filename) return null;
        return CONFIG.PDF_BASE_PATH + filename;
    },

    getCategoryColor: (category) => {
        const colors = {
            '基础理论': '#ef4444',
            '变分分割': '#3b82f6',
            '深度学习': '#10b981',
            '雷达与无线电': '#8b5cf6',
            '医学图像': '#f59e0b',
            '其他': '#6b7280'
        };
        return colors[category] || '#6b7280';
    },

    getCategoryClass: (category) => {
        const classes = {
            '基础理论': 'category-theory',
            '变分分割': 'category-segmentation',
            '深度学习': 'category-deep',
            '雷达与无线电': 'category-signal',
            '医学图像': 'category-medical',
            '其他': 'category-other'
        };
        return classes[category] || 'category-other';
    },

    /**
     * 简单的 Markdown 转 HTML
     * 支持：标题、列表、代码块、粗体、斜体、链接
     */
    markdownToHTML: (markdown) => {
        if (!markdown) return '';
        
        return markdown
            // 代码块
            .replace(/```(\w+)?\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>')
            // 行内代码
            .replace(/`([^`]+)`/g, '<code>$1</code>')
            // 标题
            .replace(/^### (.*$)/gim, '<h3>$1</h3>')
            .replace(/^## (.*$)/gim, '<h2>$1</h2>')
            .replace(/^# (.*$)/gim, '<h1>$1</h1>')
            // 粗体和斜体
            .replace(/\*\*\*(.*?)\*\*\*/g, '<strong><em>$1</em></strong>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            // 链接
            .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>')
            // 图片
            .replace(/!\[([^\]]*)\]\(([^)]+)\)/g, '<img src="$2" alt="$1" style="max-width:100%;">')
            // 无序列表
            .replace(/^\s*[-*+]\s+(.+)$/gim, '<li>$1</li>')
            // 有序列表
            .replace(/^\s*\d+\.\s+(.+)$/gim, '<li>$1</li>')
            // 引用块
            .replace(/^>\s*(.+)$/gim, '<blockquote>$1</blockquote>')
            // 分割线
            .replace(/^---+$/gim, '<hr>')
            // 表格（简化处理）
            .replace(/\|(.+)\|/g, (match, content) => {
                const cells = content.split('|').map(c => c.trim()).filter(Boolean);
                if (cells.length === 0) return '';
                return '<tr>' + cells.map(c => `<td>${c}</td>`).join('') + '</tr>';
            })
            // 段落（必须在最后）
            .replace(/\n\n/g, '</p><p>')
            .replace(/\n/g, '<br>')
            // 包裹段落
            .replace(/^(.+)$/gim, (match) => {
                if (match.startsWith('<')) return match;
                return `<p>${match}</p>`;
            })
            // 清理空标签
            .replace(/<p><\/p>/g, '')
            .replace(/<p>(<\w+>)/g, '$1')
            .replace(/(<\/\w+>)<\/p>/g, '$1');
    },

    /**
     * 显示加载状态
     */
    showLoading: (element, message = '加载中...') => {
        element.innerHTML = `
            <div class="loading-state">
                <div class="loading-spinner"></div>
                <p>${message}</p>
            </div>
        `;
    },

    /**
     * 显示错误状态
     */
    showError: (element, message) => {
        element.innerHTML = `
            <div class="error-state">
                <div class="error-icon">⚠️</div>
                <p><strong>加载失败</strong></p>
                <p>${message}</p>
            </div>
        `;
    },

    /**
     * 显示空状态
     */
    showEmpty: (element, message = '暂无内容') => {
        element.innerHTML = `
            <div class="empty-state">
                <div class="empty-icon">📭</div>
                <p>${message}</p>
            </div>
        `;
    }
};

// ===== 图表管理器 =====
const ChartManager = {
    /**
     * 销毁图表实例
     */
    dispose: (chartName) => {
        if (state.charts[chartName]) {
            state.charts[chartName].dispose?.();
            state.charts[chartName].destroy?.();
            delete state.charts[chartName];
        }
    },

    /**
     * 响应式调整
     */
    resize: () => {
        Object.values(state.charts).forEach(chart => {
            chart?.resize?.();
        });
    }
};

// ===== 页面初始化 =====
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 论文精读系统初始化...');
    
    try {
        initNavigation();
        initDashboard();
        initPapersPage();
        initModalEvents();
        initKeyboardShortcuts();
        
        console.log('✅ 初始化完成');
    } catch (error) {
        console.error('❌ 初始化失败:', error);
        showGlobalError('系统初始化失败，请刷新页面重试');
    }
});

// ===== 导航功能 =====
function initNavigation() {
    // 侧边栏导航
    const sidebarNav = document.querySelector('.sidebar-nav');
    if (sidebarNav) {
        sidebarNav.addEventListener('click', (e) => {
            const navItem = e.target.closest('.nav-item');
            if (!navItem) return;
            
            const page = navItem.dataset.page;
            if (page) {
                e.preventDefault();
                switchPage(page);
            }
        });
    }
    
    // 移动端菜单切换
    const menuToggle = document.getElementById('menuToggle');
    if (menuToggle) {
        menuToggle.addEventListener('click', () => {
            document.getElementById('sidebar')?.classList.toggle('open');
        });
    }
    
    // 全局搜索（带防抖）
    const globalSearch = document.getElementById('globalSearch');
    if (globalSearch) {
        globalSearch.addEventListener('input', 
            Utils.debounce((e) => {
                const query = e.target.value.trim().toLowerCase();
                if (query.length > 0) {
                    state.searchQuery = query;
                    searchPapers(query);
                } else {
                    state.searchQuery = '';
                    filterPapers();
                }
            }, CONFIG.DEBOUNCE_DELAY)
        );
    }
    
    // 点击外部关闭侧边栏
    document.addEventListener('click', (e) => {
        const sidebar = document.getElementById('sidebar');
        const menuToggle = document.getElementById('menuToggle');
        
        if (sidebar?.classList.contains('open') && 
            !sidebar.contains(e.target) && 
            !menuToggle?.contains(e.target)) {
            sidebar.classList.remove('open');
        }
    });
}

function initKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
        // ESC 关闭弹窗
        if (e.key === 'Escape') {
            closeModal();
        }
        
        // / 聚焦搜索框
        if (e.key === '/' && !e.target.matches('input, textarea')) {
            e.preventDefault();
            document.getElementById('globalSearch')?.focus();
        }
    });
}

function switchPage(page) {
    if (state.currentPage === page) return;
    
    console.log(`📄 切换到页面: ${page}`);
    state.currentPage = page;
    
    // 更新导航状态
    document.querySelectorAll('.nav-item').forEach(item => {
        item.classList.toggle('active', item.dataset.page === page);
    });
    
    // 更新页面标题
    const titles = {
        dashboard: '数据仪表盘',
        papers: '论文列表',
        network: '引用网络',
        timeline: '研究时间线'
    };
    const pageTitle = document.getElementById('pageTitle');
    if (pageTitle) {
        pageTitle.textContent = titles[page] || '';
        pageTitle.classList.add('fade-in');
        setTimeout(() => pageTitle.classList.remove('fade-in'), CONFIG.ANIMATION_DURATION);
    }
    
    // 显示/隐藏页面
    document.querySelectorAll('.page').forEach(p => {
        p.classList.toggle('active', p.id === page);
    });
    
    // 关闭移动端菜单
    document.getElementById('sidebar')?.classList.remove('open');
    
    // 初始化页面特定图表
    if (page === 'network' && !state.isInitialized.network) {
        setTimeout(initNetworkPage, 100);
    }
    if (page === 'timeline' && !state.isInitialized.timeline) {
        setTimeout(initTimelinePage, 100);
    }
    
    // 滚动到顶部
    document.querySelector('.content-container')?.scrollTo({ top: 0, behavior: 'smooth' });
}

// ===== 仪表盘 =====
function initDashboard() {
    const { summary, categories } = PAPERS_DATA;
    
    // 更新统计数据
    updateElement('totalPapers', summary.total);
    updateElement('completedPapers', summary.filled);
    updateElement('papersCountBadge', summary.total);
    updateElement('progressText', `${summary.filled}/${summary.total} 篇已完成`);
    
    const progressPercent = Math.round((summary.filled / summary.total) * 100);
    const progressPercentEl = document.querySelector('.progress-percent');
    if (progressPercentEl) {
        progressPercentEl.textContent = `${progressPercent}%`;
    }
    
    // 分类分布图
    initCategoryChart(categories);
    
    // 完成状态图
    initCompletionChart(summary);
    
    // 进度条
    initProgressBars(categories);
}

function updateElement(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
}

function initCategoryChart(categories) {
    const ctx = document.getElementById('categoryChart');
    if (!ctx || typeof Chart === 'undefined') return;
    
    const labels = Object.keys(categories);
    const data = labels.map(cat => categories[cat].count);
    const colors = labels.map(cat => categories[cat].color);
    
    ChartManager.dispose('category');
    
    state.charts.category = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels,
            datasets: [{
                data,
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
                    labels: { padding: 20, usePointStyle: true }
                },
                tooltip: {
                    callbacks: {
                        label: (context) => {
                            const label = context.label || '';
                            const value = context.raw || 0;
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

function initCompletionChart(summary) {
    const ctx = document.getElementById('completionChart');
    if (!ctx || typeof Chart === 'undefined') return;
    
    ChartManager.dispose('completion');
    
    state.charts.completion = new Chart(ctx, {
        type: 'pie',
        data: {
            labels: ['已完成', '待填充'],
            datasets: [{
                data: [summary.filled, summary.templates],
                backgroundColor: ['#10b981', '#f59e0b'],
                borderWidth: 2,
                borderColor: '#fff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'bottom' }
            }
        }
    });
}

function initProgressBars(categories) {
    const container = document.getElementById('categoryProgress');
    if (!container) return;
    
    container.innerHTML = '';
    
    Object.entries(categories).forEach(([name, data], index) => {
        const percentage = (data.filled / data.count * 100).toFixed(1);
        
        const item = document.createElement('div');
        item.className = 'progress-item';
        item.innerHTML = `
            <div class="progress-item-header">
                <span class="progress-label">
                    <span class="progress-color-dot" style="background:${data.color}"></span>
                    ${name}
                </span>
                <span class="progress-value">${data.filled}/${data.count} (${percentage}%)</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" style="width:0%;background:${data.color}"></div>
            </div>
        `;
        
        container.appendChild(item);
        
        // 动画效果
        setTimeout(() => {
            item.querySelector('.progress-fill').style.width = `${percentage}%`;
        }, 100 + index * 100);
    });
}

// ===== 论文列表页 =====
function initPapersPage() {
    state.filteredPapers = [...PAPERS_DATA.papers];
    
    // 绑定筛选事件
    document.getElementById('categoryFilter')?.addEventListener('change', filterPapers);
    document.getElementById('sortFilter')?.addEventListener('change', filterPapers);
    
    filterPapers();
}

function filterPapers() {
    let papers = [...PAPERS_DATA.papers];
    
    // 分类筛选
    const categoryFilter = document.getElementById('categoryFilter');
    if (categoryFilter?.value !== 'all') {
        papers = papers.filter(p => p.category === categoryFilter.value);
    }
    
    // 搜索筛选
    if (state.searchQuery) {
        const query = state.searchQuery.toLowerCase();
        papers = papers.filter(p => 
            p.title.toLowerCase().includes(query) || 
            p.id.toLowerCase().includes(query) ||
            p.category.toLowerCase().includes(query)
        );
    }
    
    // 排序
    const sortFilter = document.getElementById('sortFilter');
    const sortBy = sortFilter?.value || 'id';
    
    papers.sort((a, b) => {
        if (sortBy === 'year') return (b.year || 0) - (a.year || 0);
        if (sortBy === 'category') return a.category.localeCompare(b.category);
        return a.id.localeCompare(b.id);
    });
    
    state.filteredPapers = papers;
    state.currentPageNum = 1;
    renderPapers();
}

function searchPapers(query) {
    switchPage('papers');
    state.searchQuery = query.toLowerCase();
    filterPapers();
}

function renderPapers() {
    const grid = document.getElementById('papersGrid');
    if (!grid) return;
    
    const start = (state.currentPageNum - 1) * CONFIG.ITEMS_PER_PAGE;
    const end = start + CONFIG.ITEMS_PER_PAGE;
    const pagePapers = state.filteredPapers.slice(start, end);
    
    if (pagePapers.length === 0) {
        Utils.showEmpty(grid, '没有找到匹配的论文');
        renderPagination();
        return;
    }
    
    grid.innerHTML = pagePapers.map(paper => `
        <article class="paper-card ${Utils.getCategoryClass(paper.category)}" 
                 onclick="openPaperModalById('${paper.id}')"
                 data-category="${paper.category}"
                 data-year="${paper.year}">
            <div class="paper-card-header">
                <span class="paper-id-badge">[${paper.id}]</span>
                <span class="paper-status ${paper.status}" title="${paper.status === 'filled' ? '已完成' : '待填充'}">
                    ${paper.status === 'filled' ? '✓' : '○'}
                </span>
            </div>
            <h3 class="paper-title">${escapeHtml(paper.title)}</h3>
            <div class="paper-footer">
                <span class="badge ${Utils.getCategoryClass(paper.category)}">${paper.category}</span>
                <span class="paper-year">${paper.year}</span>
                ${paper.noteFile ? '<span class="note-indicator" title="有精读笔记">📝</span>' : ''}
            </div>
        </article>
    `).join('');
    
    renderPagination();
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function renderPagination() {
    const container = document.getElementById('pagination');
    if (!container) return;
    
    const totalPages = Math.ceil(state.filteredPapers.length / CONFIG.ITEMS_PER_PAGE);
    
    if (totalPages <= 1) {
        container.innerHTML = '';
        return;
    }
    
    let html = '';
    
    // 上一页
    html += `<button class="page-btn ${state.currentPageNum === 1 ? 'disabled' : ''}" 
                     onclick="goToPage(${state.currentPageNum - 1})" 
                     ${state.currentPageNum === 1 ? 'disabled' : ''}>←</button>`;
    
    // 页码
    for (let i = 1; i <= totalPages; i++) {
        if (i === 1 || i === totalPages || (i >= state.currentPageNum - 2 && i <= state.currentPageNum + 2)) {
            html += `<button class="page-btn ${i === state.currentPageNum ? 'active' : ''}" 
                             onclick="goToPage(${i})">${i}</button>`;
        } else if (i === state.currentPageNum - 3 || i === state.currentPageNum + 3) {
            html += `<span class="page-ellipsis">...</span>`;
        }
    }
    
    // 下一页
    html += `<button class="page-btn ${state.currentPageNum === totalPages ? 'disabled' : ''}" 
                     onclick="goToPage(${state.currentPageNum + 1})" 
                     ${state.currentPageNum === totalPages ? 'disabled' : ''}>→</button>`;
    
    container.innerHTML = html;
}

function goToPage(n) {
    const totalPages = Math.ceil(state.filteredPapers.length / CONFIG.ITEMS_PER_PAGE);
    if (n < 1 || n > totalPages) return;
    
    state.currentPageNum = n;
    renderPapers();
    
    // 滚动到列表顶部
    document.getElementById('papers')?.scrollIntoView({ behavior: 'smooth' });
}

function openPaperModalById(id) {
    const paper = PAPERS_DATA.papers.find(p => p.id === id);
    if (paper) openPaperModal(paper);
}

// ===== 弹窗功能 =====
function initModalEvents() {
    document.querySelector('.modal-close')?.addEventListener('click', closeModal);
    document.querySelector('.modal-overlay')?.addEventListener('click', closeModal);
    document.getElementById('btnReadNote')?.addEventListener('click', openNote);
    document.getElementById('btnViewPDF')?.addEventListener('click', openPDF);
}

function openPaperModal(paper) {
    state.currentPaper = paper;
    
    updateElement('modalPaperId', `[${paper.id}]`);
    updateElement('modalPaperTitle', paper.title);
    
    const catBadge = document.getElementById('modalPaperCategory');
    if (catBadge) {
        catBadge.textContent = paper.category;
        catBadge.className = `badge ${Utils.getCategoryClass(paper.category)}`;
    }
    
    updateElement('modalPaperYear', paper.year);
    
    const statusBadge = document.getElementById('modalPaperStatus');
    if (statusBadge) {
        const isFilled = paper.status === 'filled';
        statusBadge.textContent = isFilled ? '✓ 已完成' : '○ 待填充';
        statusBadge.className = `badge ${isFilled ? 'success' : 'warning'}`;
    }
    
    const notePreview = document.getElementById('notePreview');
    if (notePreview) {
        notePreview.innerHTML = `
            <div class="placeholder">
                <div class="placeholder-icon">📖</div>
                <p>点击上方"查看精读笔记"加载内容</p>
            </div>
        `;
    }
    
    // 更新按钮状态
    const hasNote = Boolean(Utils.getNotePath(paper));
    const btnReadNote = document.getElementById('btnReadNote');
    if (btnReadNote) {
        btnReadNote.disabled = !hasNote;
        btnReadNote.style.opacity = hasNote ? '1' : '0.5';
        btnReadNote.title = hasNote ? '查看精读笔记' : '暂无笔记';
    }
    
    const hasPDF = paper.hasPDF && paper.pdfFile;
    const btnViewPDF = document.getElementById('btnViewPDF');
    if (btnViewPDF) {
        btnViewPDF.disabled = !hasPDF;
        btnViewPDF.style.opacity = hasPDF ? '1' : '0.5';
        btnViewPDF.title = hasPDF ? '查看论文原文' : '暂无PDF';
    }
    
    document.getElementById('paperModal')?.classList.add('active');
    document.body.style.overflow = 'hidden';
}

function closeModal() {
    document.getElementById('paperModal')?.classList.remove('active');
    document.body.style.overflow = '';
    state.currentPaper = null;
}

async function openNote() {
    const paper = state.currentPaper;
    if (!paper) return;
    
    const notePath = Utils.getNotePath(paper);
    if (!notePath) {
        Utils.showError(document.getElementById('notePreview'), '该论文暂无精读笔记');
        return;
    }
    
    const preview = document.getElementById('notePreview');
    Utils.showLoading(preview, '正在加载笔记...');
    
    try {
        const response = await fetch(notePath);
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const markdown = await response.text();
        const html = Utils.markdownToHTML(markdown);
        
        preview.innerHTML = `<div class="markdown-body">${html}</div>`;
        
        // 添加语法高亮（如果有 Prism.js）
        if (window.Prism) {
            preview.querySelectorAll('code').forEach(block => {
                Prism.highlightElement(block);
            });
        }
        
    } catch (error) {
        console.error('加载笔记失败:', error);
        Utils.showError(preview, 
            `无法加载笔记文件<br><small>${notePath}</small><br><small>${error.message}</small>`
        );
    }
}

function openPDF() {
    const paper = state.currentPaper;
    if (!paper?.pdfFile) {
        alert('该论文暂无PDF文件');
        return;
    }
    
    const pdfPath = Utils.getPDFPath(paper.pdfFile);
    window.open(pdfPath, '_blank', 'noopener,noreferrer');
}

// ===== 引用网络页 =====
function initNetworkPage() {
    if (state.isInitialized.network) return;
    
    const chartDom = document.getElementById('networkChart');
    if (!chartDom || typeof echarts === 'undefined') {
        Utils.showError(chartDom, '图表库加载失败，请刷新页面重试');
        return;
    }
    
    try {
        ChartManager.dispose('network');
        
        state.charts.network = echarts.init(chartDom);
        state.isInitialized.network = true;
        
        const colors = {};
        Object.entries(PAPERS_DATA.categories).forEach(([cat, data]) => {
            colors[cat] = data.color;
        });
        
        // 构建节点
        const nodeMap = new Set();
        const nodes = PAPERS_DATA.papers
            .filter(p => p.id)
            .map(p => {
                nodeMap.add(p.id);
                return {
                    name: p.id,
                    value: p.id,
                    symbolSize: 20 + (Math.random() * 10), // 随机大小增加视觉区分
                    itemStyle: { color: colors[p.category] || '#999' },
                    paper: p,
                    label: {
                        show: true,
                        formatter: p.id
                    }
                };
            });
        
        // 构建链接
        const links = PAPERS_DATA.citations
            ?.filter(c => nodeMap.has(c.source) && nodeMap.has(c.target))
            .map(c => ({
                source: c.source,
                target: c.target,
                value: c.strength || 1,
                lineStyle: {
                    width: Math.max(1, c.strength || 1)
                }
            })) || [];
        
        const option = {
            tooltip: {
                trigger: 'item',
                formatter: (params) => {
                    if (params.dataType === 'node' && params.data.paper) {
                        const p = params.data.paper;
                        return `
                            <div style="padding:8px;">
                                <strong style="color:${colors[p.category]}">[${p.id}]</strong><br/>
                                ${p.title}<br/>
                                <small>${p.category} · ${p.year}</small>
                            </div>
                        `;
                    }
                    return `${params.name}`;
                }
            },
            series: [{
                type: 'graph',
                layout: 'force',
                data: nodes,
                links: links,
                roam: true,
                draggable: true,
                focusNodeAdjacency: true,
                force: {
                    repulsion: 300,
                    gravity: 0.1,
                    edgeLength: 100
                },
                emphasis: {
                    focus: 'adjacency',
                    lineStyle: {
                        width: 4
                    }
                }
            }]
        };
        
        state.charts.network.setOption(option);
        
        // 点击事件
        state.charts.network.on('click', (params) => {
            if (params.data?.paper) {
                openPaperModal(params.data.paper);
            }
        });
        
        // 窗口调整
        window.addEventListener('resize', () => state.charts.network?.resize());
        
    } catch (error) {
        console.error('网络图初始化失败:', error);
        Utils.showError(chartDom, `图表加载失败: ${error.message}`);
    }
}

// ===== 时间线页 =====
function initTimelinePage() {
    if (state.isInitialized.timeline) return;
    
    const chartDom = document.getElementById('timelineChart');
    if (!chartDom || typeof echarts === 'undefined') {
        Utils.showError(chartDom, '图表库加载失败，请刷新页面重试');
        return;
    }
    
    try {
        ChartManager.dispose('timeline');
        
        state.charts.timeline = echarts.init(chartDom);
        state.isInitialized.timeline = true;
        
        const colors = {};
        Object.entries(PAPERS_DATA.categories).forEach(([cat, data]) => {
            colors[cat] = data.color;
        });
        
        // 过滤有效年份的论文
        const validPapers = PAPERS_DATA.papers
            .filter(p => p.year && p.year !== 'unknown' && !isNaN(parseInt(p.year)))
            .sort((a, b) => parseInt(a.year) - parseInt(b.year));
        
        if (validPapers.length === 0) {
            Utils.showEmpty(chartDom, '暂无可显示的时间线数据');
            return;
        }
        
        // 获取唯一年份
        const years = [...new Set(validPapers.map(p => p.year))].sort();
        
        // 按年份分组
        const yearGroups = validPapers.reduce((acc, p) => {
            acc[p.year] = acc[p.year] || [];
            acc[p.year].push(p);
            return acc;
        }, {});
        
        // 构建数据点
        const data = validPapers.map(paper => ({
            value: [years.indexOf(paper.year), yearGroups[paper.year].indexOf(paper)],
            paper,
            itemStyle: { color: colors[paper.category] || '#999' },
            symbolSize: 16
        }));
        
        const maxInYear = Math.max(...Object.values(yearGroups).map(g => g.length));
        
        const option = {
            tooltip: {
                trigger: 'item',
                backgroundColor: 'rgba(255, 255, 255, 0.95)',
                borderColor: '#2563eb',
                borderWidth: 1,
                textStyle: { color: '#1f2937', fontSize: 13 },
                extraCssText: 'box-shadow: 0 4px 12px rgba(0,0,0,0.15); border-radius: 8px; padding: 12px;',
                formatter: (params) => {
                    if (params.data?.paper) {
                        const p = params.data.paper;
                        return `
                            <div style="margin-bottom:6px;">
                                <span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${colors[p.category]};margin-right:6px;"></span>
                                <strong style="color:${colors[p.category]}">${p.category}</strong>
                            </div>
                            <div style="font-size:14px;font-weight:600;margin-bottom:4px;">${p.title}</div>
                            <div style="color:#6b7280;font-size:12px;">[${p.id}] 年份: ${p.year}</div>
                        `;
                    }
                    return '';
                }
            },
            legend: {
                data: Object.keys(colors),
                orient: 'vertical',
                right: '2%',
                top: 'center',
                itemGap: 15,
                textStyle: { fontSize: 12, color: '#374151' }
            },
            grid: { left: '8%', right: '18%', bottom: '15%', top: '8%' },
            xAxis: {
                type: 'category',
                data: years,
                name: '年份',
                nameLocation: 'middle',
                nameGap: 35,
                nameTextStyle: { fontSize: 13, fontWeight: 600 },
                axisLabel: { rotate: 30, fontSize: 11, color: '#4b5563' },
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
                data,
                symbolSize: 16,
                itemStyle: { opacity: 0.85, borderWidth: 2, borderColor: '#fff' },
                emphasis: {
                    itemStyle: { opacity: 1, borderColor: '#1f2937', borderWidth: 2, shadowBlur: 10, shadowColor: 'rgba(0,0,0,0.3)' },
                    scale: 1.3
                }
            }]
        };
        
        state.charts.timeline.setOption(option);
        
        // 点击事件
        state.charts.timeline.on('click', (params) => {
            if (params.data?.paper) openPaperModal(params.data.paper);
        });
        
        // 筛选按钮
        initTimelineFilters(data, colors);
        
        // 窗口调整
        window.addEventListener('resize', () => state.charts.timeline?.resize());
        
    } catch (error) {
        console.error('时间线初始化失败:', error);
        Utils.showError(chartDom, `图表加载失败: ${error.message}`);
    }
}

function initTimelineFilters(data, colors) {
    const filterButtons = document.querySelectorAll('.timeline-filters .filter-btn');
    
    filterButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            // 更新激活状态
            filterButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            // 筛选数据
            const filter = btn.dataset.filter;
            const filteredData = filter === 'all' 
                ? data 
                : data.filter(d => d.paper?.category === filter);
            
            state.charts.timeline.setOption({ series: [{ data: filteredData }] });
        });
    });
}

// ===== 全局错误处理 =====
function showGlobalError(message) {
    const container = document.querySelector('.content-container');
    if (container) {
        container.innerHTML = `
            <div class="global-error">
                <div class="error-icon">⚠️</div>
                <h2>出错了</h2>
                <p>${message}</p>
                <button onclick="location.reload()" class="btn btn-primary">刷新页面</button>
            </div>
        `;
    }
}

// 全局错误监听
window.addEventListener('error', (e) => {
    console.error('全局错误:', e.error);
});

window.addEventListener('unhandledrejection', (e) => {
    console.error('未处理的 Promise 错误:', e.reason);
});
