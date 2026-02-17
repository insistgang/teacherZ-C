// Main Application
let currentPage = 'dashboard';
let currentPaper = null;
let charts = {};

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    console.log('Initializing...');
    initNavigation();
    initDashboard();
    initPapersPage();
    initModalEvents();
});

// ===== Navigation =====
function initNavigation() {
    // Sidebar navigation
    var sidebarNav = document.querySelector('.sidebar-nav');
    if (sidebarNav) {
        sidebarNav.addEventListener('click', function(e) {
            var navItem = e.target.closest('.nav-item');
            if (navItem) {
                var page = navItem.getAttribute('data-page');
                if (page) {
                    e.preventDefault();
                    switchPage(page);
                }
            }
        });
    }
    
    // Mobile menu toggle
    var menuToggle = document.getElementById('menuToggle');
    if (menuToggle) {
        menuToggle.addEventListener('click', function() {
            var sidebar = document.getElementById('sidebar');
            if (sidebar) sidebar.classList.toggle('open');
        });
    }
    
    // Global search
    var globalSearch = document.getElementById('globalSearch');
    if (globalSearch) {
        globalSearch.addEventListener('input', function(e) {
            var query = e.target.value.toLowerCase();
            if (query.length > 2) {
                searchPapers(query);
            }
        });
    }
}

function initModalEvents() {
    var modalClose = document.querySelector('.modal-close');
    if (modalClose) modalClose.addEventListener('click', closeModal);
    
    var modalOverlay = document.querySelector('.modal-overlay');
    if (modalOverlay) modalOverlay.addEventListener('click', closeModal);
    
    var btnReadNote = document.getElementById('btnReadNote');
    if (btnReadNote) btnReadNote.addEventListener('click', openNote);
    
    var btnViewPDF = document.getElementById('btnViewPDF');
    if (btnViewPDF) btnViewPDF.addEventListener('click', openPDF);
}

function switchPage(page) {
    console.log('Switching to page: ' + page);
    currentPage = page;
    
    // Update navigation active state
    var navItems = document.querySelectorAll('.nav-item');
    for (var i = 0; i < navItems.length; i++) {
        var item = navItems[i];
        if (item.getAttribute('data-page') === page) {
            item.classList.add('active');
        } else {
            item.classList.remove('active');
        }
    }
    
    // Update page title
    var titles = {
        dashboard: '数据仪表盘',
        papers: '论文列表',
        network: '引用网络',
        timeline: '研究时间线'
    };
    var pageTitle = document.getElementById('pageTitle');
    if (pageTitle && titles[page]) {
        pageTitle.textContent = titles[page];
    }
    
    // Show/hide pages
    var pages = document.querySelectorAll('.page');
    for (var j = 0; j < pages.length; j++) {
        var p = pages[j];
        if (p.id === page) {
            p.classList.add('active');
        } else {
            p.classList.remove('active');
        }
    }
    
    // Close mobile menu
    var sidebar = document.getElementById('sidebar');
    if (sidebar) sidebar.classList.remove('open');
    
    // Initialize charts for specific pages
    if (page === 'network') {
        setTimeout(initNetworkPage, 100);
    }
    if (page === 'timeline') {
        setTimeout(initTimelinePage, 100);
    }
}

// ===== Dashboard =====
function initDashboard() {
    // Category Chart
    var categoryCtx = document.getElementById('categoryChart');
    if (categoryCtx && typeof Chart !== 'undefined') {
        var categoryLabels = [];
        var categoryData = [];
        var categoryColors = ['#ef4444', '#3b82f6', '#10b981', '#8b5cf6', '#f59e0b', '#6b7280'];
        
        var i = 0;
        for (var cat in PAPERS_DATA.categories) {
            categoryLabels.push(cat);
            categoryData.push(PAPERS_DATA.categories[cat].count);
            i++;
        }
        
        charts.category = new Chart(categoryCtx, {
            type: 'doughnut',
            data: {
                labels: categoryLabels,
                datasets: [{
                    data: categoryData,
                    backgroundColor: categoryColors,
                    borderWidth: 2,
                    borderColor: '#fff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    }
    
    // Completion Chart
    var completionCtx = document.getElementById('completionChart');
    if (completionCtx && typeof Chart !== 'undefined') {
        charts.completion = new Chart(completionCtx, {
            type: 'pie',
            data: {
                labels: ['已完成', '待填充'],
                datasets: [{
                    data: [PAPERS_DATA.summary.filled, PAPERS_DATA.summary.templates],
                    backgroundColor: ['#10b981', '#f59e0b'],
                    borderWidth: 2,
                    borderColor: '#fff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false
            }
        });
    }
    
    // Progress bars
    var progressContainer = document.getElementById('categoryProgress');
    if (progressContainer) {
        var colors = {
            '基础理论': '#ef4444',
            '变分分割': '#3b82f6',
            '深度学习': '#10b981',
            '雷达与无线电': '#8b5cf6',
            '医学图像': '#f59e0b',
            '其他': '#6b7280'
        };
        
        for (var catName in PAPERS_DATA.categories) {
            var data = PAPERS_DATA.categories[catName];
            var percentage = (data.filled / data.count * 100).toFixed(1);
            var color = colors[catName] || '#6b7280';
            
            var item = document.createElement('div');
            item.className = 'progress-item';
            item.innerHTML = 
                '<div class="progress-item-header">' +
                    '<span class="progress-label"><span class="progress-color-dot" style="background:' + color + '"></span>' + catName + '</span>' +
                    '<span class="progress-value">' + data.filled + '/' + data.count + ' (' + percentage + '%)</span>' +
                '</div>' +
                '<div class="progress-bar"><div class="progress-fill" style="width:0%;background:' + color + '"></div></div>';
            progressContainer.appendChild(item);
            
            // Animate progress bar
            (function(el, width) {
                setTimeout(function() {
                    el.querySelector('.progress-fill').style.width = width + '%';
                }, 300);
            })(item, percentage);
        }
    }
}

// ===== Papers Page =====
var filteredPapers = [];
var currentPageNum = 1;
var itemsPerPage = 12;

function initPapersPage() {
    // Initialize with all papers
    filteredPapers = PAPERS_DATA.papers.slice();
    
    var categoryFilter = document.getElementById('categoryFilter');
    var statusFilter = document.getElementById('statusFilter');
    var sortFilter = document.getElementById('sortFilter');
    
    if (categoryFilter) categoryFilter.addEventListener('change', filterPapers);
    if (statusFilter) statusFilter.addEventListener('change', filterPapers);
    if (sortFilter) sortFilter.addEventListener('change', filterPapers);
    
    filterPapers();
}

function filterPapers() {
    var papers = PAPERS_DATA.papers.slice();
    
    var categoryFilter = document.getElementById('categoryFilter');
    if (categoryFilter && categoryFilter.value !== 'all') {
        papers = papers.filter(function(p) { return p.category === categoryFilter.value; });
    }
    
    var statusFilter = document.getElementById('statusFilter');
    if (statusFilter && statusFilter.value !== 'all') {
        papers = papers.filter(function(p) { return p.status === statusFilter.value; });
    }
    
    var sortFilter = document.getElementById('sortFilter');
    var sortBy = sortFilter ? sortFilter.value : 'id';
    
    papers.sort(function(a, b) {
        if (sortBy === 'year') return (b.year || 0) - (a.year || 0);
        if (sortBy === 'category') return a.category.localeCompare(b.category);
        return a.id.localeCompare(b.id);
    });
    
    filteredPapers = papers;
    currentPageNum = 1;
    renderPapers();
}

function renderPapers() {
    var grid = document.getElementById('papersGrid');
    if (!grid) return;
    
    var start = (currentPageNum - 1) * itemsPerPage;
    var end = start + itemsPerPage;
    var pagePapers = filteredPapers.slice(start, end);
    
    var html = '';
    for (var i = 0; i < pagePapers.length; i++) {
        var paper = pagePapers[i];
        var categoryClass = Utils.getCategoryClass(paper.category);
        html += 
            '<div class="paper-card ' + categoryClass + '" onclick="openPaperModalById(\'' + paper.id + '\')">' +
                '<div class="paper-card-header">' +
                    '<span class="paper-id-badge">[' + paper.id + ']</span>' +
                    '<span class="paper-status ' + paper.status + '">' + (paper.status === 'filled' ? '✓' : '○') + '</span>' +
                '</div>' +
                '<div class="paper-title">' + paper.title + '</div>' +
                '<div class="paper-footer">' +
                    '<span class="badge ' + categoryClass + '">' + paper.category + '</span>' +
                    '<span class="paper-year">' + paper.year + '</span>' +
                '</div>' +
            '</div>';
    }
    
    grid.innerHTML = html;
    renderPagination();
}

function renderPagination() {
    var container = document.getElementById('pagination');
    if (!container) return;
    
    var totalPages = Math.ceil(filteredPapers.length / itemsPerPage);
    if (totalPages <= 1) {
        container.innerHTML = '';
        return;
    }
    
    var html = '';
    for (var i = 1; i <= totalPages; i++) {
        html += '<button class="page-btn ' + (i === currentPageNum ? 'active' : '') + '" onclick="goToPage(' + i + ')">' + i + '</button>';
    }
    container.innerHTML = html;
}

function goToPage(n) {
    currentPageNum = n;
    renderPapers();
}

function openPaperModalById(id) {
    var paper = null;
    for (var i = 0; i < PAPERS_DATA.papers.length; i++) {
        if (PAPERS_DATA.papers[i].id === id) {
            paper = PAPERS_DATA.papers[i];
            break;
        }
    }
    if (paper) openPaperModal(paper);
}

function searchPapers(query) {
    switchPage('papers');
    filteredPapers = PAPERS_DATA.papers.filter(function(p) {
        return p.title.toLowerCase().includes(query) || p.id.toLowerCase().includes(query);
    });
    currentPageNum = 1;
    renderPapers();
}

// ===== Modal =====
function openPaperModal(paper) {
    currentPaper = paper;
    
    document.getElementById('modalPaperId').textContent = '[' + paper.id + ']';
    document.getElementById('modalPaperTitle').textContent = paper.title;
    
    var catBadge = document.getElementById('modalPaperCategory');
    catBadge.textContent = paper.category;
    catBadge.className = 'badge ' + Utils.getCategoryClass(paper.category);
    
    document.getElementById('modalPaperYear').textContent = paper.year;
    
    var statusBadge = document.getElementById('modalPaperStatus');
    statusBadge.textContent = paper.status === 'filled' ? '✓ 已完成' : '○ 待填充';
    statusBadge.className = 'badge ' + (paper.status === 'filled' ? 'success' : 'warning');
    
    document.getElementById('notePreview').innerHTML = '<div class="placeholder" style="text-align:center;padding:4rem 2rem"><div style="font-size:3rem;margin-bottom:1rem">📖</div><p>点击上方"查看精读笔记"加载内容</p></div>';
    
    var hasNote = paper.noteFile && paper.noteFile.trim() !== '';
    var btnReadNote = document.getElementById('btnReadNote');
    btnReadNote.disabled = !hasNote;
    btnReadNote.style.opacity = hasNote ? '1' : '0.5';
    
    var hasPDF = paper.pdfFile && paper.pdfFile.trim() !== '';
    var btnViewPDF = document.getElementById('btnViewPDF');
    btnViewPDF.disabled = !hasPDF;
    btnViewPDF.style.opacity = hasPDF ? '1' : '0.5';
    
    document.getElementById('paperModal').classList.add('active');
}

function closeModal() {
    document.getElementById('paperModal').classList.remove('active');
    currentPaper = null;
}

function openNote() {
    if (!currentPaper || !currentPaper.noteFile) return;
    
    var preview = document.getElementById('notePreview');
    preview.innerHTML = '<div class="placeholder" style="padding:3rem">加载中...</div>';
    
    fetch(Utils.getNotePath(currentPaper.noteFile))
        .then(function(response) {
            if (!response.ok) throw new Error('Failed');
            return response.text();
        })
        .then(function(markdown) {
            var html = markdown.replace(/\n/g, '<br>');
            preview.innerHTML = '<div class="markdown-body">' + html + '</div>';
        })
        .catch(function(err) {
            preview.innerHTML = '<div style="padding:1.5rem;background:#fef3c7;border-radius:8px"><p><strong>⚠️ 无法加载笔记</strong></p><p>' + err.message + '</p></div>';
        });
}

function openPDF() {
    if (!currentPaper || !currentPaper.pdfFile) {
        alert('该论文暂无PDF文件');
        return;
    }
    window.open('/00_papers/' + encodeURIComponent(currentPaper.pdfFile), '_blank');
}

// ===== Network Page =====
var networkInited = false;
function initNetworkPage() {
    if (networkInited) return;
    
    var chartDom = document.getElementById('networkChart');
    if (!chartDom) return;
    
    if (typeof echarts === 'undefined') {
        chartDom.innerHTML = '<div style="padding:40px;text-align:center"><h3>⚠️ 图表库加载失败</h3><p>请刷新页面重试</p></div>';
        return;
    }
    
    try {
        console.log('Initializing network chart...');
        charts.network = echarts.init(chartDom);
        networkInited = true;
        
        console.log('Building nodes...');
        var colors = {
            '基础理论': '#ef4444',
            '变分分割': '#3b82f6',
            '深度学习': '#10b981',
            '雷达与无线电': '#8b5cf6',
            '医学图像': '#f59e0b',
            '其他': '#6b7280'
        };
        
        // Build nodes array with unique IDs
        var nodeMap = {};
        var nodes = [];
        for (var i = 0; i < PAPERS_DATA.papers.length; i++) {
            var p = PAPERS_DATA.papers[i];
            if (p.id && !nodeMap[p.id]) {
                nodeMap[p.id] = true;
                nodes.push({
                    name: String(p.id),  // Ensure string
                    value: String(p.id),
                    symbolSize: 20,
                    itemStyle: { color: colors[p.category] || '#999' },
                    paper: p
                });
            }
        }
        
        // Build valid links
        var links = [];
        for (var k = 0; k < PAPERS_DATA.citations.length; k++) {
            var c = PAPERS_DATA.citations[k];
            var src = String(c.source);
            var tgt = String(c.target);
            if (nodeMap[src] && nodeMap[tgt]) {
                links.push({ 
                    source: src, 
                    target: tgt,
                    value: c.strength || 1
                });
            }
        }
        
        console.log('Nodes:', nodes.length, 'Links:', links.length);
        
        if (nodes.length === 0) {
            chartDom.innerHTML = '<div style="padding:40px;text-align:center"><p>暂无节点数据</p></div>';
            return;
        }
        
        // Ultra simple config
        var option = {
            tooltip: {
                trigger: 'item'
            },
            series: [{
                type: 'graph',
                layout: 'force',
                data: nodes,
                links: links,
                roam: true,
                label: { show: true },
                force: { repulsion: 300 }
            }]
        };
        
        console.log('Setting option:', JSON.stringify(option.series[0].data.length), 'nodes,', JSON.stringify(option.series[0].links.length), 'links');
        charts.network.setOption(option);
        
        charts.network.on('click', function(params) {
            if (params.data && params.data.paper) {
                openPaperModal(params.data.paper);
            }
        });
        
    } catch (e) {
        console.error('Network chart error:', e);
        chartDom.innerHTML = '<div style="padding:40px;text-align:center;color:red"><h3>图表加载失败</h3><p>' + e.message + '</p></div>';
    }
}

// ===== Timeline Page =====
var timelineInited = false;
function initTimelinePage() {
    if (timelineInited) return;
    
    var chartDom = document.getElementById('timelineChart');
    if (!chartDom) return;
    
    if (typeof echarts === 'undefined') {
        chartDom.innerHTML = '<div style="padding:40px;text-align:center"><h3>⚠️ 图表库加载失败</h3><p>请刷新页面重试</p></div>';
        return;
    }
    
    try {
        charts.timeline = echarts.init(chartDom);
        timelineInited = true;
        
        var colors = {
            '基础理论': '#ef4444',
            '变分分割': '#3b82f6',
            '深度学习': '#10b981',
            '雷达与无线电': '#8b5cf6',
            '医学图像': '#f59e0b',
            '其他': '#6b7280'
        };
        
        // Filter papers with valid years
        var sorted = [];
        for (var i = 0; i < PAPERS_DATA.papers.length; i++) {
            var p = PAPERS_DATA.papers[i];
            if (p.year && p.year !== 'unknown' && !isNaN(parseInt(p.year))) {
                sorted.push(p);
            }
        }
        sorted.sort(function(a, b) { return parseInt(a.year) - parseInt(b.year); });
        
        if (sorted.length === 0) {
            chartDom.innerHTML = '<div style="padding:40px;text-align:center"><p>暂无时间线数据</p></div>';
            return;
        }
        
        // Get unique years
        var years = [];
        var yearMap = {};
        for (var j = 0; j < sorted.length; j++) {
            var year = sorted[j].year;
            if (!yearMap[year]) {
                yearMap[year] = true;
                years.push(year);
            }
        }
        
        // Group papers by year for y-position
        var yearGroups = {};
        for (var k = 0; k < sorted.length; k++) {
            var yr = sorted[k].year;
            if (!yearGroups[yr]) yearGroups[yr] = [];
            yearGroups[yr].push(sorted[k]);
        }
        
        // Create data points
        var data = [];
        for (var m = 0; m < sorted.length; m++) {
            var paper = sorted[m];
            var yearIdx = years.indexOf(paper.year);
            var idxInYear = yearGroups[paper.year].indexOf(paper);
            data.push({
                value: [yearIdx, idxInYear],
                paper: paper,
                itemStyle: { color: colors[paper.category] || '#999' },
                symbolSize: 14
            });
        }
        
        // Calculate max papers per year for Y axis scaling
        var maxInYear = 0;
        for (var yr in yearGroups) {
            maxInYear = Math.max(maxInYear, yearGroups[yr].length);
        }
        
        // Build legend data
        var legendData = ['基础理论', '变分分割', '深度学习', '雷达与无线电', '医学图像', '其他'];
        var legendColors = {
            '基础理论': '#ef4444',
            '变分分割': '#3b82f6',
            '深度学习': '#10b981',
            '雷达与无线电': '#8b5cf6',
            '医学图像': '#f59e0b',
            '其他': '#6b7280'
        };
        
        charts.timeline.setOption({
            tooltip: {
                trigger: 'item',
                formatter: function(params) {
                    if (params.data && params.data.paper) {
                        var p = params.data.paper;
                        return '<strong>[' + p.id + '] ' + p.year + '</strong><br/>' + p.title;
                    }
                    return '';
                }
            },
            legend: {
                data: legendData,
                orient: 'vertical',
                right: '2%',
                top: 'center',
                itemGap: 15,
                textStyle: {
                    fontSize: 12,
                    color: '#374151'
                },
                formatter: function(name) {
                    return name;
                }
            },
            grid: { left: '8%', right: '18%', bottom: '15%', top: '8%' },
            xAxis: {
                type: 'category',
                data: years,
                name: '年份',
                nameLocation: 'middle',
                nameGap: 35,
                axisLabel: { rotate: 30, fontSize: 11 }
            },
            yAxis: {
                type: 'value',
                show: false,
                min: -0.5,
                max: maxInYear
            },
            series: [{
                type: 'scatter',
                data: data,
                label: {
                    show: true,
                    position: 'right',
                    formatter: function(params) {
                        if (params.data && params.data.paper) {
                            var title = params.data.paper.title;
                            // Truncate long titles
                            if (title.length > 18) {
                                title = title.substring(0, 18) + '...';
                            }
                            return title;
                        }
                        return '';
                    },
                    fontSize: 10,
                    color: '#374151',
                    backgroundColor: 'rgba(255,255,255,0.7)',
                    padding: [2, 4],
                    borderRadius: 3
                },
                labelLayout: {
                    hideOverlap: true,
                    moveOverlap: 'shiftY'
                },
                emphasis: {
                    label: {
                        show: true,
                        fontSize: 11,
                        fontWeight: 'bold',
                        backgroundColor: 'rgba(255,255,255,0.95)'
                    },
                    itemStyle: {
                        borderColor: '#333',
                        borderWidth: 2
                    }
                }
            }]
        });
        
        charts.timeline.on('click', function(params) {
            if (params.data && params.data.paper) openPaperModal(params.data.paper);
        });
        
        // Bind filter buttons
        var filterButtons = document.querySelectorAll('.timeline-filters .filter-btn');
        for (var f = 0; f < filterButtons.length; f++) {
            filterButtons[f].addEventListener('click', function() {
                // Update active state
                var allBtns = document.querySelectorAll('.timeline-filters .filter-btn');
                for (var b = 0; b < allBtns.length; b++) {
                    allBtns[b].classList.remove('active');
                }
                this.classList.add('active');
                
                // Filter data
                var filter = this.getAttribute('data-filter');
                var filteredData = [];
                
                if (filter === 'all') {
                    filteredData = data;
                } else {
                    for (var d = 0; d < data.length; d++) {
                        if (data[d].paper && data[d].paper.category === filter) {
                            filteredData.push(data[d]);
                        }
                    }
                }
                
                // Update chart
                charts.timeline.setOption({
                    series: [{
                        data: filteredData
                    }]
                });
            });
        }
        
    } catch (e) {
        console.error('Timeline chart error:', e);
        chartDom.innerHTML = '<div style="padding:40px;text-align:center;color:red"><h3>图表加载失败</h3><p>' + e.message + '</p></div>';
    }
}

// Resize charts on window resize
window.addEventListener('resize', function() {
    if (charts.network) charts.network.resize();
    if (charts.timeline) charts.timeline.resize();
});
