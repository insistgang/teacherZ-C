document.addEventListener('DOMContentLoaded', function() {
    const searchInput = document.getElementById('search-input');
    const searchResults = document.getElementById('search-results');
    
    const searchIndex = [
        { name: 'ROFDenoiser', type: 'class', module: 'denoising', url: 'modules/denoising.html#rofdenoiser', desc: 'ROF图像去噪器' },
        { name: 'TVLDenoiser', type: 'class', module: 'denoising', url: 'modules/denoising.html#tvldenoiser', desc: 'TV-L1图像去噪器' },
        { name: 'NLMDenoiser', type: 'class', module: 'denoising', url: 'modules/denoising.html#nlmdenoiser', desc: '非局部均值去噪器' },
        { name: 'SLatSegmenter', type: 'class', module: 'segmentation', url: 'modules/segmentation.html#slatsegmenter', desc: '光谱-空间潜在分割器' },
        { name: 'GraphCutter', type: 'class', module: 'segmentation', url: 'modules/segmentation.html#graphcutter', desc: '图割分割器' },
        { name: 'TuckerDecomposer', type: 'class', module: 'tensor', url: 'modules/tensor.html#tuckerdecomposer', desc: 'Tucker张量分解器' },
        { name: 'CPDecomposer', type: 'class', module: 'tensor', url: 'modules/tensor.html#cpdecomposer', desc: 'CP张量分解器' },
        { name: 'PointCloudProcessor', type: 'class', module: 'pointcloud', url: 'modules/pointcloud.html#pointcloudprocessor', desc: '点云处理器' },
        { name: 'LoRALayer', type: 'class', module: 'peft', url: 'modules/peft.html#loralayer', desc: 'LoRA适配层' },
        { name: 'denoise', type: 'method', module: 'denoising', url: 'modules/denoising.html#denoise', desc: '执行图像去噪' },
        { name: 'fit', type: 'method', module: 'segmentation', url: 'modules/segmentation.html#fit', desc: '训练分割模型' },
        { name: 'predict', type: 'method', module: 'segmentation', url: 'modules/segmentation.html#predict', desc: '预测分割结果' },
        { name: 'decompose', type: 'method', module: 'tensor', url: 'modules/tensor.html#decompose', desc: '执行张量分解' },
        { name: 'reconstruct', type: 'method', module: 'tensor', url: 'modules/tensor.html#reconstruct', desc: '张量重构' }
    ];
    
    if (searchInput) {
        searchInput.addEventListener('input', function(e) {
            const query = e.target.value.toLowerCase().trim();
            
            if (query.length < 2) {
                searchResults.innerHTML = '';
                searchResults.style.display = 'none';
                return;
            }
            
            const results = searchIndex.filter(item => 
                item.name.toLowerCase().includes(query) ||
                item.desc.toLowerCase().includes(query) ||
                item.module.toLowerCase().includes(query)
            );
            
            if (results.length === 0) {
                searchResults.innerHTML = '<div class="no-results">未找到结果</div>';
                searchResults.style.display = 'block';
                return;
            }
            
            const html = results.map(item => `
                <div class="search-result-item">
                    <a href="${getBasePath()}${item.url}">
                        <span class="result-type ${item.type}">${item.type}</span>
                        <span class="result-name">${item.name}</span>
                        <span class="result-module">${item.module}</span>
                    </a>
                    <p class="result-desc">${item.desc}</p>
                </div>
            `).join('');
            
            searchResults.innerHTML = html;
            searchResults.style.display = 'block';
        });
        
        document.addEventListener('click', function(e) {
            if (!searchInput.contains(e.target) && !searchResults.contains(e.target)) {
                searchResults.style.display = 'none';
            }
        });
    }
    
    function getBasePath() {
        const path = window.location.pathname;
        if (path.includes('/modules/') || path.includes('/examples/')) {
            return '../';
        }
        return '';
    }
    
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    });
    
    const currentPage = window.location.pathname.split('/').pop();
    document.querySelectorAll('.sidebar nav a').forEach(link => {
        if (link.getAttribute('href').includes(currentPage)) {
            link.classList.add('active');
        }
    });
    
    document.querySelectorAll('.example-box pre').forEach(block => {
        const code = block.innerHTML;
        const highlighted = code
            .replace(/(#.*$)/gm, '<span class="comment">$1</span>')
            .replace(/\b(from|import|class|def|return|if|else|for|in|as|with|try|except|raise|pass|True|False|None)\b/g, '<span class="keyword">$1</span>')
            .replace(/(['"])(.*?)\1/g, '<span class="string">$1$2$1</span>')
            .replace(/\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(/g, '<span class="function">$1</span>(');
        block.innerHTML = highlighted;
    });
});
