# SVG 矢量图描述文件

本文档描述可转换为SVG的思维导图结构，支持直接渲染。

---

## 1. 研究全景图 SVG 结构

```svg
<svg viewBox="0 0 1200 800" xmlns="http://www.w3.org/2000/svg">
  <!-- 中心节点 -->
  <circle cx="600" cy="400" r="60" fill="#4A90D9" stroke="#2E5C8A" stroke-width="3"/>
  <text x="600" y="395" text-anchor="middle" fill="white" font-size="14" font-weight="bold">Xiaohao Cai</text>
  <text x="600" y="415" text-anchor="middle" fill="white" font-size="12">研究成果</text>
  
  <!-- 变分图像分割分支 (左上) -->
  <g transform="translate(200, 200)">
    <rect x="-80" y="-25" width="160" height="50" rx="10" fill="#E74C3C" opacity="0.9"/>
    <text x="0" y="5" text-anchor="middle" fill="white" font-size="14">变分图像分割</text>
    
    <!-- 子节点 -->
    <g transform="translate(-120, 80)">
      <rect x="-40" y="-15" width="80" height="30" rx="5" fill="#C0392B"/>
      <text x="0" y="5" text-anchor="middle" fill="white" font-size="11">ROF模型</text>
    </g>
    <g transform="translate(0, 80)">
      <rect x="-50" y="-15" width="100" height="30" rx="5" fill="#C0392B"/>
      <text x="0" y="5" text-anchor="middle" fill="white" font-size="11">Mumford-Shah</text>
    </g>
    <g transform="translate(120, 80)">
      <rect x="-40" y="-15" width="80" height="30" rx="5" fill="#C0392B"/>
      <text x="0" y="5" text-anchor="middle" fill="white" font-size="11">SLaT框架</text>
    </g>
  </g>
  
  <!-- 射电天文分支 (右上) -->
  <g transform="translate(1000, 200)">
    <rect x="-60" y="-25" width="120" height="50" rx="10" fill="#9B59B6" opacity="0.9"/>
    <text x="0" y="5" text-anchor="middle" fill="white" font-size="14">射电天文</text>
    
    <g transform="translate(-60, 80)">
      <rect x="-50" y="-15" width="100" height="30" rx="5" fill="#8E44AD"/>
      <text x="0" y="5" text-anchor="middle" fill="white" font-size="11">不确定性量化</text>
    </g>
    <g transform="translate(60, 80)">
      <rect x="-40" y="-15" width="80" height="30" rx="5" fill="#8E44AD"/>
      <text x="0" y="5" text-anchor="middle" fill="white" font-size="11">在线成像</text>
    </g>
  </g>
  
  <!-- 医学影像分支 (左下) -->
  <g transform="translate(200, 600)">
    <rect x="-60" y="-25" width="120" height="50" rx="10" fill="#27AE60" opacity="0.9"/>
    <text x="0" y="5" text-anchor="middle" fill="white" font-size="14">医学影像</text>
    
    <g transform="translate(-60, 80)">
      <rect x="-40" y="-15" width="80" height="30" rx="5" fill="#1E8449"/>
      <text x="0" y="5" text-anchor="middle" fill="white" font-size="11">血管分割</text>
    </g>
    <g transform="translate(60, 80)">
      <rect x="-40" y="-15" width="80" height="30" rx="5" fill="#1E8449"/>
      <text x="0" y="5" text-anchor="middle" fill="white" font-size="11">MRI重建</text>
    </g>
  </g>
  
  <!-- 3D视觉分支 (右下) -->
  <g transform="translate(1000, 600)">
    <rect x="-50" y="-25" width="100" height="50" rx="10" fill="#F39C12" opacity="0.9"/>
    <text x="0" y="5" text-anchor="middle" fill="white" font-size="14">3D视觉</text>
    
    <g transform="translate(-60, 80)">
      <rect x="-40" y="-15" width="80" height="30" rx="5" fill="#D68910"/>
      <text x="0" y="5" text-anchor="middle" fill="white" font-size="11">点云处理</text>
    </g>
    <g transform="translate(60, 80)">
      <rect x="-40" y="-15" width="80" height="30" rx="5" fill="#D68910"/>
      <text x="0" y="5" text-anchor="middle" fill="white" font-size="11">遥感应用</text>
    </g>
  </g>
  
  <!-- 深度学习分支 (中上) -->
  <g transform="translate(600, 100)">
    <rect x="-50" y="-25" width="100" height="50" rx="10" fill="#3498DB" opacity="0.9"/>
    <text x="0" y="5" text-anchor="middle" fill="white" font-size="14">深度学习</text>
    
    <g transform="translate(-70, -70)">
      <rect x="-40" y="-15" width="80" height="30" rx="5" fill="#2980B9"/>
      <text x="0" y="5" text-anchor="middle" fill="white" font-size="11">张量分解</text>
    </g>
    <g transform="translate(70, -70)">
      <rect x="-40" y="-15" width="80" height="30" rx="5" fill="#2980B9"/>
      <text x="0" y="5" text-anchor="middle" fill="white" font-size="11">多模态</text>
    </g>
  </g>
  
  <!-- 连接线 -->
  <g stroke="#666" stroke-width="2" fill="none">
    <path d="M560,380 Q400,300 280,225"/>
    <path d="M640,380 Q800,300 940,225"/>
    <path d="M560,420 Q400,500 280,575"/>
    <path d="M640,420 Q800,500 950,575"/>
    <path d="M600,340 L600,150"/>
  </g>
</svg>
```

---

## 2. 方法演进时间线 SVG 结构

```svg
<svg viewBox="0 0 1400 400" xmlns="http://www.w3.org/2000/svg">
  <!-- 时间轴 -->
  <line x1="50" y1="200" x2="1350" y2="200" stroke="#333" stroke-width="3"/>
  
  <!-- 奠基期 (2011-2015) -->
  <g transform="translate(100, 200)">
    <circle r="8" fill="#E74C3C"/>
    <text y="-20" text-anchor="middle" font-size="12" fill="#333">2011</text>
    <text y="40" text-anchor="middle" font-size="10" fill="#666">ROF模型</text>
  </g>
  
  <g transform="translate(200, 200)">
    <circle r="8" fill="#E74C3C"/>
    <text y="-20" text-anchor="middle" font-size="12" fill="#333">2012</text>
    <text y="40" text-anchor="middle" font-size="10" fill="#666">Mumford-Shah</text>
  </g>
  
  <g transform="translate(300, 200)">
    <circle r="8" fill="#E74C3C"/>
    <text y="-20" text-anchor="middle" font-size="12" fill="#333">2013</text>
    <text y="40" text-anchor="middle" font-size="10" fill="#666">SLaT框架</text>
  </g>
  
  <g transform="translate(400, 200)">
    <circle r="8" fill="#E74C3C"/>
    <text y="-20" text-anchor="middle" font-size="12" fill="#333">2014</text>
    <text y="40" text-anchor="middle" font-size="10" fill="#666">凸优化</text>
  </g>
  
  <g transform="translate(500, 200)">
    <circle r="8" fill="#E74C3C"/>
    <text y="-20" text-anchor="middle" font-size="12" fill="#333">2015</text>
    <text y="40" text-anchor="middle" font-size="10" fill="#666">选择性分割</text>
  </g>
  
  <!-- 拓展期 (2016-2020) -->
  <g transform="translate(650, 200)">
    <circle r="8" fill="#9B59B6"/>
    <text y="-20" text-anchor="middle" font-size="12" fill="#333">2016-17</text>
    <text y="40" text-anchor="middle" font-size="10" fill="#666">射电天文</text>
  </g>
  
  <g transform="translate(800, 200)">
    <circle r="8" fill="#9B59B6"/>
    <text y="-20" text-anchor="middle" font-size="12" fill="#333">2018-19</text>
    <text y="40" text-anchor="middle" font-size="10" fill="#666">医学影像</text>
  </g>
  
  <g transform="translate(950, 200)">
    <circle r="8" fill="#9B59B6"/>
    <text y="-20" text-anchor="middle" font-size="12" fill="#333">2020</text>
    <text y="40" text-anchor="middle" font-size="10" fill="#666">在线成像</text>
  </g>
  
  <!-- 融合期 (2021-2023) -->
  <g transform="translate(1080, 200)">
    <circle r="8" fill="#27AE60"/>
    <text y="-20" text-anchor="middle" font-size="12" fill="#333">2021-23</text>
    <text y="40" text-anchor="middle" font-size="10" fill="#666">深度学习融合</text>
  </g>
  
  <!-- 创新期 (2024-2026) -->
  <g transform="translate(1250, 200)">
    <circle r="8" fill="#3498DB"/>
    <text y="-20" text-anchor="middle" font-size="12" fill="#333">2024-26</text>
    <text y="40" text-anchor="middle" font-size="10" fill="#666">高效微调</text>
  </g>
  
  <!-- 阶段标签 -->
  <text x="300" y="80" text-anchor="middle" font-size="14" fill="#E74C3C" font-weight="bold">奠基期</text>
  <text x="800" y="80" text-anchor="middle" font-size="14" fill="#9B59B6" font-weight="bold">拓展期</text>
  <text x="1080" y="80" text-anchor="middle" font-size="14" fill="#27AE60" font-weight="bold">融合期</text>
  <text x="1250" y="80" text-anchor="middle" font-size="14" fill="#3498DB" font-weight="bold">创新期</text>
</svg>
```

---

## 3. 数学工具网络图 SVG 结构

```svg
<svg viewBox="0 0 1000 600" xmlns="http://www.w3.org/2000/svg">
  <!-- 变分方法集群 -->
  <g transform="translate(150, 150)">
    <rect x="-70" y="-25" width="140" height="50" rx="10" fill="#E74C3C" opacity="0.2"/>
    <text x="0" y="0" text-anchor="middle" font-size="14" font-weight="bold" fill="#C0392B">变分方法</text>
    <circle cx="-30" cy="50" r="20" fill="#E74C3C"/>
    <text x="-30" y="55" text-anchor="middle" font-size="10" fill="white">能量泛函</text>
    <circle cx="30" cy="50" r="20" fill="#E74C3C"/>
    <text x="30" y="55" text-anchor="middle" font-size="10" fill="white">梯度下降</text>
  </g>
  
  <!-- 凸优化集群 -->
  <g transform="translate(350, 150)">
    <rect x="-60" y="-25" width="120" height="50" rx="10" fill="#9B59B6" opacity="0.2"/>
    <text x="0" y="0" text-anchor="middle" font-size="14" font-weight="bold" fill="#8E44AD">凸优化</text>
    <circle cx="-30" cy="50" r="20" fill="#9B59B6"/>
    <text x="-30" y="55" text-anchor="middle" font-size="10" fill="white">ADMM</text>
    <circle cx="30" cy="50" r="20" fill="#9B59B6"/>
    <text x="30" y="55" text-anchor="middle" font-size="10" fill="white">凸松弛</text>
  </g>
  
  <!-- 贝叶斯集群 -->
  <g transform="translate(150, 400)">
    <rect x="-60" y="-25" width="120" height="50" rx="10" fill="#27AE60" opacity="0.2"/>
    <text x="0" y="0" text-anchor="middle" font-size="14" font-weight="bold" fill="#1E8449">贝叶斯</text>
    <circle cx="-30" cy="50" r="20" fill="#27AE60"/>
    <text x="-30" y="55" text-anchor="middle" font-size="10" fill="white">MCMC</text>
    <circle cx="30" cy="50" r="20" fill="#27AE60"/>
    <text x="30" y="55" text-anchor="middle" font-size="10" fill="white">变分推断</text>
  </g>
  
  <!-- 张量分析集群 -->
  <g transform="translate(350, 400)">
    <rect x="-60" y="-25" width="120" height="50" rx="10" fill="#F39C12" opacity="0.2"/>
    <text x="0" y="0" text-anchor="middle" font-size="14" font-weight="bold" fill="#D68910">张量分析</text>
    <circle cx="-30" cy="50" r="20" fill="#F39C12"/>
    <text x="-30" y="55" text-anchor="middle" font-size="10" fill="white">CP分解</text>
    <circle cx="30" cy="50" r="20" fill="#F39C12"/>
    <text x="30" y="55" text-anchor="middle" font-size="10" fill="white">Tucker</text>
  </g>
  
  <!-- 深度学习集群 -->
  <g transform="translate(700, 300)">
    <rect x="-70" y="-25" width="140" height="50" rx="10" fill="#3498DB" opacity="0.2"/>
    <text x="0" y="0" text-anchor="middle" font-size="14" font-weight="bold" fill="#2980B9">深度学习</text>
    <circle cx="-50" cy="60" r="25" fill="#3498DB"/>
    <text x="-50" y="65" text-anchor="middle" font-size="10" fill="white">CNN</text>
    <circle cx="0" cy="60" r="25" fill="#3498DB"/>
    <text x="0" y="65" text-anchor="middle" font-size="10" fill="white">Attention</text>
    <circle cx="50" cy="60" r="25" fill="#3498DB"/>
    <text x="50" y="65" text-anchor="middle" font-size="10" fill="white">PEFT</text>
  </g>
  
  <!-- 连接线 -->
  <g stroke="#999" stroke-width="1.5" stroke-dasharray="5,3" fill="none">
    <path d="M180,180 Q260,200 320,180"/>
    <path d="M180,220 Q260,350 180,370"/>
    <path d="M380,180 Q450,280 630,300"/>
    <path d="M380,420 Q500,350 630,320"/>
    <path d="M180,420 Q280,350 320,420"/>
  </g>
</svg>
```

---

## 4. 应用领域辐射图 SVG 结构

```svg
<svg viewBox="0 0 800 800" xmlns="http://www.w3.org/2000/svg">
  <!-- 中心 -->
  <circle cx="400" cy="400" r="50" fill="#2C3E50"/>
  <text x="400" y="405" text-anchor="middle" font-size="14" fill="white" font-weight="bold">应用领域</text>
  
  <!-- 医学影像 (12点方向) -->
  <g transform="translate(400, 200)">
    <circle r="40" fill="#27AE60"/>
    <text y="5" text-anchor="middle" font-size="12" fill="white">医学影像</text>
    <line x1="0" y1="40" x2="0" y2="150" stroke="#27AE60" stroke-width="2"/>
    <circle cx="-40" cy="100" r="25" fill="#58D68D"/>
    <text x="-40" y="105" text-anchor="middle" font-size="9" fill="white">眼底</text>
    <circle cx="40" cy="100" r="25" fill="#58D68D"/>
    <text x="40" y="105" text-anchor="middle" font-size="9" fill="white">MRI</text>
  </g>
  
  <!-- 遥感 (2点方向) -->
  <g transform="translate(580, 280)">
    <circle r="35" fill="#E74C3C"/>
    <text y="5" text-anchor="middle" font-size="11" fill="white">遥感影像</text>
  </g>
  
  <!-- 射电天文 (5点方向) -->
  <g transform="translate(600, 500)">
    <circle r="35" fill="#9B59B6"/>
    <text y="5" text-anchor="middle" font-size="11" fill="white">射电天文</text>
  </g>
  
  <!-- 自动驾驶 (7点方向) -->
  <g transform="translate(400, 600)">
    <circle r="35" fill="#F39C12"/>
    <text y="5" text-anchor="middle" font-size="11" fill="white">自动驾驶</text>
  </g>
  
  <!-- 工业检测 (10点方向) -->
  <g transform="translate(200, 500)">
    <circle r="35" fill="#1ABC9C"/>
    <text y="5" text-anchor="middle" font-size="11" fill="white">工业检测</text>
  </g>
  
  <!-- 3D视觉 (11点方向) -->
  <g transform="translate(200, 280)">
    <circle r="35" fill="#3498DB"/>
    <text y="5" text-anchor="middle" font-size="11" fill="white">3D视觉</text>
  </g>
  
  <!-- 辐射线 -->
  <g stroke="#BDC3C7" stroke-width="1">
    <line x1="400" y1="350" x2="400" y2="240"/>
    <line x1="445" y1="360" x2="545" y2="300"/>
    <line x1="445" y1="440" x2="565" y2="480"/>
    <line x1="400" y1="450" x2="400" y2="565"/>
    <line x1="355" y1="440" x2="235" y2="480"/>
    <line x1="355" y1="360" x2="235" y2="300"/>
  </g>
</svg>
```

---

## 5. 论文关系网络图 SVG 结构

```svg
<svg viewBox="0 0 1000 700" xmlns="http://www.w3.org/2000/svg">
  <!-- 论文数量指示 -->
  <defs>
    <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#E74C3C;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#C0392B;stop-opacity:1" />
    </linearGradient>
  </defs>
  
  <!-- 中心节点 -->
  <circle cx="500" cy="350" r="60" fill="#2C3E50"/>
  <text x="500" y="345" text-anchor="middle" font-size="12" fill="white" font-weight="bold">68篇论文</text>
  <text x="500" y="365" text-anchor="middle" font-size="10" fill="#BDC3C7">Xiaohao Cai</text>
  
  <!-- 变分分割 (15篇) -->
  <g transform="translate(250, 200)">
    <circle r="45" fill="url(#grad1)"/>
    <text y="-5" text-anchor="middle" font-size="11" fill="white">变分分割</text>
    <text y="12" text-anchor="middle" font-size="14" fill="white" font-weight="bold">15</text>
    
    <circle cx="-60" cy="-30" r="20" fill="#F1948A"/>
    <text x="-60" y="-25" text-anchor="middle" font-size="8" fill="white">ROF</text>
    <circle cx="60" cy="-30" r="20" fill="#F1948A"/>
    <text x="60" y="-25" text-anchor="middle" font-size="8" fill="white">MS</text>
    <circle cx="0" cy="50" r="20" fill="#F1948A"/>
    <text x="0" y="55" text-anchor="middle" font-size="8" fill="white">SLaT</text>
  </g>
  
  <!-- 医学影像 (18篇) -->
  <g transform="translate(750, 200)">
    <circle r="50" fill="#27AE60"/>
    <text y="-5" text-anchor="middle" font-size="11" fill="white">医学影像</text>
    <text y="12" text-anchor="middle" font-size="14" fill="white" font-weight="bold">18</text>
    
    <circle cx="-60" cy="-30" r="22" fill="#58D68D"/>
    <text x="-60" y="-25" text-anchor="middle" font-size="8" fill="white">血管</text>
    <circle cx="60" cy="-30" r="22" fill="#58D68D"/>
    <text x="60" y="-25" text-anchor="middle" font-size="8" fill="white">MRI</text>
  </g>
  
  <!-- 射电天文 (12篇) -->
  <g transform="translate(250, 500)">
    <circle r="40" fill="#9B59B6"/>
    <text y="-5" text-anchor="middle" font-size="11" fill="white">射电天文</text>
    <text y="12" text-anchor="middle" font-size="14" fill="white" font-weight="bold">12</text>
    
    <circle cx="-50" cy="-25" r="18" fill="#BB8FCE"/>
    <text x="-50" y="-20" text-anchor="middle" font-size="8" fill="white">UQ</text>
    <circle cx="50" cy="-25" r="18" fill="#BB8FCE"/>
    <text x="50" y="-20" text-anchor="middle" font-size="8" fill="white">在线</text>
  </g>
  
  <!-- 3D视觉 (13篇) -->
  <g transform="translate(750, 500)">
    <circle r="42" fill="#F39C12"/>
    <text y="-5" text-anchor="middle" font-size="11" fill="white">3D视觉</text>
    <text y="12" text-anchor="middle" font-size="14" fill="white" font-weight="bold">13</text>
    
    <circle cx="-50" cy="-25" r="18" fill="#F5B041"/>
    <text x="-50" y="-20" text-anchor="middle" font-size="8" fill="white">点云</text>
    <circle cx="50" cy="-25" r="18" fill="#F5B041"/>
    <text x="50" y="-20" text-anchor="middle" font-size="8" fill="white">遥感</text>
  </g>
  
  <!-- 深度学习 (10篇) -->
  <g transform="translate(500, 150)">
    <circle r="35" fill="#3498DB"/>
    <text y="-5" text-anchor="middle" font-size="10" fill="white">深度学习</text>
    <text y="10" text-anchor="middle" font-size="12" fill="white" font-weight="bold">10</text>
    
    <circle cx="-40" cy="40" r="15" fill="#5DADE2"/>
    <text x="-40" y="45" text-anchor="middle" font-size="7" fill="white">张量</text>
    <circle cx="40" cy="40" r="15" fill="#5DADE2"/>
    <text x="40" y="45" text-anchor="middle" font-size="7" fill="white">多模态</text>
  </g>
  
  <!-- 关联线 (虚线表示弱关联) -->
  <g stroke="#BDC3C7" stroke-width="1" fill="none">
    <path d="M295,200 Q400,150 450,175" stroke-dasharray="4,2"/>
    <path d="M705,200 Q600,150 550,175" stroke-dasharray="4,2"/>
    <path d="M290,200 Q350,350 290,460"/>
    <path d="M710,200 Q650,350 710,460"/>
    <path d="M450,350 L290,500"/>
    <path d="M550,350 L710,500"/>
  </g>
  
  <!-- 图例 -->
  <g transform="translate(50, 600)">
    <text x="0" y="0" font-size="12" fill="#333" font-weight="bold">图例:</text>
    <circle cx="20" cy="25" r="8" fill="#27AE60"/>
    <text x="35" y="30" font-size="10" fill="#666">医学影像 (18篇)</text>
    <circle cx="130" cy="25" r="8" fill="#E74C3C"/>
    <text x="145" y="30" font-size="10" fill="#666">变分分割 (15篇)</text>
    <circle cx="245" cy="25" r="8" fill="#F39C12"/>
    <text x="260" y="30" font-size="10" fill="#666">3D视觉 (13篇)</text>
    <circle cx="350" cy="25" r="8" fill="#9B59B6"/>
    <text x="365" y="30" font-size="10" fill="#666">射电天文 (12篇)</text>
    <circle cx="470" cy="25" r="8" fill="#3498DB"/>
    <text x="485" y="30" font-size="10" fill="#666">深度学习 (10篇)</text>
  </g>
</svg>
```

---

## 渲染说明

1. **Mermaid文件**: 使用支持Mermaid的编辑器(如Typora、VS Code+插件)打开`.mmd`文件
2. **XMind文件**: 使用XMind软件导入`.xmind` XML文件
3. **SVG描述**: 将上述SVG代码保存为`.svg`文件，可直接在浏览器中打开
4. **在线工具**: 
   - Mermaid: https://mermaid.live/
   - SVG: https://www.aconvert.com/cn/document/md-to-svg/

---

*文档版本: 1.0*
*更新日期: 2026-02-16*
