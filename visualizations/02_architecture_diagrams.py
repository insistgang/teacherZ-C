"""
SVG架构图生成器
生成各种深度学习模型架构图
"""

import os

SVG_HEADER = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 500">
<defs>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
        <path d="M0,0 L0,6 L9,3 z" fill="#333"/>
    </marker>
    <linearGradient id="blueGrad" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" style="stop-color:#3498db"/>
        <stop offset="100%" style="stop-color:#2980b9"/>
    </linearGradient>
    <linearGradient id="redGrad" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" style="stop-color:#e74c3c"/>
        <stop offset="100%" style="stop-color:#c0392b"/>
    </linearGradient>
    <linearGradient id="greenGrad" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" style="stop-color:#2ecc71"/>
        <stop offset="100%" style="stop-color:#27ae60"/>
    </linearGradient>
    <linearGradient id="purpleGrad" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" style="stop-color:#9b59b6"/>
        <stop offset="100%" style="stop-color:#8e44ad"/>
    </linearGradient>
    <linearGradient id="orangeGrad" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" style="stop-color:#f39c12"/>
        <stop offset="100%" style="stop-color:#e67e22"/>
    </linearGradient>
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
        <feDropShadow dx="2" dy="2" stdDeviation="3" flood-opacity="0.3"/>
    </filter>
</defs>
"""

SVG_FOOTTER = "</svg>"

def save_svg(filename, content):
    filepath = f"D:/Documents/zx/visualizations/svgs/{filename}"
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(SVG_HEADER + content + SVG_FOOTTER)
    return filepath

diagrams = {}

diagrams['01_hifi_mamba'] = """
    <!-- 标题 -->
    <text x="400" y="30" text-anchor="middle" font-size="24" font-weight="bold" fill="#2c3e50">HiFi-Mamba Architecture</text>
    
    <!-- 输入 -->
    <rect x="30" y="200" width="80" height="100" fill="url(#blueGrad)" rx="8" filter="url(#shadow)"/>
    <text x="70" y="255" text-anchor="middle" fill="white" font-size="14">Input</text>
    <text x="70" y="275" text-anchor="middle" fill="white" font-size="10">H×W×3</text>
    
    <!-- Patch Embedding -->
    <rect x="140" y="200" width="80" height="100" fill="url(#purpleGrad)" rx="8" filter="url(#shadow)"/>
    <text x="180" y="245" text-anchor="middle" fill="white" font-size="12">Patch</text>
    <text x="180" y="265" text-anchor="middle" fill="white" font-size="12">Embed</text>
    
    <!-- Mamba Blocks -->
    <rect x="250" y="150" width="300" height="200" fill="url(#redGrad)" rx="12" filter="url(#shadow)"/>
    <text x="400" y="200" text-anchor="middle" fill="white" font-size="16" font-weight="bold">Mamba Blocks</text>
    
    <!-- 内部块 -->
    <rect x="270" y="220" width="80" height="50" fill="white" opacity="0.9" rx="5"/>
    <text x="310" y="250" text-anchor="middle" font-size="10" fill="#333">SSM</text>
    
    <rect x="370" y="220" width="80" height="50" fill="white" opacity="0.9" rx="5"/>
    <text x="410" y="250" text-anchor="middle" font-size="10" fill="#333">MLP</text>
    
    <rect x="470" y="220" width="60" height="50" fill="white" opacity="0.9" rx="5"/>
    <text x="500" y="250" text-anchor="middle" font-size="10" fill="#333">Norm</text>
    
    <!-- Decoder -->
    <rect x="580" y="200" width="80" height="100" fill="url(#greenGrad)" rx="8" filter="url(#shadow)"/>
    <text x="620" y="245" text-anchor="middle" fill="white" font-size="12">Decoder</text>
    <text x="620" y="265" text-anchor="middle" fill="white" font-size="10">Upsample</text>
    
    <!-- 输出 -->
    <rect x="690" y="200" width="80" height="100" fill="url(#blueGrad)" rx="8" filter="url(#shadow)"/>
    <text x="730" y="245" text-anchor="middle" fill="white" font-size="12">Output</text>
    <text x="730" y="265" text-anchor="middle" fill="white" font-size="10">H×W×C</text>
    
    <!-- 箭头 -->
    <path d="M110 250 L140 250" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>
    <path d="M220 250 L250 250" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>
    <path d="M550 250 L580 250" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>
    <path d="M660 250 L690 250" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>
"""

diagrams['02_unet_architecture'] = """
    <text x="400" y="30" text-anchor="middle" font-size="24" font-weight="bold" fill="#2c3e50">U-Net Architecture</text>
    
    <!-- 编码器 -->
    <rect x="50" y="80" width="60" height="60" fill="url(#blueGrad)" rx="5" filter="url(#shadow)"/>
    <text x="80" y="115" text-anchor="middle" fill="white" font-size="10">Conv</text>
    
    <rect x="50" y="160" width="60" height="50" fill="url(#blueGrad)" rx="5" filter="url(#shadow)"/>
    <rect x="50" y="230" width="60" height="40" fill="url(#blueGrad)" rx="5" filter="url(#shadow)"/>
    <rect x="50" y="290" width="60" height="30" fill="url(#blueGrad)" rx="5" filter="url(#shadow)"/>
    
    <!-- 下采样箭头 -->
    <path d="M110 110 L130 135" stroke="#e74c3c" stroke-width="2" marker-end="url(#arrow)"/>
    <path d="M110 185 L130 210" stroke="#e74c3c" stroke-width="2" marker-end="url(#arrow)"/>
    <path d="M110 250 L130 265" stroke="#e74c3c" stroke-width="2" marker-end="url(#arrow)"/>
    
    <!-- 瓶颈层 -->
    <rect x="130" y="350" width="540" height="50" fill="url(#redGrad)" rx="8" filter="url(#shadow)"/>
    <text x="400" y="380" text-anchor="middle" fill="white" font-size="14" font-weight="bold">Bottleneck</text>
    
    <!-- 解码器 -->
    <rect x="690" y="80" width="60" height="60" fill="url(#greenGrad)" rx="5" filter="url(#shadow)"/>
    <text x="720" y="115" text-anchor="middle" fill="white" font-size="10">Conv</text>
    
    <rect x="690" y="160" width="60" height="50" fill="url(#greenGrad)" rx="5" filter="url(#shadow)"/>
    <rect x="690" y="230" width="60" height="40" fill="url(#greenGrad)" rx="5" filter="url(#shadow)"/>
    <rect x="690" y="290" width="60" height="30" fill="url(#greenGrad)" rx="5" filter="url(#shadow)"/>
    
    <!-- 上采样箭头 -->
    <path d="M690 110 L670 135" stroke="#2ecc71" stroke-width="2" marker-end="url(#arrow)"/>
    <path d="M690 185 L670 210" stroke="#2ecc71" stroke-width="2" marker-end="url(#arrow)"/>
    <path d="M690 250 L670 265" stroke="#2ecc71" stroke-width="2" marker-end="url(#arrow)"/>
    
    <!-- 跳跃连接 -->
    <path d="M110 110 Q400 60 690 110" stroke="#9b59b6" stroke-width="2" stroke-dasharray="5,5" fill="none"/>
    <path d="M110 185 Q400 140 690 185" stroke="#9b59b6" stroke-width="2" stroke-dasharray="5,5" fill="none"/>
    <path d="M110 250 Q400 220 690 250" stroke="#9b59b6" stroke-width="2" stroke-dasharray="5,5" fill="none"/>
    
    <text x="400" y="55" text-anchor="middle" fill="#9b59b6" font-size="12">Skip Connections</text>
    
    <!-- 标签 -->
    <text x="80" y="420" text-anchor="middle" fill="#3498db" font-size="12">Encoder</text>
    <text x="720" y="420" text-anchor="middle" fill="#2ecc71" font-size="12">Decoder</text>
"""

diagrams['03_transformer_encoder'] = """
    <text x="400" y="30" text-anchor="middle" font-size="24" font-weight="bold" fill="#2c3e50">Transformer Encoder Block</text>
    
    <!-- 输入 -->
    <rect x="350" y="430" width="100" height="40" fill="url(#blueGrad)" rx="5" filter="url(#shadow)"/>
    <text x="400" y="455" text-anchor="middle" fill="white" font-size="12">Input</text>
    
    <!-- 多头注意力 -->
    <rect x="300" y="340" width="200" height="60" fill="url(#orangeGrad)" rx="8" filter="url(#shadow)"/>
    <text x="400" y="375" text-anchor="middle" fill="white" font-size="14" font-weight="bold">Multi-Head Attention</text>
    
    <!-- Add & Norm 1 -->
    <rect x="325" y="280" width="150" height="40" fill="url(#purpleGrad)" rx="5" filter="url(#shadow)"/>
    <text x="400" y="305" text-anchor="middle" fill="white" font-size="12">Add & Norm</text>
    
    <!-- FFN -->
    <rect x="300" y="200" width="200" height="50" fill="url(#greenGrad)" rx="8" filter="url(#shadow)"/>
    <text x="400" y="230" text-anchor="middle" fill="white" font-size="14" font-weight="bold">Feed Forward</text>
    
    <!-- Add & Norm 2 -->
    <rect x="325" y="130" width="150" height="40" fill="url(#purpleGrad)" rx="5" filter="url(#shadow)"/>
    <text x="400" y="155" text-anchor="middle" fill="white" font-size="12">Add & Norm</text>
    
    <!-- 输出 -->
    <rect x="350" y="60" width="100" height="40" fill="url(#redGrad)" rx="5" filter="url(#shadow)"/>
    <text x="400" y="85" text-anchor="middle" fill="white" font-size="12">Output</text>
    
    <!-- 箭头 -->
    <path d="M400 430 L400 400" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>
    <path d="M400 340 L400 320" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>
    <path d="M400 280 L400 250" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>
    <path d="M400 200 L400 170" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>
    <path d="M400 130 L400 100" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>
    
    <!-- 残差连接 -->
    <path d="M300 450 L200 450 L200 300 L325 300" stroke="#e74c3c" stroke-width="2" stroke-dasharray="5,5" fill="none" marker-end="url(#arrow)"/>
    <path d="M300 270 L220 270 L220 150 L325 150" stroke="#e74c3c" stroke-width="2" stroke-dasharray="5,5" fill="none" marker-end="url(#arrow)"/>
    
    <text x="180" y="380" text-anchor="middle" fill="#e74c3c" font-size="11" transform="rotate(-90, 180, 380)">Residual</text>
"""

diagrams['04_attention_mechanism'] = """
    <text x="400" y="30" text-anchor="middle" font-size="24" font-weight="bold" fill="#2c3e50">Self-Attention Mechanism</text>
    
    <!-- 输入 -->
    <rect x="350" y="450" width="100" height="30" fill="url(#blueGrad)" rx="5"/>
    <text x="400" y="470" text-anchor="middle" fill="white" font-size="11">Input X</text>
    
    <!-- Q, K, V -->
    <rect x="150" y="380" width="70" height="40" fill="url(#orangeGrad)" rx="5" filter="url(#shadow)"/>
    <text x="185" y="405" text-anchor="middle" fill="white" font-size="14" font-weight="bold">Q</text>
    
    <rect x="365" y="380" width="70" height="40" fill="url(#greenGrad)" rx="5" filter="url(#shadow)"/>
    <text x="400" y="405" text-anchor="middle" fill="white" font-size="14" font-weight="bold">K</text>
    
    <rect x="580" y="380" width="70" height="40" fill="url(#purpleGrad)" rx="5" filter="url(#shadow)"/>
    <text x="615" y="405" text-anchor="middle" fill="white" font-size="14" font-weight="bold">V</text>
    
    <!-- MatMul QK -->
    <circle cx="270" cy="300" r="25" fill="#e74c3c" filter="url(#shadow)"/>
    <text x="270" y="305" text-anchor="middle" fill="white" font-size="10">×</text>
    
    <!-- Scale -->
    <rect x="230" y="220" width="80" height="30" fill="#3498db" rx="5"/>
    <text x="270" y="240" text-anchor="middle" fill="white" font-size="10">Scale</text>
    
    <!-- Softmax -->
    <rect x="230" y="150" width="80" height="30" fill="#f39c12" rx="5"/>
    <text x="270" y="170" text-anchor="middle" fill="white" font-size="10">Softmax</text>
    
    <!-- MatMul with V -->
    <circle cx="430" cy="165" r="25" fill="#e74c3c" filter="url(#shadow)"/>
    <text x="430" y="170" text-anchor="middle" fill="white" font-size="10">×</text>
    
    <!-- 输出 -->
    <rect x="380" y="60" width="100" height="30" fill="url(#redGrad)" rx="5"/>
    <text x="430" y="80" text-anchor="middle" fill="white" font-size="11">Output</text>
    
    <!-- 连接线 -->
    <path d="M400 450 L185 420" stroke="#333" stroke-width="1.5"/>
    <path d="M400 450 L400 420" stroke="#333" stroke-width="1.5"/>
    <path d="M400 450 L615 420" stroke="#333" stroke-width="1.5"/>
    
    <path d="M185 380 L185 300 L245 300" stroke="#333" stroke-width="1.5" marker-end="url(#arrow)"/>
    <path d="M400 380 L295 300" stroke="#333" stroke-width="1.5" marker-end="url(#arrow)"/>
    
    <path d="M270 275 L270 250" stroke="#333" stroke-width="1.5" marker-end="url(#arrow)"/>
    <path d="M270 220 L270 180" stroke="#333" stroke-width="1.5" marker-end="url(#arrow)"/>
    <path d="M310 165 L405 165" stroke="#333" stroke-width="1.5" marker-end="url(#arrow)"/>
    
    <path d="M615 380 L615 165 L455 165" stroke="#333" stroke-width="1.5" marker-end="url(#arrow)"/>
    
    <path d="M430 140 L430 90" stroke="#333" stroke-width="1.5" marker-end="url(#arrow)"/>
"""

diagrams['05_mamba_ssm'] = """
    <text x="400" y="30" text-anchor="middle" font-size="24" font-weight="bold" fill="#2c3e50">Mamba State Space Model</text>
    
    <!-- 输入 -->
    <rect x="50" y="200" width="80" height="100" fill="url(#blueGrad)" rx="8" filter="url(#shadow)"/>
    <text x="90" y="240" text-anchor="middle" fill="white" font-size="12">Input</text>
    <text x="90" y="260" text-anchor="middle" fill="white" font-size="10">x(t)</text>
    
    <!-- 线性投影 -->
    <rect x="160" y="200" width="100" height="100" fill="url(#purpleGrad)" rx="8" filter="url(#shadow)"/>
    <text x="210" y="240" text-anchor="middle" fill="white" font-size="11">Linear</text>
    <text x="210" y="260" text-anchor="middle" fill="white" font-size="11">Projection</text>
    
    <!-- SSM核心 -->
    <rect x="290" y="120" width="220" height="260" fill="url(#redGrad)" rx="12" filter="url(#shadow)"/>
    <text x="400" y="150" text-anchor="middle" fill="white" font-size="14" font-weight="bold">Selective SSM</text>
    
    <!-- SSM内部组件 -->
    <rect x="310" y="170" width="80" height="50" fill="white" opacity="0.9" rx="5"/>
    <text x="350" y="195" text-anchor="middle" font-size="10" fill="#333">Δ (Delta)</text>
    <text x="350" y="210" text-anchor="middle" font-size="8" fill="#666">时间步长</text>
    
    <rect x="410" y="170" width="80" height="50" fill="white" opacity="0.9" rx="5"/>
    <text x="450" y="195" text-anchor="middle" font-size="10" fill="#333">B, C</text>
    <text x="450" y="210" text-anchor="middle" font-size="8" fill="#666">输入/输出</text>
    
    <rect x="310" y="240" width="180" height="60" fill="white" opacity="0.9" rx="5"/>
    <text x="400" y="265" text-anchor="middle" font-size="10" fill="#333">Discretization</text>
    <text x="400" y="285" text-anchor="middle" font-size="8" fill="#666">Ā = f(A, Δ), B̄ = f(B, Δ)</text>
    
    <rect x="310" y="320" width="180" height="40" fill="white" opacity="0.9" rx="5"/>
    <text x="400" y="345" text-anchor="middle" font-size="10" fill="#333">h(t) = Āh(t-1) + B̄x(t)</text>
    
    <!-- 输出投影 -->
    <rect x="540" y="200" width="100" height="100" fill="url(#greenGrad)" rx="8" filter="url(#shadow)"/>
    <text x="590" y="240" text-anchor="middle" fill="white" font-size="11">Output</text>
    <text x="590" y="260" text-anchor="middle" fill="white" font-size="11">Projection</text>
    
    <!-- 输出 -->
    <rect x="670" y="200" width="80" height="100" fill="url(#blueGrad)" rx="8" filter="url(#shadow)"/>
    <text x="710" y="240" text-anchor="middle" fill="white" font-size="12">Output</text>
    <text x="710" y="260" text-anchor="middle" fill="white" font-size="10">y(t)</text>
    
    <!-- 箭头 -->
    <path d="M130 250 L160 250" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>
    <path d="M260 250 L290 250" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>
    <path d="M510 250 L540 250" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>
    <path d="M640 250 L670 250" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>
"""

diagrams['06_resnet_block'] = """
    <text x="400" y="30" text-anchor="middle" font-size="24" font-weight="bold" fill="#2c3e50">ResNet Residual Block</text>
    
    <!-- 输入 -->
    <rect x="350" y="450" width="100" height="30" fill="url(#blueGrad)" rx="5"/>
    <text x="400" y="470" text-anchor="middle" fill="white" font-size="12">Input x</text>
    
    <!-- Conv1 -->
    <rect x="325" y="380" width="150" height="40" fill="url(#greenGrad)" rx="5" filter="url(#shadow)"/>
    <text x="400" y="405" text-anchor="middle" fill="white" font-size="12">Conv 3×3</text>
    
    <!-- BN1 -->
    <rect x="325" y="320" width="150" height="30" fill="url(#orangeGrad)" rx="5"/>
    <text x="400" y="340" text-anchor="middle" fill="white" font-size="11">BatchNorm</text>
    
    <!-- ReLU1 -->
    <rect x="325" y="270" width="150" height="30" fill="url(#purpleGrad)" rx="5"/>
    <text x="400" y="290" text-anchor="middle" fill="white" font-size="11">ReLU</text>
    
    <!-- Conv2 -->
    <rect x="325" y="210" width="150" height="40" fill="url(#greenGrad)" rx="5" filter="url(#shadow)"/>
    <text x="400" y="235" text-anchor="middle" fill="white" font-size="12">Conv 3×3</text>
    
    <!-- BN2 -->
    <rect x="325" y="160" width="150" height="30" fill="url(#orangeGrad)" rx="5"/>
    <text x="400" y="180" text-anchor="middle" fill="white" font-size="11">BatchNorm</text>
    
    <!-- Add -->
    <circle cx="400" cy="120" r="20" fill="#e74c3c" filter="url(#shadow)"/>
    <text x="400" y="125" text-anchor="middle" fill="white" font-size="16">+</text>
    
    <!-- ReLU2 -->
    <rect x="325" y="70" width="150" height="30" fill="url(#purpleGrad)" rx="5"/>
    <text x="400" y="90" text-anchor="middle" fill="white" font-size="11">ReLU</text>
    
    <!-- 输出 -->
    <rect x="350" y="20" width="100" height="30" fill="url(#blueGrad)" rx="5"/>
    <text x="400" y="40" text-anchor="middle" fill="white" font-size="12">Output</text>
    
    <!-- 主路径箭头 -->
    <path d="M400 450 L400 420" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>
    <path d="M400 380 L400 350" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>
    <path d="M400 320 L400 300" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>
    <path d="M400 270 L400 250" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>
    <path d="M400 210 L400 190" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>
    <path d="M400 160 L400 140" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>
    <path d="M400 100 L400 70" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>
    <path d="M400 70 L400 50" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>
    
    <!-- 残差连接 -->
    <path d="M325 465 L150 465 L150 120 L380 120" stroke="#e74c3c" stroke-width="3" stroke-dasharray="8,4" fill="none" marker-end="url(#arrow)"/>
    <text x="120" y="300" text-anchor="middle" fill="#e74c3c" font-size="12" transform="rotate(-90, 120, 300)">Identity</text>
"""

diagrams['07_fpn_architecture'] = """
    <text x="400" y="30" text-anchor="middle" font-size="24" font-weight="bold" fill="#2c3e50">Feature Pyramid Network (FPN)</text>
    
    <!-- 自底向上 -->
    <rect x="100" y="380" width="100" height="50" fill="url(#blueGrad)" rx="5" filter="url(#shadow)"/>
    <text x="150" y="410" text-anchor="middle" fill="white" font-size="11">C1</text>
    
    <rect x="100" y="300" width="100" height="50" fill="url(#blueGrad)" rx="5" filter="url(#shadow)"/>
    <text x="150" y="330" text-anchor="middle" fill="white" font-size="11">C2</text>
    
    <rect x="100" y="220" width="100" height="50" fill="url(#blueGrad)" rx="5" filter="url(#shadow)"/>
    <text x="150" y="250" text-anchor="middle" fill="white" font-size="11">C3</text>
    
    <rect x="100" y="140" width="100" height="50" fill="url(#blueGrad)" rx="5" filter="url(#shadow)"/>
    <text x="150" y="170" text-anchor="middle" fill="white" font-size="11">C4</text>
    
    <rect x="100" y="60" width="100" height="50" fill="url(#blueGrad)" rx="5" filter="url(#shadow)"/>
    <text x="150" y="90" text-anchor="middle" fill="white" font-size="11">C5</text>
    
    <!-- 自顶向下 -->
    <rect x="600" y="60" width="100" height="50" fill="url(#greenGrad)" rx="5" filter="url(#shadow)"/>
    <text x="650" y="90" text-anchor="middle" fill="white" font-size="11">P5</text>
    
    <rect x="600" y="140" width="100" height="50" fill="url(#greenGrad)" rx="5" filter="url(#shadow)"/>
    <text x="650" y="170" text-anchor="middle" fill="white" font-size="11">P4</text>
    
    <rect x="600" y="220" width="100" height="50" fill="url(#greenGrad)" rx="5" filter="url(#shadow)"/>
    <text x="650" y="250" text-anchor="middle" fill="white" font-size="11">P3</text>
    
    <rect x="600" y="300" width="100" height="50" fill="url(#greenGrad)" rx="5" filter="url(#shadow)"/>
    <text x="650" y="330" text-anchor="middle" fill="white" font-size="11">P2</text>
    
    <!-- 1×1卷积 -->
    <rect x="350" y="60" width="80" height="50" fill="url(#orangeGrad)" rx="5"/>
    <text x="390" y="90" text-anchor="middle" fill="white" font-size="10">1×1 Conv</text>
    
    <rect x="350" y="140" width="80" height="50" fill="url(#orangeGrad)" rx="5"/>
    <text x="390" y="170" text-anchor="middle" fill="white" font-size="10">1×1 Conv</text>
    
    <rect x="350" y="220" width="80" height="50" fill="url(#orangeGrad)" rx="5"/>
    <text x="390" y="250" text-anchor="middle" fill="white" font-size="10">1×1 Conv</text>
    
    <rect x="350" y="300" width="80" height="50" fill="url(#orangeGrad)" rx="5"/>
    <text x="390" y="330" text-anchor="middle" fill="white" font-size="10">1×1 Conv</text>
    
    <!-- 上采样 -->
    <rect x="480" y="105" width="80" height="40" fill="url(#purpleGrad)" rx="5"/>
    <text x="520" y="130" text-anchor="middle" fill="white" font-size="9">Upsample</text>
    
    <!-- 连接线 -->
    <path d="M200 85 L350 85" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>
    <path d="M430 85 L600 85" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>
    
    <path d="M200 165 L350 165" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>
    <path d="M430 165 L600 165" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>
    
    <path d="M200 245 L350 245" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>
    <path d="M430 245 L600 245" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>
    
    <path d="M200 325 L350 325" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>
    <path d="M430 325 L600 325" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>
    
    <!-- 垂直连接 -->
    <path d="M150 380 L150 350" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>
    <path d="M150 300 L150 270" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>
    <path d="M150 220 L150 190" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>
    <path d="M150 140 L150 110" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>
    
    <!-- 自顶向下路径 -->
    <path d="M650 110 L650 140" stroke="#2ecc71" stroke-width="2" marker-end="url(#arrow)"/>
    <path d="M650 190 L650 220" stroke="#2ecc71" stroke-width="2" marker-end="url(#arrow)"/>
    <path d="M650 270 L650 300" stroke="#2ecc71" stroke-width="2" marker-end="url(#arrow)"/>
    
    <!-- 加法节点 -->
    <circle cx="520" cy="165" r="15" fill="#e74c3c"/>
    <text x="520" y="170" text-anchor="middle" fill="white" font-size="14">+</text>
    <circle cx="520" cy="245" r="15" fill="#e74c3c"/>
    <text x="520" y="250" text-anchor="middle" fill="white" font-size="14">+</text>
    <circle cx="520" cy="325" r="15" fill="#e74c3c"/>
    <text x="520" y="330" text-anchor="middle" fill="white" font-size="14">+</text>
    
    <text x="250" y="470" text-anchor="middle" fill="#3498db" font-size="12">Bottom-Up</text>
    <text x="650" y="400" text-anchor="middle" fill="#2ecc71" font-size="12">Top-Down</text>
"""

diagrams['08_vit_architecture'] = """
    <text x="400" y="30" text-anchor="middle" font-size="24" font-weight="bold" fill="#2c3e50">Vision Transformer (ViT)</text>
    
    <!-- 输入图像 -->
    <rect x="30" y="180" width="100" height="100" fill="url(#blueGrad)" rx="5" filter="url(#shadow)"/>
    <text x="80" y="225" text-anchor="middle" fill="white" font-size="11">Image</text>
    <text x="80" y="245" text-anchor="middle" fill="white" font-size="10">224×224</text>
    
    <!-- Patch分割 -->
    <rect x="30" y="180" width="33" height="33" fill="#e74c3c" opacity="0.5"/>
    <rect x="63" y="180" width="34" height="33" fill="#e74c3c" opacity="0.5"/>
    <rect x="97" y="180" width="33" height="33" fill="#e74c3c" opacity="0.5"/>
    
    <!-- Patch Embedding -->
    <rect x="160" y="200" width="100" height="60" fill="url(#orangeGrad)" rx="5" filter="url(#shadow)"/>
    <text x="210" y="225" text-anchor="middle" fill="white" font-size="10">Patch</text>
    <text x="210" y="245" text-anchor="middle" fill="white" font-size="10">Embedding</text>
    
    <!-- 位置编码 -->
    <rect x="290" y="200" width="80" height="60" fill="url(#purpleGrad)" rx="5" filter="url(#shadow)"/>
    <text x="330" y="225" text-anchor="middle" fill="white" font-size="10">Position</text>
    <text x="330" y="245" text-anchor="middle" fill="white" font-size="10">Encoding</text>
    
    <!-- CLS Token -->
    <rect x="290" y="120" width="80" height="40" fill="url(#greenGrad)" rx="5" filter="url(#shadow)"/>
    <text x="330" y="145" text-anchor="middle" fill="white" font-size="10">[CLS] Token</text>
    
    <!-- Transformer Encoder -->
    <rect x="400" y="80" width="180" height="220" fill="url(#redGrad)" rx="10" filter="url(#shadow)"/>
    <text x="490" y="110" text-anchor="middle" fill="white" font-size="12" font-weight="bold">Transformer</text>
    <text x="490" y="125" text-anchor="middle" fill="white" font-size="12" font-weight="bold">Encoder × L</text>
    
    <rect x="420" y="145" width="140" height="35" fill="white" opacity="0.9" rx="5"/>
    <text x="490" y="168" text-anchor="middle" font-size="10" fill="#333">MSA</text>
    
    <rect x="420" y="195" width="140" height="35" fill="white" opacity="0.9" rx="5"/>
    <text x="490" y="218" text-anchor="middle" font-size="10" fill="#333">MLP</text>
    
    <rect x="420" y="245" width="140" height="35" fill="white" opacity="0.9" rx="5"/>
    <text x="490" y="268" text-anchor="middle" font-size="10" fill="#333">LayerNorm</text>
    
    <!-- MLP Head -->
    <rect x="620" y="200" width="80" height="60" fill="url(#greenGrad)" rx="5" filter="url(#shadow)"/>
    <text x="660" y="225" text-anchor="middle" fill="white" font-size="10">MLP</text>
    <text x="660" y="245" text-anchor="middle" fill="white" font-size="10">Head</text>
    
    <!-- 输出 -->
    <rect x="730" y="210" width="50" height="40" fill="url(#blueGrad)" rx="5"/>
    <text x="755" y="235" text-anchor="middle" fill="white" font-size="10">Class</text>
    
    <!-- 箭头 -->
    <path d="M130 230 L160 230" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>
    <path d="M260 230 L290 230" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>
    <path d="M330 160 L330 200" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>
    <path d="M370 230 L400 200" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>
    <path d="M580 200 L620 230" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>
    <path d="M700 230 L730 230" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>
"""

diagrams['09_deeplabv3_plus'] = """
    <text x="400" y="30" text-anchor="middle" font-size="24" font-weight="bold" fill="#2c3e50">DeepLabV3+ Architecture</text>
    
    <!-- Backbone -->
    <rect x="30" y="200" width="120" height="100" fill="url(#blueGrad)" rx="8" filter="url(#shadow)"/>
    <text x="90" y="240" text-anchor="middle" fill="white" font-size="11">Backbone</text>
    <text x="90" y="260" text-anchor="middle" fill="white" font-size="10">ResNet/Xception</text>
    
    <!-- ASPP -->
    <rect x="180" y="120" width="200" height="260" fill="url(#redGrad)" rx="10" filter="url(#shadow)"/>
    <text x="280" y="150" text-anchor="middle" fill="white" font-size="14" font-weight="bold">ASPP</text>
    
    <!-- ASPP分支 -->
    <rect x="200" y="170" width="60" height="30" fill="white" opacity="0.9" rx="3"/>
    <text x="230" y="190" text-anchor="middle" font-size="9" fill="#333">1×1 Conv</text>
    
    <rect x="200" y="210" width="60" height="30" fill="white" opacity="0.9" rx="3"/>
    <text x="230" y="230" text-anchor="middle" font-size="9" fill="#333">3×3 r=6</text>
    
    <rect x="200" y="250" width="60" height="30" fill="white" opacity="0.9" rx="3"/>
    <text x="230" y="270" text-anchor="middle" font-size="9" fill="#333">3×3 r=12</text>
    
    <rect x="200" y="290" width="60" height="30" fill="white" opacity="0.9" rx="3"/>
    <text x="230" y="310" text-anchor="middle" font-size="9" fill="#333">3×3 r=18</text>
    
    <rect x="200" y="330" width="60" height="30" fill="white" opacity="0.9" rx="3"/>
    <text x="230" y="350" text-anchor="middle" font-size="9" fill="#333">GAP</text>
    
    <!-- Concat -->
    <rect x="280" y="240" width="80" height="40" fill="url(#orangeGrad)" rx="5"/>
    <text x="320" y="265" text-anchor="middle" fill="white" font-size="10">Concat</text>
    
    <!-- Decoder -->
    <rect x="420" y="150" width="180" height="200" fill="url(#greenGrad)" rx="10" filter="url(#shadow)"/>
    <text x="510" y="180" text-anchor="middle" fill="white" font-size="14" font-weight="bold">Decoder</text>
    
    <rect x="440" y="200" width="140" height="30" fill="white" opacity="0.9" rx="3"/>
    <text x="510" y="220" text-anchor="middle" font-size="9" fill="#333">1×1 Conv (low-level)</text>
    
    <rect x="440" y="250" width="140" height="30" fill="white" opacity="0.9" rx="3"/>
    <text x="510" y="270" text-anchor="middle" font-size="9" fill="#333">Upsample ×4</text>
    
    <rect x="440" y="300" width="140" height="30" fill="white" opacity="0.9" rx="3"/>
    <text x="510" y="320" text-anchor="middle" font-size="9" fill="#333">3×3 Conv + Upsample</text>
    
    <!-- 输出 -->
    <rect x="640" y="220" width="100" height="60" fill="url(#purpleGrad)" rx="8" filter="url(#shadow)"/>
    <text x="690" y="250" text-anchor="middle" fill="white" font-size="11">Output</text>
    <text x="690" y="268" text-anchor="middle" fill="white" font-size="9">Segmentation</text>
    
    <!-- 箭头 -->
    <path d="M150 250 L180 250" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>
    <path d="M260 250 L280 260" stroke="#333" stroke-width="1.5"/>
    <path d="M260 310 L280 260" stroke="#333" stroke-width="1.5"/>
    <path d="M360 260 L420 250" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>
    <path d="M600 250 L640 250" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>
    
    <!-- Low-level连接 -->
    <path d="M90 200 Q90 140 440 215" stroke="#3498db" stroke-width="2" stroke-dasharray="5,5" fill="none" marker-end="url(#arrow)"/>
    <text x="250" y="130" text-anchor="middle" fill="#3498db" font-size="10">Low-level features</text>
"""

diagrams['10_segformer_architecture'] = """
    <text x="400" y="30" text-anchor="middle" font-size="24" font-weight="bold" fill="#2c3e50">SegFormer Architecture</text>
    
    <!-- 输入 -->
    <rect x="50" y="220" width="80" height="60" fill="url(#blueGrad)" rx="5" filter="url(#shadow)"/>
    <text x="90" y="255" text-anchor="middle" fill="white" font-size="11">Input</text>
    
    <!-- Hierarchical Transformer Encoder -->
    <rect x="160" y="80" width="280" height="340" fill="url(#redGrad)" rx="10" filter="url(#shadow)"/>
    <text x="300" y="110" text-anchor="middle" fill="white" font-size="12" font-weight="bold">Hierarchical Transformer</text>
    
    <!-- Stage 1 -->
    <rect x="180" y="130" width="100" height="50" fill="white" opacity="0.9" rx="5"/>
    <text x="230" y="155" text-anchor="middle" font-size="10" fill="#333">Stage 1</text>
    <text x="230" y="170" text-anchor="middle" font-size="8" fill="#666">H/4 × W/4</text>
    
    <!-- Stage 2 -->
    <rect x="180" y="200" width="100" height="50" fill="white" opacity="0.9" rx="5"/>
    <text x="230" y="225" text-anchor="middle" font-size="10" fill="#333">Stage 2</text>
    <text x="230" y="240" text-anchor="middle" font-size="8" fill="#666">H/8 × W/8</text>
    
    <!-- Stage 3 -->
    <rect x="180" y="270" width="100" height="50" fill="white" opacity="0.9" rx="5"/>
    <text x="230" y="295" text-anchor="middle" font-size="10" fill="#333">Stage 3</text>
    <text x="230" y="310" text-anchor="middle" font-size="8" fill="#666">H/16 × W/16</text>
    
    <!-- Stage 4 -->
    <rect x="180" y="340" width="100" height="50" fill="white" opacity="0.9" rx="5"/>
    <text x="230" y="365" text-anchor="middle" font-size="10" fill="#333">Stage 4</text>
    <text x="230" y="380" text-anchor="middle" font-size="8" fill="#666">H/32 × W/32</text>
    
    <!-- MLP Decode Head -->
    <rect x="480" y="120" width="200" height="280" fill="url(#greenGrad)" rx="10" filter="url(#shadow)"/>
    <text x="580" y="150" text-anchor="middle" fill="white" font-size="12" font-weight="bold">MLP Decode Head</text>
    
    <!-- MLP layers -->
    <rect x="500" y="170" width="80" height="35" fill="white" opacity="0.9" rx="3"/>
    <text x="540" y="192" text-anchor="middle" font-size="9" fill="#333">MLP</text>
    
    <rect x="500" y="220" width="80" height="35" fill="white" opacity="0.9" rx="3"/>
    <text x="540" y="242" text-anchor="middle" font-size="9" fill="#333">MLP</text>
    
    <rect x="500" y="270" width="80" height="35" fill="white" opacity="0.9" rx="3"/>
    <text x="540" y="292" text-anchor="middle" font-size="9" fill="#333">MLP</text>
    
    <rect x="500" y="320" width="80" height="35" fill="white" opacity="0.9" rx="3"/>
    <text x="540" y="342" text-anchor="middle" font-size="9" fill="#333">MLP</text>
    
    <!-- Upsample & Concat -->
    <rect x="590" y="195" width="70" height="135" fill="url(#orangeGrad)" rx="5"/>
    <text x="625" y="240" text-anchor="middle" fill="white" font-size="9">Upsample</text>
    <text x="625" y="255" text-anchor="middle" fill="white" font-size="9">&</text>
    <text x="625" y="270" text-anchor="middle" fill="white" font-size="9">Concat</text>
    
    <!-- Fusion -->
    <rect x="520" y="360" width="120" height="25" fill="url(#purpleGrad)" rx="3"/>
    <text x="580" y="377" text-anchor="middle" fill="white" font-size="9">Fusion</text>
    
    <!-- 输出 -->
    <rect x="720" y="220" width="60" height="60" fill="url(#blueGrad)" rx="5" filter="url(#shadow)"/>
    <text x="750" y="255" text-anchor="middle" fill="white" font-size="10">Output</text>
    
    <!-- 箭头 -->
    <path d="M130 250 L160 250" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>
    
    <path d="M280 155 L500 187" stroke="#333" stroke-width="1.5" marker-end="url(#arrow)"/>
    <path d="M280 225 L500 237" stroke="#333" stroke-width="1.5" marker-end="url(#arrow)"/>
    <path d="M280 295 L500 287" stroke="#333" stroke-width="1.5" marker-end="url(#arrow)"/>
    <path d="M280 365 L500 337" stroke="#333" stroke-width="1.5" marker-end="url(#arrow)"/>
    
    <path d="M580 385 L580 410" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>
    <path d="M660 260 L720 250" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>
"""

for name, content in diagrams.items():
    path = save_svg(f"{name}.svg", content)
    print(f"Generated: {path}")
