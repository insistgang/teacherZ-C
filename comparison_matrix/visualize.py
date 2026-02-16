import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Patch

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

def plot_method_task_heatmap():
    df = pd.read_csv('method_task.csv', index_col=0)
    
    fig, ax = plt.subplots(figsize=(10, 12))
    
    cmap = sns.color_palette(["#f5f5f5", "#2ecc71"], as_cmap=True)
    
    sns.heatmap(df, annot=True, cmap=cmap, cbar_kws={'label': '支持 (1) / 不支持 (0)'},
                linewidths=0.5, linecolor='white', ax=ax,
                annot_kws={'size': 12, 'weight': 'bold'})
    
    ax.set_title('方法×任务 覆盖矩阵', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('任务类型', fontsize=12)
    ax.set_ylabel('方法', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('heatmap_method_task.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("已保存: heatmap_method_task.png")

def plot_method_data_heatmap():
    df = pd.read_csv('method_data.csv', index_col=0)
    
    fig, ax = plt.subplots(figsize=(10, 12))
    
    cmap = sns.color_palette(["#f5f5f5", "#3498db"], as_cmap=True)
    
    sns.heatmap(df, annot=True, cmap=cmap, cbar_kws={'label': '支持 (1) / 不支持 (0)'},
                linewidths=0.5, linecolor='white', ax=ax,
                annot_kws={'size': 12, 'weight': 'bold'})
    
    ax.set_title('方法×数据类型 适配矩阵', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('数据类型', fontsize=12)
    ax.set_ylabel('方法', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('heatmap_method_data.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("已保存: heatmap_method_data.png")

def plot_method_features_heatmap():
    df = pd.read_csv('method_features.csv', index_col=0)
    
    fig, ax = plt.subplots(figsize=(10, 12))
    
    cmap = sns.color_palette(["#f5f5f5", "#9b59b6"], as_cmap=True)
    
    sns.heatmap(df, annot=True, cmap=cmap, cbar_kws={'label': '具备 (1) / 不具备 (0)'},
                linewidths=0.5, linecolor='white', ax=ax,
                annot_kws={'size': 12, 'weight': 'bold'})
    
    ax.set_title('方法×技术特性 矩阵', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('技术特性', fontsize=12)
    ax.set_ylabel('方法', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('heatmap_method_features.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("已保存: heatmap_method_features.png")

def plot_paper_metrics_radar():
    df = pd.read_csv('paper_metrics.csv')
    
    df_norm = df.copy()
    df_norm['mIoU(%)'] = df['mIoU(%)'] / 100
    df_norm['mAP(%)'] = df['mAP(%)'] / 100
    df_norm['PSNR'] = df['PSNR'] / 35
    df_norm['Parameters(M)'] = 1 - df['Parameters(M)'] / 120
    df_norm['Speed(fps)'] = df['Speed(fps)'] / 100
    df_norm = df_norm.fillna(0)
    
    categories = ['mIoU', 'mAP', 'PSNR', '轻量化', '速度']
    N = len(categories)
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(df)))
    
    for idx, (_, row) in enumerate(df_norm.iterrows()):
        values = [row['mIoU(%)'], row['mAP(%)'], row['PSNR'], 
                  row['Parameters(M)'], row['Speed(fps)']]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=row['Paper'], color=colors[idx])
        ax.fill(angles, values, alpha=0.1, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title('论文性能雷达图对比', fontsize=14, fontweight='bold', pad=20)
    
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)
    
    plt.tight_layout()
    plt.savefig('radar_paper_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("已保存: radar_paper_metrics.png")

def plot_paper_metrics_bar():
    df = pd.read_csv('paper_metrics.csv')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax1 = axes[0, 0]
    mIoU_data = df.dropna(subset=['mIoU(%)'])
    bars = ax1.barh(mIoU_data['Paper'], mIoU_data['mIoU(%)'], color='#3498db')
    ax1.set_xlabel('mIoU (%)', fontsize=11)
    ax1.set_title('分割性能 (mIoU)', fontsize=12, fontweight='bold')
    ax1.set_xlim(0, 100)
    for bar, val in zip(bars, mIoU_data['mIoU(%)']):
        ax1.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.1f}', 
                 va='center', fontsize=9)
    
    ax2 = axes[0, 1]
    mAP_data = df.dropna(subset=['mAP(%)'])
    bars = ax2.barh(mAP_data['Paper'], mAP_data['mAP(%)'], color='#e74c3c')
    ax2.set_xlabel('mAP (%)', fontsize=11)
    ax2.set_title('检测性能 (mAP)', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 100)
    for bar, val in zip(bars, mAP_data['mAP(%)']):
        ax2.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.1f}', 
                 va='center', fontsize=9)
    
    ax3 = axes[1, 0]
    bars = ax3.barh(df['Paper'], df['Parameters(M)'], color='#9b59b6')
    ax3.set_xlabel('参数量 (M)', fontsize=11)
    ax3.set_title('模型参数量', fontsize=12, fontweight='bold')
    for bar, val in zip(bars, df['Parameters(M)']):
        ax3.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.1f}M', 
                 va='center', fontsize=9)
    
    ax4 = axes[1, 1]
    bars = ax4.barh(df['Paper'], df['Speed(fps)'], color='#2ecc71')
    ax4.set_xlabel('推理速度 (fps)', fontsize=11)
    ax4.set_title('推理速度', fontsize=12, fontweight='bold')
    for bar, val in zip(bars, df['Speed(fps)']):
        ax4.text(val + 10, bar.get_y() + bar.get_height()/2, f'{val}', 
                 va='center', fontsize=9)
    
    plt.suptitle('论文性能指标对比', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('bar_paper_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("已保存: bar_paper_metrics.png")

def plot_category_coverage():
    df_task = pd.read_csv('method_task.csv', index_col=0)
    df_data = pd.read_csv('method_data.csv', index_col=0)
    
    task_coverage = df_task.sum()
    data_coverage = df_data.sum()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1 = axes[0]
    colors1 = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    wedges, texts, autotexts = ax1.pie(task_coverage, labels=task_coverage.index, 
                                        autopct='%1.0f%%', colors=colors1,
                                        explode=[0.05]*len(task_coverage))
    ax1.set_title('任务类型覆盖分布', fontsize=12, fontweight='bold')
    
    ax2 = axes[1]
    colors2 = ['#9b59b6', '#1abc9c', '#e67e22', '#34495e']
    wedges, texts, autotexts = ax2.pie(data_coverage, labels=data_coverage.index,
                                        autopct='%1.0f%%', colors=colors2,
                                        explode=[0.05]*len(data_coverage))
    ax2.set_title('数据类型覆盖分布', fontsize=12, fontweight='bold')
    
    plt.suptitle('方法覆盖度分析', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('pie_coverage.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("已保存: pie_coverage.png")

def plot_tech_evolution():
    papers = {
        'Year': [2015, 2016, 2017, 2017, 2017, 2018, 2019, 2021, 2021, 2022],
        'Method': ['U-Net', 'V-Net', 'PointNet', 'PointNet++', 'Mask R-CNN', 
                   'DeepLabv3+', 'DGCNN', 'TransUNet', 'nnU-Net', 'Swin-UNet'],
        'Category': ['CNN', 'CNN', 'Point Cloud', 'Point Cloud', 'CNN',
                     'CNN', 'Point Cloud', 'Transformer', 'CNN', 'Transformer'],
        'Impact': [95, 78, 92, 88, 85, 82, 75, 70, 80, 65]
    }
    df = pd.DataFrame(papers)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    categories = df['Category'].unique()
    colors = {'CNN': '#3498db', 'Point Cloud': '#e74c3c', 'Transformer': '#2ecc71'}
    
    for cat in categories:
        subset = df[df['Category'] == cat]
        ax.scatter(subset['Year'], subset['Impact'], s=subset['Impact']*5, 
                   c=colors[cat], label=cat, alpha=0.7, edgecolors='black', linewidth=1)
        
        for _, row in subset.iterrows():
            ax.annotate(row['Method'], (row['Year'], row['Impact']),
                       xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax.set_xlabel('发表年份', fontsize=12)
    ax.set_ylabel('影响力指数', fontsize=12)
    ax.set_title('技术演进趋势图', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(2014, 2023)
    ax.set_ylim(60, 100)
    
    plt.tight_layout()
    plt.savefig('scatter_evolution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("已保存: scatter_evolution.png")

def generate_all():
    print("开始生成可视化图表...\n")
    plot_method_task_heatmap()
    plot_method_data_heatmap()
    plot_method_features_heatmap()
    plot_paper_metrics_radar()
    plot_paper_metrics_bar()
    plot_category_coverage()
    plot_tech_evolution()
    print("\n所有图表生成完成!")

if __name__ == '__main__':
    generate_all()
