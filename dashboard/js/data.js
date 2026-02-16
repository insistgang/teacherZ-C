const ResearchData = {
  profile: {
    name: "张三",
    institution: "清华大学计算机系",
    researchYears: 12,
    totalPapers: 156,
    totalCitations: 8420,
    hIndex: 42,
    i10Index: 89
  },

  papers: [
    { id: 1, title: "深度学习在图像识别中的应用研究", year: 2024, citations: 45, journal: "Nature", field: "计算机视觉", authors: ["张三", "李四", "王五"], impact: 9.8 },
    { id: 2, title: "基于Transformer的自然语言处理方法", year: 2024, citations: 32, journal: "Science", field: "自然语言处理", authors: ["张三", "赵六"], impact: 9.2 },
    { id: 3, title: "强化学习在机器人控制中的应用", year: 2023, citations: 128, journal: "IEEE TPAMI", field: "机器人学", authors: ["张三", "李四"], impact: 8.5 },
    { id: 4, title: "图神经网络: 方法与应用", year: 2023, citations: 256, journal: "JMLR", field: "机器学习", authors: ["张三", "王五", "孙七"], impact: 8.2 },
    { id: 5, title: "联邦学习隐私保护机制研究", year: 2023, citations: 89, journal: "ACM Computing", field: "隐私计算", authors: ["张三"], impact: 7.8 },
    { id: 6, title: "大规模预训练模型综述", year: 2022, citations: 520, journal: "ACM CSUR", field: "机器学习", authors: ["张三", "李四", "赵六", "王五"], impact: 16.6 },
    { id: 7, title: "知识图谱构建与应用", year: 2022, citations: 312, journal: "IEEE TKDE", field: "知识工程", authors: ["张三", "孙七"], impact: 8.9 },
    { id: 8, title: "多模态学习方法研究", year: 2022, citations: 198, journal: "CVPR", field: "多模态学习", authors: ["张三", "李四"], impact: 12.0 },
    { id: 9, title: "目标检测算法优化", year: 2021, citations: 425, journal: "ICCV", field: "计算机视觉", authors: ["张三", "王五"], impact: 11.5 },
    { id: 10, title: "生成对抗网络进展", year: 2021, citations: 356, journal: "NeurIPS", field: "生成模型", authors: ["张三", "李四", "赵六"], impact: 12.8 },
    { id: 11, title: "推荐系统算法研究", year: 2020, citations: 289, journal: "KDD", field: "推荐系统", authors: ["张三"], impact: 10.2 },
    { id: 12, title: "时间序列预测方法", year: 2020, citations: 178, journal: "AAAI", field: "机器学习", authors: ["张三", "孙七"], impact: 9.5 },
    { id: 13, title: "文本分类深度学习方法", year: 2019, citations: 534, journal: "ACL", field: "自然语言处理", authors: ["张三", "李四"], impact: 14.2 },
    { id: 14, title: "迁移学习理论分析", year: 2019, citations: 467, journal: "ICML", field: "机器学习", authors: ["张三", "王五", "赵六"], impact: 13.5 },
    { id: 15, title: "卷积神经网络架构设计", year: 2018, citations: 892, journal: "IEEE TPAMI", field: "深度学习", authors: ["张三"], impact: 16.2 },
  ],

  fields: ["计算机视觉", "自然语言处理", "机器学习", "深度学习", "机器人学", "知识工程", "隐私计算", "推荐系统", "生成模型", "多模态学习"],

  journals: ["Nature", "Science", "IEEE TPAMI", "JMLR", "ACM Computing", "ACM CSUR", "IEEE TKDE", "CVPR", "ICCV", "NeurIPS", "KDD", "AAAI", "ACL", "ICML"],

  collaborators: [
    { name: "李四", institution: "北京大学", collaborations: 12, avatar: "L" },
    { name: "王五", institution: "中科院", collaborations: 8, avatar: "W" },
    { name: "赵六", institution: "浙江大学", collaborations: 6, avatar: "Z" },
    { name: "孙七", institution: "复旦大学", collaborations: 5, avatar: "S" },
    { name: "周八", institution: "上海交通大学", collaborations: 3, avatar: "Z" },
    { name: "吴九", institution: "南京大学", collaborations: 2, avatar: "W" },
  ],

  citationsByYear: {
    2018: 320, 2019: 580, 2020: 890, 2021: 1250, 2022: 1680, 2023: 2100, 2024: 1600
  },

  papersByYear: {
    2013: 5, 2014: 8, 2015: 10, 2016: 12, 2017: 14, 2018: 15, 2019: 16, 2020: 18, 2021: 16, 2022: 14, 2023: 15, 2024: 13
  },

  fieldEvolution: [
    { period: "2013-2015", fields: { "机器学习": 8, "计算机视觉": 5, "自然语言处理": 3, "深度学习": 2 } },
    { period: "2016-2018", fields: { "机器学习": 12, "计算机视觉": 10, "深度学习": 8, "自然语言处理": 6 } },
    { period: "2019-2021", fields: { "深度学习": 15, "计算机视觉": 14, "自然语言处理": 12, "机器学习": 10, "生成模型": 8 } },
    { period: "2022-2024", fields: { "深度学习": 18, "自然语言处理": 16, "多模态学习": 12, "计算机视觉": 10, "隐私计算": 8, "知识工程": 6 } },
  ],

  heatmapData: {
    methods: ["深度学习", "强化学习", "迁移学习", "联邦学习", "图神经网络"],
    applications: ["计算机视觉", "自然语言处理", "推荐系统", "机器人学", "医疗健康"],
    values: [
      [45, 38, 22, 8, 28],
      [12, 8, 5, 3, 15],
      [18, 22, 10, 4, 12],
      [8, 5, 6, 15, 10],
      [25, 20, 15, 5, 18]
    ]
  },

  comparisonPeers: [
    { name: "同行A", papers: 142, citations: 7800, hIndex: 38, field: "计算机视觉" },
    { name: "同行B", papers: 168, citations: 9100, hIndex: 45, field: "机器学习" },
    { name: "同行C", papers: 130, citations: 6500, hIndex: 35, field: "自然语言处理" },
    { name: "自己", papers: 156, citations: 8420, hIndex: 42, field: "综合" },
  ],

  networkNodes: [
    { id: "main", name: "张三", citations: 8420, field: "综合", isMain: true },
    { id: "p1", name: "深度学习研究", citations: 1200, field: "深度学习" },
    { id: "p2", name: "计算机视觉研究", citations: 980, field: "计算机视觉" },
    { id: "p3", name: "自然语言处理研究", citations: 850, field: "自然语言处理" },
    { id: "p4", name: "机器学习研究", citations: 1100, field: "机器学习" },
    { id: "p5", name: "推荐系统研究", citations: 450, field: "推荐系统" },
    { id: "p6", name: "知识图谱研究", citations: 380, field: "知识工程" },
    { id: "c1", name: "李四", citations: 3200, field: "机器学习" },
    { id: "c2", name: "王五", citations: 2800, field: "计算机视觉" },
    { id: "c3", name: "赵六", citations: 2100, field: "自然语言处理" },
  ],

  networkLinks: [
    { source: "main", target: "p1", weight: 25 },
    { source: "main", target: "p2", weight: 18 },
    { source: "main", target: "p3", weight: 15 },
    { source: "main", target: "p4", weight: 20 },
    { source: "main", target: "p5", weight: 8 },
    { source: "main", target: "p6", weight: 6 },
    { source: "p1", target: "p2", weight: 10 },
    { source: "p1", target: "p4", weight: 12 },
    { source: "p2", target: "p4", weight: 8 },
    { source: "main", target: "c1", weight: 12 },
    { source: "main", target: "c2", weight: 8 },
    { source: "main", target: "c3", weight: 6 },
    { source: "c1", target: "p4", weight: 15 },
    { source: "c2", target: "p2", weight: 12 },
    { source: "c3", target: "p3", weight: 10 },
  ]
};

const FieldColors = {
  "计算机视觉": "#3b82f6",
  "自然语言处理": "#10b981",
  "机器学习": "#8b5cf6",
  "深度学习": "#f59e0b",
  "机器人学": "#ef4444",
  "知识工程": "#06b6d4",
  "隐私计算": "#ec4899",
  "推荐系统": "#14b8a6",
  "生成模型": "#f97316",
  "多模态学习": "#6366f1",
  "综合": "#64748b"
};

function getFieldColor(field) {
  return FieldColors[field] || "#64748b";
}
