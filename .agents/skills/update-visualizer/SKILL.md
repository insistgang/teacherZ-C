---
name: update-visualizer
description: |
  Update the Xiaohao Cai paper visualizer website and deploy to GitHub Pages.
  Use when: updating website code, fixing UI bugs, adding new features, or syncing changes to docs/.
author: Claude Code Academic Workflow
version: 1.0.0
argument-hint: "[commit-message]"
---

# Update Visualizer Website

Sync changes from `visualizer_complete/` to `docs/` and deploy to GitHub Pages.

## Directory Structure

```
zx/
├── visualizer_complete/    # 开发目录 (在这里修改代码)
│   ├── index.html
│   ├── app.js
│   ├── style.css
│   ├── data.js
│   ├── 00_papers/         # PDF 文件
│   └── notes/             # 精读笔记
│
└── docs/                   # GitHub Pages 部署目录
    ├── index.html
    ├── app.js
    ├── style.css
    ├── data.js
    ├── 00_papers/
    └── notes/
```

## Workflow Steps

### 1. Modify Code in visualizer_complete/

```bash
# 在 visualizer_complete/ 目录修改代码
# 例如: 修改 app.js 中的 Markdown 解析器
# 例如: 修改 style.css 中的样式
# 例如: 修改 index.html 中的布局
```

### 2. Sync to docs/ Directory

```bash
cd /d/Documents/zx

# 同步主要文件
cp visualizer_complete/index.html docs/
cp visualizer_complete/app.js docs/
cp visualizer_complete/style.css docs/
cp visualizer_complete/data.js docs/
```

### 3. Commit and Push to GitHub

```bash
git add docs/
git commit -m "更新网站: [描述修改内容]"
git push origin main
```

### 4. Verify Deployment

- GitHub Pages URL: `https://insistgang.github.io/teacherZ-C/`
- Wait 1-2 minutes for deployment
- Refresh browser to see changes

## Common Updates

### Update Name/Text
```bash
# 例如修正姓名
sed -i 's/蔡晓昊/蔡晓浩/g' visualizer_complete/*.html visualizer_complete/*.js
cp visualizer_complete/*.html visualizer_complete/*.js docs/
```

### Update Styles
```bash
# 修改 style.css 后同步
cp visualizer_complete/style.css docs/
```

### Update JavaScript Logic
```bash
# 修改 app.js 后同步 (Markdown解析、目录导航等)
cp visualizer_complete/app.js docs/
```

## Quick Commands

```bash
# 完整更新流程
cd /d/Documents/zx && \
cp visualizer_complete/index.html docs/ && \
cp visualizer_complete/app.js docs/ && \
cp visualizer_complete/style.css docs/ && \
cp visualizer_complete/data.js docs/ && \
git add docs/ && \
git commit -m "同步完整版可视化系统到docs目录" && \
git push origin main
```

## Troubleshooting

### Changes not showing on GitHub Pages
1. Check if changes were pushed: `git log origin/main`
2. Wait 1-2 minutes for GitHub Actions
3. Hard refresh browser (Ctrl+Shift+R)

### LaTeX formulas not rendering
- Check console for KaTeX errors
- Verify KaTeX scripts loaded in index.html
- Check markdownToHtml() function in app.js

### Tables not parsing correctly
- Check marked.js is loaded
- Verify GFM tables enabled in marked config
- Check table CSS in style.css

## References
- GitHub Repo: https://github.com/insistgang/teacherZ-C
- Live Site: https://insistgang.github.io/teacherZ-C/
