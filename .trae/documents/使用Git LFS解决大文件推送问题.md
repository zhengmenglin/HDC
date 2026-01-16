## 使用Git LFS解决大文件推送问题

### 步骤1: 安装和配置Git LFS
```bash
git lfs install
```

### 步骤2: 将大文件类型添加到Git LFS跟踪
```bash
git lfs track "*.pkl"
git lfs track "*.pth"
git lfs track "*.pth.tar"
```

### 步骤3: 提交.gitattributes文件
```bash
git add .gitattributes
git commit -m "Add Git LFS tracking for model files"
```

### 步骤4: 迁移现有大文件到Git LFS
```bash
git lfs migrate import --include="*.pkl,*.pth" --everything
```

### 步骤5: 强制推送到远程仓库
```bash
git push origin master --force
```

### 可选步骤：如果迁移失败，可以删除大文件后重新提交
```bash
# 删除本地大文件
git rm --cached models/*.pkl pipeline_test_results/*.pth
git commit -m "Remove large files from Git tracking"

# 添加到.gitignore
echo "*.pkl" >> .gitignore
echo "*.pth" >> .gitignore
git add .gitignore
git commit -m "Add model files to gitignore"

# 推送
git push origin master
```