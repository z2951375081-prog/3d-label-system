# 数据集说明

`data/` 中的 OBJ 三维模型使用 Git LFS 管理。服务器上克隆仓库后，请先安装并初始化 Git LFS：

```bash
git lfs install
git lfs pull
```

完成后即可运行数据集检查和训练命令。不要把数据目录移动到项目外部，否则项目中的数据路径会失效。
