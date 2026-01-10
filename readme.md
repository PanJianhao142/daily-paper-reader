# 📖 3分钟建立你自己的 Daily Paper Reader

### 1. 复制项目
点击右上角的 **Fork** 按钮，支持修改仓库名。

### 2. 开启自动更新
注意下面这个链接需要在自己fork的仓库下点击：
[👉 点击这里前往 Actions 页面](../../actions) 
* 点击绿色的 **"I understand my workflows, go ahead and enable them"** 按钮。

### 3. 发布网站
[👉 点击这里前往 Pages 设置页面](../../settings/pages)
* **Source** (来源): 选择 `Deploy from a branch`
* **Branch** (分支): 选择 `main`，文件夹选 `/ (root)`
* 点击 **Save** (保存)
* 等待**一分钟**~刷新，在页面上方获得网站地址，一般为：{username}.github.io/daily-paper-reader

### 4. 进入网站页面
* 地址栏中输入上一步获得地址

### 5. 订阅柏拉图
柏拉图它是个相当便宜的API聚合平台，这里会用到它的reranker模型，是0.001￥发起一次qwen3 4b的reranker请求。
以及会集成其中的GPT5, Gemini3 Flash, Gemini3 pro, Deepseek v3.2exp作为聊天模型使用。
#### 5.2 注册
[blt](https://api.bltcy.ai/register?aff=wrM957407) 
使用当前链接注册
#### 5.3 充值
点击右上角用户头像——立即充值——充值5元
#### 5.4 令牌
在同一个后台页面，点击左侧令牌——新建令牌——填写名称，可全默认。


---
**🎉 恭喜！等待约 1 分钟，刷新页面，你就会看到顶部的链接，那就是你的专属网站！**




写技术的时候把原因和考虑写上：
https://huggingface.co/Qwen/Qwen3-Reranker-0.6B  这里面有benchmark，比较了BGE-reranker-v2-m3 Jina-multilingual-reranker-v2-base gte-multilingual-reranker-base

https://huggingface.co/Qwen/Qwen3-Embedding-0.6B embedding也用qwen3，0.6b体量下打败了BGE-M3 multilingual-e5-large-instruct


#TODO
[] 自动sync 不需要手动点sync同步上游代码
[] 关键词模糊检索，先让大模型生成模糊关键词，再去匹配
[] 云聊天记录，公共加密云数据库
[] 会议的纳入
[] 夜间模式
[] 期刊的纳入


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=ziwenhahaha/daily-paper-reader&type=date&legend=top-left)](https://www.star-history.com/#ziwenhahaha/daily-paper-reader&type=date&legend=top-left)
