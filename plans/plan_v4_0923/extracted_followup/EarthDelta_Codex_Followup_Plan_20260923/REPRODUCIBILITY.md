# 复现与安全执行

## 身份与读取

- 原计划实现 `a3596e6b804e9d23b72d1247b08c47129d6b50b1`，比较 `d749a1c62521226df857587e08f7d067b0f15355`；本次观测HEAD `fdd92d79b1b03e8397d4c487d7fa4a347ae6fc37`，关键代码树相同。
- 在已有repo执行`git rev-parse HEAD`/`git status`/差异和source hash读取；不要reset、强制checkout、stash/drop或覆盖用户工作区。
- 原计划/旧审计/新F0计划分别存档。执行只认已登记protocol，不能挑更容易PASS的定义。
- 实际module.__file__及其hash要入source manifest；只写commit标签不足。模型架构、actual state、NPZ源精度/算术版本、input/coordinates均绑定。

## 环境

保存Python、Torch/torchvision/lightning/timm/xformers及CUDA/driver、算子实际前反向、TF32设置、determinism、设备名、seed、CPU线程数、导入路径。
复用现有授权环境和工具，必要依赖缺失则BLOCKED；不在共享环境盲目pip升级或伪造导入stub来认证官方执行。
优先单GPU串行S0；以后多个worker必须独立进程/独立模型实例，公共Fs文件只读。

## 随机性和数据

固定Fs初始化、训练采样次序、候选checkpoint、选择规则、seed与source。不同worker的Fs摘要必须相同，不以seed相同代替权重相同。
完整UTC support做purge；标记过去所有debug/训练/调查曝光。2020等曾被历史训练/人工看过的时间，不自动成为未触碰confirm。
只读取实际可获得的t-12/t-6/t作为策略输入；未来标签只在训练资格/离线评分允许路径出现。

## 时间/资源计账

保存submit/start/end UTC、排队、进程GPU时间、节点占用、前向/反向/候选轨迹数、峰值显存、磁盘和失败尝试。
`estimated_minutes=null`直到实测profile；估算必须保留公式/观测来源。原J0/J1/J2时长是管理cap，不是实际已运行时间。

## 证据出版

原证据只读；新source或config用新run目录和新父证书链；所有JSON禁止NaN，缺字段写null+MISSING，不写伪0。
保存失败raw日志，不只summary。若数据/weights太大，可保留在Codex可访问的持久存储并提交hash/路径；不得伪造sandbox下载地址。
包自身文件SHA256由SHA256SUMS.json校验，ZIP SHA在外部.sha256；不在ZIP内自引用整个ZIP的SHA。
