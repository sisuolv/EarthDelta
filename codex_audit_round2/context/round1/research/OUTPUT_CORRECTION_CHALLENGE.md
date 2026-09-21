# 为什么参数编辑不是天然必需？

对任意一个具体编辑后的单步映射Fa，都可以定义Ca(x)=Fa(x)−Fref(x)。那么Fa(x)=Fref(x)+Ca(x)。因此，表达能力不受限的、将修正反馈到下一步的输出/状态corrector可以完全复现同一轨迹。这是一个存在性恒等式，不代表低成本网络一定学得出Ca。

所以参数编辑可争取的是有限数据/算力下的归纳偏置、权重共享和表示效率，不是“只有它能改变动力学”。同一个输出corrector也可以跨步共享、联合预测所有变量，并通过后续Fref产生一致响应。

若I含参考forecast且平方误差目标固定，E[e0|I]+Fref=E[Y|I]。预测得足够好的e0直接产生强残差校正。又若e0在span(du)之外有不可预测噪声，选择只需要其相关投影，不必完整重建未来；P0四格因此需同时报告全场和响应方向误差。

必需挑战：
- 末端多lead多变量残差网络（不是弱的单变量one-step）；
- 迭代全变量feedbackcorrector（同history、hold、multisteploss、总成本）；
- virtual editedforecast=reference+preddu（检验是否还值得运行被编辑backbone）；
- 同状态编码器、同字典的directrouter。

参数方法赢的合格证据：相同总资源下在未见天气或未见动作有更好MSE/稳定性；或达到相同skill所需fitlabel显著更少；或一致的跨lead/变量收益而输出基线在充分调参后仍达不到。观察到风场图更平滑、某个变量更准、一次headroom更高，都不够。
