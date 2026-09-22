"""Build an evidence index; retrieval alone is never counted as paper review."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Read scopes are recorded manually after reading the named primary-text sections.
CARDS = [
    ("2609.20051v1", "DART", "CV/video diffusion", "METHOD_AND_SELECTED_RESULTS",
     "Sections 3-5, limitations; complete Section 4", "目标日程上的配对响应校准；分开功能保留与质量",
     "复用幅度/hold/时效变化的响应模型，计入前向校准成本",
     "非天气；有限局部响应模型，部分 adapter 仍失败，原文主表排除了近零源收益 adapter"),
    ("2609.08618v1", "L-State", "NLP/training response", "METHOD_AND_SELECTED_RESULTS",
     "Sections 3.1-3.3, 3.6, 4; selected introduction/results",
     "共同初态、标准微干预；自由读出与结构算子读出对照",
     "共同 Fs 上的行为探针；按动作/目标留出评价，而非仅响应 RMSE",
     "原文是微训练而非固定编辑 rollout；跨 family 共享坐标有失败边界"),
    ("2603.15990v1", "W2T", "NLP/LoRA representation", "METHOD_AND_SELECTED_RESULTS",
     "Canonicalization and encoder sections; selected Sections 3-4",
     "处理 LoRA 因子不唯一性，从权重读出能力",
     "先用 DeltaW 的固定 sketch，再测试行为探针；不先建大型 encoder",
     "四专家不足以验证 meta 泛化；SVD 符号/重根约定须明确"),
    ("2608.22370v2", "LiST", "NLP/VLM adapter composition", "METHOD_AND_SELECTED_RESULTS",
     "Sections 3.2-3.4, selected ablations and limitations",
     "参数与行为联合描述、局部候选、接受与回退",
     "复用离线行为描述与经验性回退，对照方法获得相同信息",
     "在线 CMA-ES/随机前向有成本；代理能量下降不是天气准确率证书"),
    ("2608.11499v1", "HyperFix", "CV/model merging", "METHOD_AND_SELECTED_RESULTS",
     "Complete Method section; selected theory, experiment and limitation paragraphs",
     "低阶子集训练组合残差，减少逐组合调参",
     "先量测实际响应非加性，必要时增加小型交互残差",
     "原文修正权重，条件是局部小扰动；不自动保证长程天气响应"),
    ("2609.09148v3", "Proxy Policy Steering", "CV/robotics", "METHOD_AND_SELECTED_RESULTS",
     "Sections 4.1-4.3, selected 5.3/limitations/Appendix D",
     "匹配参考代理与任务代理的差分，沿生成轨迹施加",
     "共享编码器差分蒸馏、virtual edit 和逐步反馈强基线",
     "flow-matching 假设不直接适用 Stormer；摘要/正文增益数字不一致，本文不转述该数字"),
    ("2509.22020v2", "WeatherPEFT", "weather adaptation", "METHOD_AND_SELECTED_RESULTS",
     "Sections 4.1-4.2 (TADP/SFAS)", "任务 prompt 与参数敏感性驱动适配",
     "作为天气 PEFT 近邻，明确本项目的逐起报响应与决策贡献",
     "不是逐起报预测所有有限编辑的未来响应；本文未复现其结果"),
    ("2603.19325v1", "TaCT", "weather adaptation", "METHOD_AND_SELECTED_RESULTS",
     "Sections 3.2-3.3", "SAE 概念激活门控 LoRA/adapter 残差",
     "比较简单 regime/风险门，不把条件化天气编辑本身称为新概念",
     "不能据概念门控推断本项目的响应预测或多时效收益成立"),
    ("2608.02528v1", "VI-MoLE", "NLP/LoRA acquisition", "METHOD_AND_SELECTED_RESULTS",
     "Sections 3-4", "pre-acquisition risk/value head 与预算分配、abstention",
     "把执行前价值估计视为已有先例，强调天气目标和实际成本证据",
     "实验/发布计划不等于完整数值复现；风险上界之差不能自动当增益下界"),
    ("2605.14546v1", "CCM", "PDE adaptation", "METHOD_AND_SELECTED_RESULTS",
     "Section 3", "same-anchor PDE experts 与坐标组合",
     "所有天气候选共享 Fs；检验响应字典结构而非给月份专家强加语义",
     "受控 PDE 参数方向不自动变成真实大气机制"),
    ("2609.12278v1", "CLAW", "world models/RL", "METHOD_AND_SELECTED_RESULTS",
     "Section 4", "transition context 到分块 LoRA 的 hypernetwork",
     "作为直接条件化适配的挑战者，暂缓重训 backbone/hypernetwork",
     "原文联合预训练设置与当前固定 Stormer 不同"),
    ("2609.13257v1", "CVA", "CV/video world models", "SELECTED_METHOD_RESULTS_LIMITATIONS",
     "Selected compute accounting, selector, results and limitation paragraphs",
     "oracle headroom、合法选择收益与全部成本分开评价",
     "完整候选表、cheap legal、serving 成本和失败分母",
     "存在回顾性/重分析限制；采用实验纪律，不转述为普遍定理"),
    ("2609.10954v1", "Counterfactual update utility protocol", "world models", "METHOD_AND_SELECTED_RESULTS",
     "Method Section 3", "update/hold 配对分支、共同随机性、保留失败尝试",
     "Fs 稳定性对照和资格面板，全部尝试保留",
     "固定更新机制的因果比较范围有限，不证明任意适配机制有效"),
    ("2609.21740v1", "Sandwich-Residuals", "CV/world-model adaptation", "SELECTED_METHOD_RESULTS_LIMITATIONS",
     "Selected method, matched-budget evaluation and limitation paragraphs",
     "冻结视觉 world model 周围的小型残差适配",
     "认真比较反馈/状态订正，避免声称只有参数编辑能改变轨迹",
     "本文未逐式核验全部结构，天气迁移仍是待检验假设"),
    ("2609.18381v1", "Learned Atmospheric Critic", "weather evaluation", "METHOD_AND_SELECTED_RESULTS",
     "Critic construction, evaluation setup and limitations",
     "学习的分布真实性指标补充固定指标盲点",
     "主目标冻结后可作附加诊断，不增加 GAN 训练",
     "边际真实性不等于逐样本预报准确性；四个地面变量及专用 critic 范围有限"),
    ("2606.19549v1", "MergeProbe", "NLP/model merging", "METHOD_AND_SELECTED_RESULTS",
     "Sections 2.4-4, experiment provenance and selected appendices",
     "早期权重/梯度/激活信号预测合并后效用；按 adapter/domain 划分",
     "借鉴便宜描述与防止 pair-row 泄漏；新库按库留出",
     "原文明示部分结果来自受控 simulator/pilot；不作为成熟大规模性能证据"),
    ("2604.15557v1", "LAP", "NLP/steering", "SELECTED_METHOD_RESULTS_LIMITATIONS",
     "Selected formulation, Section 4.2 and depth-control paragraphs",
     "以读出对齐预测 steering 有效层，而非仅静态激活强度",
     "训练侧比较 block 的效果/代价，用于有预算的字典改良",
     "可读出性不自动给出天气收益或 off-target 保证"),
    ("2609.22782v1", "Look Before You Steer", "NLP/SAE steering", "METHOD_AND_SELECTED_RESULTS",
     "Sections 3-4.1 and abstract boundary conditions",
     "几何量能部分预测干预幅度，存在架构/层深边界",
     "把便宜权重描述作为基线，比较实际行为探针是否额外有用",
     "实验 feature 筛选含输出因果过滤；不声称整套过程完全无需前向或提供风险保证"),
    ("2606.14222v1", "ORCA", "time-series adaptation", "ABSTRACT_ONLY",
     "API abstract; HTML retrieved but not method-audited",
     "输入和基础输出的误差上下文用于黑盒残差适配",
     "后续反馈订正基线的实现候选",
     "仅摘要级定位；实现前需读方法并确认因果可用信息"),
    ("2609.03937v1", "RATL", "time-series adaptation", "ABSTRACT_ONLY",
     "API abstract; HTML retrieved but not method-audited",
     "训练侧 residual memory 与路由",
     "若合法状态信息不足，后续才考虑已成熟真值的历史残差",
     "不在首轮引入 memory；时间可用性需独立核查"),
    ("2506.06105v2", "Text-to-LoRA", "NLP/hypernetworks", "ABSTRACT_ONLY",
     "API abstract; v1/v2 HTML retrieved but not method-audited",
     "自然语言任务描述生成 LoRA",
     "说明 descriptor-to-adapter 的成熟方向；本轮优先效果预测",
     "未据本文给出天气实验主张；不声称逐节复核"),
    ("2609.08412v1", "SPW", "weather ensembles", "ABSTRACT_ONLY",
     "API abstract; HTML retrieved but not method-audited",
     "随机权重扰动构造确定天气模型的集合",
     "若后续转概率预报，可参考扰动机制与集合评估",
     "集合/CRPS 收益不能等同当前确定性编辑的 native MSE 收益"),
    ("2605.22222v3", "ARC-STAR", "PDE post-hoc correction", "ABSTRACT_ONLY",
     "API abstract; HTML retrieval failed",
     "冻结 host 的全局订正、局部精修及预算风险路由",
     "后续输出订正挑战者",
     "正文未取回，不作为具体算法细节或性能结论的依据"),
    ("2609.21787v1", "Compact but Moving", "CV/recurrent world models", "ABSTRACT_ONLY",
     "API abstract; PDF downloaded, extraction unavailable",
     "低秩干预相关几何随 rollout 运输变化",
     "提示检查各时效响应基覆盖，不把 LoRA rank 当输出响应 rank",
     "原文是 latent-state 干预；正文未解析，不声称已验证其定理"),
    ("2609.17042v1", "Adapter Banks", "motor control/imitation", "ABSTRACT_ONLY",
     "API abstract; HTML retrieved but not method-audited",
     "共享 recurrent 模型上的 adapter options 和组合控制",
     "记录 adapter bank 近邻，避免泛化首次主张",
     "不是本轮必须复现的天气基线；不简单归类为纯 RL 回报路由"),
    ("2609.11445v1", "FARM", "robotic world-model readouts", "ABSTRACT_ONLY",
     "API abstract; HTML retrieved but not method-audited",
     "冻结 predictive features 的便宜 failure readout",
     "低容量合法状态基线，完整计入特征提取成本",
     "只测 readout 的低延迟不能代表 EarthDelta 完整 serving 路径"),
]


def main():
    papers = {}
    searches = []
    for name in ("arxiv_searches.json", "additional_neighbors.json", "cv_nlp_searches.json"):
        document = json.loads((HERE / name).read_text())
        groups = document if isinstance(document, list) else [document]
        for group in groups:
            results = group.get("results", [])
            searches.append({"source_file": name, "query": group.get("query_name", group.get("name", "id_lookup")),
                             "url": group.get("url"), "returned_entries": len(results)})
            for paper in results:
                papers[paper["id"].rsplit("/", 1)[-1]] = paper

    records = []
    for index, card in enumerate(CARDS, 1):
        pid, short, domain, scope, sections, idea, transfer, limits = card
        metadata = papers[pid]
        files = []
        for suffix in (".html", ".readable.txt", ".pdf"):
            path = HERE / (pid + suffix)
            if path.exists():
                files.append({"path": "evidence/" + path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        records.append({
            "reference_id": f"L{index:02}", "arxiv_id": pid, "short_name": short,
            "title": metadata["title"], "authors": metadata.get("authors", []),
            "published": metadata["published"], "updated": metadata["updated"],
            "primary_url": "https://arxiv.org/abs/" + pid,
            "domain": domain, "read_scope": scope, "sections_read": sections,
            "overlap_or_useful_idea": idea, "proposed_transfer_to_earthdelta": transfer,
            "interpretation_limits": limits, "reported_results_independently_reproduced": False,
            "primary_text_files": files,
        })
    output = {
        "review_date": "2026-09-22", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "search_cutoff": "2026-09-22", "search_platform": "arXiv API and primary arXiv HTML/PDF",
        "scope_note": "Focused research review, not an exhaustive systematic review. Downloaded text is not automatically reviewed. Dates are API metadata, not inferred from arXiv identifiers. Peer-review acceptance is not independently verified.",
        "research_direction": "Reusable finite intervention-response prediction for goal/action transfer, with strong vector-gain and output-correction baselines",
        "gpu_executions_for_this_research_review": 0,
        "external_llm_api_calls_for_this_research_review": 0,
        "searches": searches, "papers": records,
    }
    (HERE.parent / "literature_review.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"papers_indexed": len(records), "scopes": {s: sum(r["read_scope"] == s for r in records) for s in sorted({r["read_scope"] for r in records})}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
