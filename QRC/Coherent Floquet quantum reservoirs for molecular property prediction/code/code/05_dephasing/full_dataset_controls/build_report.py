"""Build an evidence report from saved supplementary measurements, not a manuscript."""
from common import *

TARGETS = ["gap_1fs", "gap_5fs", "gap_10fs", "min_gap_next_5fs"]
CN = {"gap_1fs": "1-fs 能隙", "gap_5fs": "5-fs 能隙", "gap_10fs": "10-fs 能隙", "min_gap_next_5fs": "未来 5-fs 最小能隙"}


def table(headers, rows):
    return "\n".join(["| " + " | ".join(map(str, headers)) + " |",
                      "| " + " | ".join(["---"] * len(headers)) + " |"] +
                     ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def interval(mean, low, high):
    return f"{mean:+.4f} [{low:+.4f}, {high:+.4f}]"


def figure(name, alt):
    path = HERE / "figures" / name
    return f"![{alt}]({path}.png)\n\n[PDF]({path}.pdf) · [SVG]({path}.svg)"


def link(path, label):
    return f"[{label}]({HERE / path})"


def main():
    probe = pd.read_csv(HERE / "probes/summary.csv")
    graph = pd.read_csv(HERE / "graph_direct/comparison_summary.csv")
    hopping = pd.read_csv(HERE / "hopping/method_summary.csv")
    effects = pd.read_csv(HERE / "hopping/paired_effects.csv")
    selected = pd.read_csv(HERE / "hopping/qrc_validation_selected_p.csv")
    final = hopping[(hopping.readout == "final") & (hopping.decoder == "rbf_svr")]
    final_effects = effects[(effects.readout == "final") & (effects.decoder == "rbf_svr")]
    dephase_files = sorted((HERE / "dephasing").glob("*/lambda*/seed*/convergence.csv"))
    dephase_complete = len(dephase_files) == 24
    status = "本轮预先冻结的补充实验已完成；未修改论文正文或原图。" if dephase_complete else f"预测补充已完成；全 Q 退相干代表样本检查已完成 {len(dephase_files)}/24 个单元，仍在运行。"
    parts = ["# 审稿意见补充实验报告（中文）\n\n日期：2026-09-05。" + status,
        "## 1. 结论先行\n\n"
        "新结果支持一个更具体的应用结论：在当前输入和读出协议下，DTC-QRC 对部分能隙预测目标具有优势；"
        "其中近期历史的贡献随目标而变。它们没有复现旧探针所暗示的普遍长程记忆，也不支持将匹配宽度 ESN 的比较扩大为对所有经典模型的优势。\n\n"
        "- 无完整输入交叉重复的 distant-pair 在间隔 1、2、4、6 仍有仿射可读信号；delayed-motif 未得到稳定的高于随机水平结果。\n"
        "- Hopping 中，5-fs 能隙和未来 5-fs 最小能隙相对 p=1 的配对区间支持近期历史收益；1-fs、10-fs 的区间跨零。\n"
        "- 相比新调参 ESN，final/RBF-SVR QRC 的 1-fs 和最小能隙优势区间不跨零；5-fs 和 10-fs 区间跨零。\n"
        "- BACE/BBBP 相同 T 下优于原匹配宽度 ESN 的结果仍成立，但直接 token SVR 在若干条件下更好。\n"
        "- 退相干的 Monte Carlo 估计误差不可忽略；本轮检查特征精度，不把它误报为全数据预测指标已经收敛。",
        "## 2. 原文、代码和补充协议的对齐\n\n"
        "只扩展了用户批准的标签无关背景模板。原标签关系、前景位置、符号编码、Q/T、通道、测量及仿射探针读出保留。所有新增输出独立存放；未覆盖旧结果。\n\n" + table(
        ["项目", "固定协议", "本轮工作"], [
            ["BACE/BBBP", "Q=12，局部/general tokens；T=24、30；p=0.10；final Z+ZZ=78 维", "同 T 的均值汇聚/完整 token 序列 RBF-SVR"],
            ["受控跨事件", "Q=12，T=20，p=0.10；lag=1/2/4/6/8/12/16；原仿射读出", "14 个输入条件×6 seeds；128 组/512 行，原 356/76/80 划分"],
            ["Hopping", "Q=15，T=20；30 个真实 X/Z 数、45 槽；final=120、TM3=360 维", "p=1 配对、直接窗口解码、27 配置 ESN、输入匹配 ExtraTrees"],
            ["退相干", "BACE Q12/T30，hopping Q15/T20；lambda=0.5、1；p=0、0.10", "各 24 个代表输入；6 seeds；嵌套 8/32/128 Pauli 轨迹"]]) +
        "\n\n全部新增量子模拟固定 g=0.84、W=0.50、L=3，使用原 seed 生成的相互作用参数及默认缩放 1。六个 seeds 为 0、10、20、30、40、50。"
        "图任务在末事件后也 reset；hopping 只在帧间 reset。TM3 是末端之后两个无输入、无 reset 的 Floquet 周期，未重复注入。"
        "Hopping 沿用训练集标准化与截断，量子注入尺度 0.5 仅施加一次。\n\n"
        + link("PROTOCOL.md", "训练前冻结协议") + "；原文—代码审计见 "
        + f"[审计报告]({ROOT / 'BFS-QRC/PRL/revised_open_dtc_qrc/pre_supplement_protocol_audit_20260905/AUDIT_CN.md'})。",
        "## 3. 无重复完整输入的跨事件探针\n\n"
        "每个反事实组共享一个背景：对 0、4、5、6 各出现五次的 20 符号序列作独立排列，原非零前景覆盖对应位置。"
        "符号 0 仍按原编码注入，并非单位操作。没有加入微小 ID 噪声。每个条件均有 512 个不同完整输入、128 个不同背景族；"
        "84 份划分的 train/validation/test 完整输入及背景族交集均为零，组内单位置符号边缘分布逐项平衡。"
        "这里的未见输入指完整编码序列及背景族，不表示原来的前景位置/关系模板也被分割为互不相交集合。"
        "背景多样性与任务难度同时改变，因此新旧差异不是单独改变划分的一次干预。\n\n"
        + figure("01_unique_background_probes", "无重复背景模板的跨事件探针") +
        "\n\n图为六个配对 seeds 的均值±样本标准差，不是置信区间。下表给 QRC 的逐点 95% bootstrap 区间；"
        "另附七个 lag 同时区间，避免由单个延迟点得出长程结论。\n\n" + table(
        ["探针", "lag", "QRC AUC ± SD", "逐点 95% CI", "七点同时下界"], [
            ["Delayed motif" if r.family == "delayed_motif" else "Distant pair", int(r.lag),
             f"{r.auc_mean:.4f} ± {r.auc_sd:.4f}", f"[{r.ci_low:.4f}, {r.ci_high:.4f}]", f"{r.simultaneous_low:.4f}"]
            for r in probe[probe.method == "DTC-QRC"].itertuples()]) +
        "\n\nDistant-pair 的 lag=16 逐点区间虽高于 0.5，但七点同时区间跨越 0.5，不能据此延续‘16 事件稳定记忆’的旧结论。"
        "原 distant-pair 规则是两个符号的 XOR，未平衡完整直方图分布；新的直接均值输入 SVR 在这一任务 AUC=1，"
        "因此该探针只能显示跨事件组合的可读性，不能排除无序计数捷径，也不能独立证明顺序特异性记忆。"
        "直接末事件 SVR 在两探针均为 0.5；完整输入 SVR 均为 1；delayed-motif 的直接均值 SVR 为 0.5。"
        "这些诊断控制使用非线性解码器，不与仿射 QRC 混作等解码器优势比较。\n\n"
        + link("probes/input_audit.csv", "逐条件输入审计") + " · " + link("probes/metrics_all_seeds.csv", "504 行原始指标")
        + " · " + link("probes/paired_comparisons.csv", "配对差值与同时区间"),
        "## 4. Hopping：单事件映射与历史贡献\n\n"
        "保持原 178/59/60 条训练/验证/测试物理轨迹，对应 1424/472/480 个窗口。六 seeds 使用同一轨迹划分，"
        "不是六个独立数据集。沿用现稿规则：每个目标、readout、decoder 在原 27 点网格上，以六 seed 平均验证 MAE 选一个固定 p，"
        "然后使用已有 held-out 预测；没有按测试 MAE 重新选 p。\n\n"
        "p=1 删除跨编码事件的储备池历史，但最后一个编码事件仍含距离及后向差分速度。故它不是‘只使用一个原始位置帧’。\n\n"
        + figure("02_hopping_history_contribution", "验证选择的 p 相比 p=1 的历史收益") +
        "\n\n以下为 final/RBF-SVR；MAE 与差值均为 eV。负差值有利于验证选定 QRC。\n\n" + table(
        ["目标", "验证 p", "选定 QRC", "p=1", "QRC − p=1，95% CI"], [
            [CN[target], f"{float(final_effects[(final_effects.target == target) & (final_effects.baseline == 'QRC-p1')].iloc[0].selected_p):.2f}",
             f"{final[(final.target == target) & (final.method == 'Selected-QRC')].iloc[0].mae:.4f}",
             f"{final[(final.target == target) & (final.method == 'QRC-p1')].iloc[0].mae:.4f}",
             interval(*final_effects[(final_effects.target == target) & (final_effects.baseline == 'QRC-p1')].iloc[0][['delta_mae', 'ci_low', 'ci_high']])]
            for target in TARGETS]) +
        "\n\n1-fs 的主要预测优势在单事件映射中已经存在；5-fs 与最小能隙目标更明确地受益于保留近期历史。"
        "这一区分比把全部收益归为长程记忆更符合现有证据。区间为 20,000 次配对、交叉 seed/物理轨迹 bootstrap，"
        "不将 480 个重叠时间窗口当作独立重复。它们是逐点区间，未对所有目标/方法作多重比较校正。",
        "## 5. Hopping：加强后的经典对照\n\n"
        "ESN 保持 120 个节点、每事件三次更新及原 final/TM3，扫描谱半径 {0.3,0.6,0.9}、leak {0.1,0.65,1.0}、"
        "输入尺度 {0.25,0.75,1.5}，共 27 组，包含原配置。每个目标/readout/decoder 用六 seed 平均验证 MAE 选择配置；"
        "各 seed 的 Ridge/SVR 超参数仅用验证数据。原 ESN 仍保留，未替换历史结果。\n\n"
        "直接 Ridge/SVR 使用 1、2、4、8、20 个末端编码事件及原解码器网格。ExtraTrees 使用相同 T=20 的全部 600 个真实坐标，"
        "400 棵树，验证选择 max_features={0.5,1.0}、min_samples_leaf={1,3}，六模型 seeds，无额外能量或完整分子描述符输入。\n\n"
        + figure("03_hopping_strengthened_baselines", "相同 T=20 的强化经典对照") +
        "\n\n表中仅显示均值，图中为逐点 95% 区间；显著性应看配对差值而非两组区间是否重叠。\n\n" + table(
        ["目标", "QRC", "原 ESN", "调参 ESN", "直接 SVR/T20", "ExtraTrees/T20"], [
            [CN[target]] + [f"{final[(final.target == target) & (final.method == m)].iloc[0].mae:.4f}"
             for m in ("Selected-QRC", "Original-ESN", "Tuned-ESN", "Direct-20", "ExtraTrees")]
            for target in TARGETS]) + "\n\n" + table(
        ["目标", "QRC − 调参 ESN，95% CI", "QRC − ExtraTrees，95% CI"], [
            [CN[target]] + [interval(*final_effects[(final_effects.target == target) & (final_effects.baseline == m)].iloc[0][['delta_mae', 'ci_low', 'ci_high']])
             for m in ("Tuned-ESN", "ExtraTrees")]
            for target in TARGETS]) +
        "\n\n调参后 ESN 的误差明显下降。QRC 对 1-fs 和最小能隙仍有清楚的配对改善；5-fs 对调参 ESN 的区间跨零，"
        "尽管六 seed 均值差都同号。10-fs 上 ExtraTrees 点估计更好，而配对轨迹区间跨零。"
        "因此不能把四个目标表述为统一显著超越充分调参的经典方法。\n\n"
        + figure("04_direct_input_history_windows", "不同直接输入窗口的 Ridge 与 SVR") +
        "\n\n窗口图是主动改变可用历史的消融，不用于跨 T 宣称优势。直接 SVR 是确定性训练，每目标/窗口只有一次拟合；"
        "只在配对计算中与六 QRC seeds 对齐，没有伪造六个独立直接模型。全部 MAE、RMSE、R² 等字段保存在各目录 metrics.csv。\n\n"
        "下表保留所有 readout/decoder 组合，避免只呈现 SVR 的有利结果。Ridge 与调参 ESN 的八个配对区间均跨零；"
        "部分小正则 Ridge 候选出现病态矩阵警告，当前严格保留原 float32 实现，未静默更换精度或筛掉不利单元。\n\n" + table(
        ["读出", "解码器", "目标", "验证 p", "QRC MAE", "调参 ESN MAE", "配对差值 95% CI"], [
            [r.readout, "Ridge" if r.decoder == "linear" else "RBF-SVR", CN[r.target], f"{r.selected_p:.2f}",
             f"{hopping[(hopping.readout == r.readout) & (hopping.decoder == r.decoder) & (hopping.target == r.target) & (hopping.method == 'Selected-QRC')].iloc[0].mae:.4f}",
             f"{hopping[(hopping.readout == r.readout) & (hopping.decoder == r.decoder) & (hopping.target == r.target) & (hopping.method == 'Tuned-ESN')].iloc[0].mae:.4f}",
             interval(r.delta_mae, r.ci_low, r.ci_high)]
            for r in effects[effects.baseline == "Tuned-ESN"].itertuples()]) +
        "\n\n" + link("hopping/paired_effects.csv", "全部 160 个配对对照") + " · " + link("hopping/method_summary.csv", "全部方法均值/区间")
        + " · " + link("hopping/esn_validation_selected.csv", "ESN 验证选点") + " · " + link("hopping/implementation_parity.csv", "原实现数值一致性检查"),
        "## 6. BACE/BBBP：相同 T 的直接 token 对照\n\n"
        "输入仍是 Q12 对应的每事件 36 个 local/general 坐标。均值汇聚暴露 36 维；完整序列暴露 36T 维；QRC/匹配 ESN 暴露 78 维。"
        "这些是相同输入信息的对照，不是相同读出宽度的对照。所有方法使用原六个 scaffold 划分和相同 RBF-SVR 网格，"
        "新直接对照按验证 AUC → log loss → accuracy 选点。\n\n"
        + figure("05_graph_input_matched_controls", "BACE 与 BBBP 相同 T 的输入对照") +
        "\n\n" + table(["任务/T", "QRC", "匹配 ESN", "均值 + SVR", "完整序列 + SVR"], [
            [f"{d}/{t}"] + [f"{graph[(graph.dataset == d) & (graph.t == t) & (graph.method == m)].iloc[0].auc:.4f}"
              for m in ("open_dtc_qrc", "matched_esn", "Direct-mean", "Direct-full")]
            for d, t in (("BACE", 24), ("BACE", 30), ("BBBP", 24), ("BBBP", 30))]) +
        "\n\nBACE/T30 完整序列 SVR 高于 QRC，QRC−该基线的逐点配对区间为 "
        + interval(*graph[(graph.dataset == "BACE") & (graph.t == 30) & (graph.method == "Direct-full")].iloc[0][["qrc_minus_baseline", "paired_ci_low", "paired_ci_high"]]) +
        "。BBBP/T30 的均值汇聚达到 0.8105，QRC 为 0.7629；这一均值差的配对区间跨零。"
        "汇聚对照说明，一部分局部化学属性组合无需通过储备池跨事件状态也能被非线性解码器利用。"
        "同时，四个同 T 的 QRC−原匹配 ESN 区间仍全部为正；这两个事实应同时保留。\n\n"
        + link("graph_direct/comparison_summary.csv", "均值、SD 及配对区间") + " · " + link("graph_direct/comparison_metrics.csv", "逐 seed 指标"),
        "## 7. 退相干 Monte Carlo 精度\n\n"
        "BACE 代表样本从原 seed=0 的 train/validation/test 各选八个确定性位置；hopping 每个分区选八条不同物理轨迹的中间窗口。"
        "选择不使用标签或预测误差。每任务六 reservoir seeds 共用同一组 24 个代表输入；保留 Q12/Q15，没有用小 Q 结果替代原任务。\n\n"
        "每比特每事件以 lambda/2 概率施加 Z；全局 reset 仍以精确后缀权重混合。使用批次无关、后缀间独立、不同 lambda 共享均匀随机数、"
        "K=8/32/128 共享前缀的新随机流；旧 8 条缓存不复用。每个特征的标准误由独立 Pauli 轨迹估计，随后在代表样本与可观测量上取 RMS。"
        "128 条的均值只是更高精度估计，不是精确密度矩阵真值；它与相干特征的差也仍有残余估计误差。"]
    if dephase_complete:
        data = pd.concat([pd.read_csv(p) for p in dephase_files])
        aggregate = data.groupby(["task", "dephasing", "p", "trajectories"], as_index=False).agg(
            feature_se_rms=("feature_se_rms", "mean"), ratio=("se_to_physical_rms", "mean"),
            coherent_difference_rms=("coherent_difference_rms", "mean"))
        write_csv(HERE / "dephasing/convergence_summary.csv", aggregate)
        rows = []
        for (task, lam, p), block in aggregate.groupby(["task", "dephasing", "p"]):
            by_k = block.set_index("trajectories")
            rows.append(["BACE/Q12" if task == "bace" else "Hopping/Q15", f"{lam:.1f}", f"{p:.1f}"] +
                        [f"{by_k.loc[k, 'feature_se_rms']:.5f}" for k in (8, 32, 128)] +
                        [f"{100*by_k.loc[128, 'ratio']:.1f}%"])
        parts.append(figure("06_dephasing_feature_convergence", "Q12 与 Q15 的退相干特征标准误收敛") + "\n\n" +
            table(["任务", "lambda", "p", "SE/8", "SE/32", "SE/128", "SE128 / 相干差异"], rows) +
            "\n\n轨迹数每增四倍，标准误约减半，与独立 Monte Carlo 估计相符。reset 后绝对估计噪声也较小；"
            "所以旧图中 reset 后的性能损失较小，可能同时包含物理通道变化和估计噪声变化，不能仅由该图把二者分离。"
            "128 条结果仍呈现远大于抽样标准误的相干—退相干特征位移，支持该通道确实改变测量表示；"
            "这与下游性能下降的归因是两个层面。逐 seed CSV 另存两个互不重叠的 64 条系综之间的 RMS 差，"
            "以及 8/32 条均值到嵌套 128 条均值的差，不把 K128 的自差为零误作精确收敛证据。"
            "本轮没有重训全数据 K=32/128 的解码器，也没有声称预测 AUC/MAE 已经收敛。\n\n" +
            link("dephasing/convergence_all_seeds.csv", "144 行逐 seed/K/p 特征精度") + " · " + link("dephasing/convergence_summary.csv", "汇总表"))
    else:
        parts.append(f"本节尚在计算：{len(dephase_files)}/24 个任务/lambda/seed 单元已完成。汇总图只在 24 个单元齐全后输出。")
    parts.append("另外，原 BACE 退相干缓存的解码器选点规则已统一复核：使用 AUC→log loss→accuracy，"
        "重跑 lambda=0/0.5/1、p=0/0.1、六 seeds 共 36 个单元。所有选点及测试 AUC 与旧结果完全相同，"
        "因此该次级规则差异没有改变本批 BACE 退相干结论。\n\n" + link("dephasing/legacy8_decoder_audit.csv", "36 单元解码器复核"))
    strata = pd.read_csv(HERE / "hopping/gap_strata_metrics.csv")
    small = strata[strata.gap_bin_ev == "0-1"]
    parts.extend(["## 8. 能隙分布与区域误差：应用相关的探索性检查\n\n"
        "在模型和选点全部冻结之后，按真实目标能隙 [0,1)、[1,3)、[3,∞) eV 描述测试误差。"
        "这些阈值只是便于阅读的分箱，不是化学精度标准或已验证的预警阈值，也不作为预测时的输入。"
        "分箱内每条有样本的物理轨迹等权；同时保存逐窗口加权误差，避免两种平均混淆。\n\n"
        + figure("07_hopping_gap_strata", "真实能隙分区的描述性误差") +
        "\n\n低于 1 eV 的目标区间如下，MAE 单位为 eV。这里只报告描述性均值，没有对这些后验分组作显著性声明。\n\n" + table(
        ["目标", "窗口/轨迹数", "QRC", "调参 ESN", "直接 SVR", "ExtraTrees"], [
            [CN[target], f"{int(small[(small.target == target) & (small.method == 'Selected-QRC')].iloc[0].n_windows)}/{int(small[(small.target == target) & (small.method == 'Selected-QRC')].iloc[0].n_trajectories)}"] +
            [f"{small[(small.target == target) & (small.method == m)].iloc[0].trajectory_balanced_mae:.4f}" for m in ("Selected-QRC", "Tuned-ESN", "Direct-20", "ExtraTrees")]
            for target in TARGETS]) +
        "\n\nQRC 在这一分区的点估计低于列出的经典对照，但绝对误差仍较大，尤其 10-fs。"
        "这支持继续研究目标特定的分子预测表示，不等于已经达到实际 nonadiabatic 轨迹传播或小能隙预警的精度要求。\n\n"
        + link("hopping/gap_strata_distribution.csv", "训练/验证/测试的能隙分布") + " · " + link("hopping/gap_strata_metrics.csv", "全部区域误差"),
        "## 9. 统计边界、完整性与后续\n\n"
        "本轮没有做测试集调参。所有新的直接模型和 ESN 配置用原验证集选择；测试后增加的能隙分箱仅作描述。"
        "但验证集与模型网格均固定于当前数据集，因此区间反映当前协议下的重抽样不确定性，不等于对任意化学体系的泛化保证。\n\n"
        "图任务和探针的 bootstrap 单位是六个配对 seeds，这些划分使用同一基础数据集，不能把它们称为六次独立采样实验。"
        "Hopping 的区间交叉重抽 reservoir seed 和物理轨迹，在各方法间保持配对；未重抽训练过程或重新选择超参数。"
        "直接确定性模型没有 seed 方差。所有应用区间为逐点 95%；探针另给每条七延迟曲线/对照的同时带，未进一步跨探针家族校正。\n\n"
        "复核覆盖输入与划分、物理元数据、逐预测指标重算、验证选点、MC 系综均值/标准误重算。"
        "预审计的 235 份论文/代码/来源文件 SHA256 保持不变。"
        "原 ESN 特征的 CPU 新实现与已存实现相差小于 10⁻⁶；直接末事件 Ridge/SVR 的原解码器与新计算选点一致。"
        "各生成图已检查全边框、文字位置，并输出 PDF/SVG 与 PNG；全部定量图从保存数据和 Python 代码生成。\n\n"
        + link("verification.json", "机器核验明细") + " · " + link("provenance.json", "依赖版本及 SHA256") + " · " + link("README_CN.md", "运行与复现说明") +
        "\n\n下一轮若要强化物理归因，优先在预先指定的条件上补全数据退相干预测指标的 K 收敛，而不是增加更多物理网格。"
        "若要主张顺序特异性长程记忆，则需要另行设计能排除完整计数捷径的探针，并在训练前单独确定协议；"
        "本轮没有擅自改变原 foreground/标签来达成这一目标。论文修改、硬件补采、W–g 或非默认相互作用扫描均不属于本轮输出。"])
    (HERE / "REPORT_CN.md").write_text("\n\n".join(parts) + "\n")
    print("Written", HERE / "REPORT_CN.md", flush=True)


if __name__ == "__main__":
    main()
