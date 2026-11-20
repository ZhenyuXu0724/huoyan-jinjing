"""Generate traceable old/new comparison tables from recorded predictions."""
import csv
import statistics
import numpy as np
from .common import ROOT,MODULE,OUT,GROUPS,SEEDS,read,write,score_metrics,check_old
from .assess import csv_rows,verify_freeze
from ..evaluate import write_predictions

FIELDS=('f1','balanced_accuracy','recall','specificity','false_pass','false_fail','coverage')
LABELS={'f1':'FAIL F1','balanced_accuracy':'平衡准确率','recall':'FAIL 召回率','specificity':'正常图真负率',
        'false_pass':'漏放数','false_fail':'误拒数','coverage':'覆盖率'}


def run():
    frozen=verify_freeze();selected=frozen['selected_group']
    aggregates={};flat=[]
    for group in GROUPS:
        versions={}
        for version,base in [('旧',ROOT/'outputs/member04_hog/experiments'),('新',OUT/'experiments')]:
            rows=[]
            for seed in SEEDS:
                path=base/group/f'seed{seed}'
                saved=read(path/'metrics.json')
                m=score_metrics(csv_rows(path/'valid_predictions.csv'))
                for key in ('tp','fp','fn','tn'):
                    if m[key]!=saved[key]:raise ValueError('Recorded metrics differ from predictions')
                rows.append({**m,'seconds':saved['seconds'],'model_bytes':saved['model_bytes']})
            versions[version]={key:dict(mean=statistics.mean(r[key] for r in rows),std=statistics.stdev(r[key] for r in rows))
                               for key in (*FIELDS,'seconds','model_bytes')}
        aggregates[group]=versions
        for field in (*FIELDS,'seconds','model_bytes'):
            old,new=versions['旧'][field],versions['新'][field]
            flat.append(dict(group=group,metric=field,old_mean=old['mean'],old_std=old['std'],
                             new_mean=new['mean'],new_std=new['std'],delta=new['mean']-old['mean']))
    write_predictions(OUT/'comparison_table.csv',flat)
    write(OUT/'comparison_aggregates.json',aggregates)
    c=read(OUT/'assessment/comparison.json');timing=read(OUT/'assessment/timing_summary.json')
    lines=['# 新旧 baseline1 完整增强实验与对比报告','',
      '数据版本：`12c85c23bc3a57dea51f39915278f31d9133ce74`。新版 11 组 × 3 种子 × 200 轮全部完成。旧版权重、默认入口、推理代码和历史实验保持原样；报告由逐图预测重新统计。','',
      '## 数据、方法与结论边界','',
      '| 项目 | 旧版 | 新版 |','|---|---|---|',
      '| 方法 | HOG＋MLP 自编码器 | HOG＋监督式 MLP 二分类 |',
      '| 训练 | 133 PASS | 133 PASS＋52 FAIL |',
      '| 验证（共同） | 24 PASS＋77 FAIL | 相同101图 |',
      '| 历史测试（共同） | 23 PASS＋1 FAIL | 相同24图，已公开且被接触 |',
      '| 隐藏层 | 64–16–64 | 64–16 |',
      '| 训练目标 | 重建训练正常特征 | 加权二分类，PASS=0、FAIL=1 |',
      '| 阈值首要规则 | 验证 FAIL F1 | 验证平衡准确率 |',
      '| 得分 | 重建均方误差 | 未校准 FAIL 分类输出 |',
      f'| 代表模型 | A2 seed42（历史冻结） | {selected} seed42 |','',
      '两版使用相同 HOG 设置，但训练数据、学习目标、网络与选型规则共同改变；这是完整版本比较，不将全部差异归因于新增 FAIL。增强只作用于训练；CLAHE 组的确定性预处理同样作用于推理。三种子标准差只描述初始化波动，不是场景总体置信区间。','',
      '旧数据不改分区；新增52图每图3个干扰件，其中9图还缺一个M4x25。验证干扰图为6张含1件、51张含6件、20张含7件。验证PASS全来自旧包、FAIL全来自11.14新包，来源与标签混杂；真实场景独立性和近重复未核实。只排除了跨分区精确重复。','',
      '## 十一组逐项对照','',
      '表内均为三种子均值 ± 样本标准差；Δ为新版减旧版。比例指标取值0–1；漏放、误拒以图片数计。各表使用同一101张验证图。','']
    lines += ['### 总览：三种子均值','',
              '| 组 | 旧F1 | 新F1 | ΔF1 | 旧平衡准确率 | 新平衡准确率 | Δ平衡准确率 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for group,versions in aggregates.items():
        a,b=versions['旧'],versions['新']
        lines.append(f"| {group} | {a['f1']['mean']:.4f} | {b['f1']['mean']:.4f} | {b['f1']['mean']-a['f1']['mean']:+.4f} | {a['balanced_accuracy']['mean']:.4f} | {b['balanced_accuracy']['mean']:.4f} | {b['balanced_accuracy']['mean']-a['balanced_accuracy']['mean']:+.4f} |")
    lines.append('')
    for field in FIELDS:
        lines += [f'### {LABELS[field]}','', '| 组 | 旧版 | 新版 | Δ新−旧 |','|---|---:|---:|---:|']
        for group,versions in aggregates.items():
            a,b=versions['旧'][field],versions['新'][field]
            lines.append(f"| {group} | {a['mean']:.4f} ± {a['std']:.4f} | {b['mean']:.4f} ± {b['std']:.4f} | {b['mean']-a['mean']:+.4f} |")
        lines.append('')
    lines += ['## 代表模型：混淆矩阵与指标','','FAIL为正类：TP检出异常，FP误拒正常，FN漏放异常，TN放行正常。',
      '', '| 模型/分区 | TP | FP | FN | TN | Precision | Recall | F1 | 真负率 | 平衡准确率 | 覆盖率 |',
      '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for key in ('old_valid','new_valid','old_test','new_test'):
        m=c[key];lines.append(f"| {key} | {m['tp']} | {m['fp']} | {m['fn']} | {m['tn']} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['specificity']:.4f} | {m['balanced_accuracy']:.4f} | {m['coverage']:.2%} |")
    lines+=['','## 分组错误','','同一正常图片可进入来源、光照、视角多个分组，不能将这些行相加。异常行按干扰件数互斥分组。','',
            '| 分组 | 图片数 | 旧漏放 | 新漏放 | 旧误拒 | 新误拒 |','|---|---:|---:|---:|---:|---:|']
    for group,a in c['groups']['old_valid'].items():
        b=c['groups']['new_valid'][group]
        lines.append(f"| {group} | {a['total']} | {a['fn']} | {b['fn']} | {a['fp']} | {b['fp']} |")
    lines+=['','## 单项与组合消融','','以下只在新版内部比较，避免把算法变化当成增强收益。','',
            '| 对照（前者−后者） | Δ平衡准确率 | ΔF1 | Δ漏放 | Δ误拒 |','|---|---:|---:|---:|---:|']
    contrasts=[(g,'A0') for g in ('A1','A2','A3','A4','A5','B1','B2','C0')]+[('C0','C_without_geometry'),('C0','C_without_clahe')]
    for a,b in contrasts:
        delta=[aggregates[a]['新'][f]['mean']-aggregates[b]['新'][f]['mean'] for f in ('balanced_accuracy','f1','false_pass','false_fail')]
        lines.append('| '+a+' − '+b+' | '+' | '.join(f'{d:+.4f}' for d in delta)+' |')
    lines+=['','C_without_geometry 与 B2、C_without_clahe 与 A2 的设置完全相同。它们作为一致性复跑，不计作额外独立证据。','',
            '## 成本与无信息对照','','| 组 | 旧平均运行秒 | 新独立运行等价秒 | 旧平均模型MiB | 新平均模型MiB |','|---|---:|---:|---:|---:|']
    for g,v in aggregates.items():
        lines.append(f"| {g} | {v['旧']['seconds']['mean']:.2f} | {v['新']['seconds']['mean']:.2f} | {v['旧']['model_bytes']['mean']/2**20:.2f} | {v['新']['model_bytes']['mean']/2**20:.2f} |")
    lines+=['','旧运行秒为历史记录，非本轮同机重训。新版复用一次图片解码及每组特征准备，表中把共享成本计入每次运行，表示独立运行的等价成本，不应相加当实际墙钟时间。原始拟合/验证秒、共享解码秒、特征准备秒分别记录；不能据此宣称训练速度严格提升。','',
            '| 模型 | 同机Median ms | 同机P95 ms | 测量次数 |','|---|---:|---:|---:|']
    for name,m in timing['results'].items():
        lines.append(f"| {name} | {m['median_ms']:.2f} | {m['p95_ms']:.2f} | {m['observations']} |")
    lines+=['','计时：同机、单线程、模型预加载，预热后共同20张验证图（10 PASS＋10 FAIL）重复3遍，交替模型先后顺序。包含读图、预处理、HOG、推理；不包含相机、UI或模型加载，硬件信息与逐次数据见 timing_summary.json/timings.csv。','',
            '| 恒定策略（验证） | 准确率 | FAIL F1 | 平衡准确率 | 漏放 | 误拒 |','|---|---:|---:|---:|---:|---:|']
    for label,m in c['constant_valid'].items():
        lines.append(f"| 永远{label} | {m['accuracy']:.4f} | {m['f1']:.4f} | {m['balanced_accuracy']:.4f} | {m['fn']} | {m['fp']} |")
    a,b=c['old_valid'],c['new_valid'];test=c['new_test']
    old_mean=aggregates['A2']['旧'];new_mean=aggregates[selected]['新']
    if b['fn']<=a['fn'] and b['fp']<=a['fp'] and (b['fn']<a['fn'] or b['fp']<a['fp']):
        tradeoff='在代表种子的验证结果中，漏放与误拒没有一项增加，至少一项减少。'
    elif b['fn']>=a['fn'] and b['fp']>=a['fp']:
        tradeoff='在代表种子的验证结果中，未取得漏放与误拒的同时改善。'
    else:
        tradeoff='代表种子的验证结果体现漏放与误拒的取舍，不能表述为两类错误同时改善。'
    lines+=['','## 结果解读','',
      '本轮十一组逐组比较中，新版各组的平均FAIL F1和平衡准确率均低于旧版同名组。这不支持“增加FAIL训练后整体更好”的主张；保留旧默认模型。新版入选B2的平均平衡准确率高于旧默认A2，是不同组及不同选型规则的版本比较，不应掩盖同组对照的下降。','',
      f"新版最终选中 {selected}，固定seed42；验证平衡准确率从 {a['balanced_accuracy']:.4f} 变为 {b['balanced_accuracy']:.4f}，F1从 {a['f1']:.4f} 变为 {b['f1']:.4f}。漏放 {a['fn']}→{b['fn']}，误拒 {a['fp']}→{b['fp']}。这些是共同验证集上的开发证据，不是独立泛化估计。",'',
      tradeoff,'',
      f"三种子层面，旧默认组A2的平均平衡准确率为 {old_mean['balanced_accuracy']['mean']:.4f} ± {old_mean['balanced_accuracy']['std']:.4f}，新版所选组为 {new_mean['balanced_accuracy']['mean']:.4f} ± {new_mean['balanced_accuracy']['std']:.4f}。平均漏放 {old_mean['false_pass']['mean']:.2f}→{new_mean['false_pass']['mean']:.2f}，平均误拒 {old_mean['false_fail']['mean']:.2f}→{new_mean['false_fail']['mean']:.2f}。入选依据是组均值，不能用代表种子替代组总体结果。",'',
      f"相对恒定策略的平衡准确率0.5，新版差值为 {b['balanced_accuracy']-.5:+.4f}；相对永远FAIL的F1，新版差值为 {b['f1']-c['constant_valid']['FAIL']['f1']:+.4f}。不能以未经对照的高准确率或高F1判断有效。",'',
      f"历史测试新版漏放 {test['fn']}、误拒 {test['fp']}，F1为 {test['f1']:.4f}。该分区仅1个FAIL且先前已被接触，不能据此宣称未知干扰识别能力，也没有依据测试调整阈值或改选方案。",'']
    for a_group,b_group in contrasts:
        d=aggregates[a_group]['新']['balanced_accuracy']['mean']-aggregates[b_group]['新']['balanced_accuracy']['mean']
        lines.append(f'- {a_group} 相对 {b_group} 的平均平衡准确率变化 {d:+.4f}；'+('本轮开发结果提高。' if d>0 else '本轮没有提高。'))
    lines+=['','### 逐图错误变化','']
    for split in ('valid','test'):
        for kind,label in [('common','两版共有'),('fixed','新版修复'),('introduced','新版新增')]:
            ids=c[split+'_error_changes'][kind]
            lines.append(f"- {split} {label}（{len(ids)}张）："+('、'.join(ids) if ids else '无')+'。')
    lines+=['','### 干扰数量、阈值与典型图分析','']
    for number in (1,6,7):
        a_group=c['groups']['old_valid'][f'FAIL/distractors/{number}']
        b_group=c['groups']['new_valid'][f'FAIL/distractors/{number}']
        lines.append(f"- 每图{number}个干扰件（{a_group['total']}图）：旧版漏放{a_group['fn']}，新版漏放{b_group['fn']}；FAIL召回率{a_group['recall']:.2%}→{b_group['recall']:.2%}。")
    lines+=['','新版对单干扰组有所改善，却在六件、七件组新增漏放，说明本次监督学习没有自动获得对干扰数量增加的稳健性。所有分组均来自既有验证集，不能据此证明具体未见型号的效果。','',
            '| 新版验证标签 | 最小分数 | 中位分数 | P95分数 | 最大分数 |','|---|---:|---:|---:|---:|']
    selected_path=OUT/'experiments'/selected/'seed42'
    selected_rows=csv_rows(selected_path/'valid_predictions.csv')
    for label in ('PASS','FAIL'):
        q=np.quantile([float(r['prediction_score']) for r in selected_rows if r['label']==label],[0,.5,.95,1])
        lines.append('| '+label+' | '+' | '.join(f'{value:.6g}' for value in q)+' |')
    threshold=read(selected_path/'metrics.json')['threshold']
    loss=read(selected_path/'history.json')[-1]['train_loss']
    lines += ['',f'新版冻结阈值为 {threshold:.12g}，末轮训练损失为 {loss:.6g}。分类输出未校准，不能把数值当作真实异常风险；低训练损失不等于可靠泛化。阈值由验证集优化，本表不是新的独立证据；没有改用0.5或测试最优阈值。','',
      '直接核查四张代表图：IMG_20261114_160244.jpg（单环状干扰，新版修复旧漏放）、IMG_20261114_161323.jpg（六个已标注干扰，含明显大螺母及螺杆，新版新增漏放）、IMG_20261114_160430.jpg（单环状干扰，两版均漏放）、IMG_7244.png（正常图，新版新增误拒）。相似的环状干扰也会得到不同判定，不能只按干扰件数量解释；整图空间布局、尺度和背景可能共同影响HOG。此为可见外观及结果对照，不能单凭图片确认实际规格或确定失败原因。原图及检查缩略图仅留本地，不进入公开代码或交付ZIP。','']
    lines+=['','来源、背景、摆放、光照与异常标签存在关联，分组错误只描述关联而不确定因果。干扰件没有细分型号标注，不能用3/6的数量关系推导50%的识别率，也不能将同图检测到某个已见干扰解释为识别了所有未见干扰。典型外观需结合原图与标注核查；不擅自修订真值。','',
      '## 复现与产物','',
      '从仓库根目录依次执行 supervised_v2.prepare、supervised_v2.train、supervised_v2.assess、supervised_v2.report。完整命令与安全重跑约定见 supervised_v2/README.md。','',
      '新权重：weights/member04_hog/supervised_v2/；数据清单：data/member04_hog/supervised_v2/；原始实验、配置、训练历史、冻结、逐图预测、CSV表及计时：outputs/member04_hog/supervised_v2/。路径均以仓库根为基准。旧 baseline1 默认入口未替换。']
    report='\n'.join(lines)+'\n'
    (OUT/'REPORT.md').write_text(report,encoding='utf-8')
    (MODULE/'docs/COMPARISON_V2.md').write_text(report,encoding='utf-8')
    check_old()
    print('Generated comparison report and machine-readable tables',flush=True)


if __name__=='__main__':run()
