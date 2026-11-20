"""Traceable development report; no selection uses historical test."""
import csv
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from .common import ROOT,DATA,OUT,CONFIGS,SEEDS,read,write,score_metrics
NAME='Baseline1_HOGMLP_v3_Stability_Capacity_L2_EpochSelection'

def csv_rows(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def pct(x):return 'N/A' if x is None else f'{100*x:.2f}%'

def run():
    summary=read(OUT/'summary.json');frozen=read(OUT/'freeze.json');stability=read(OUT/'stability/summary.json')
    lines=[f'# {NAME}','',
      '独立开发研究；旧版及v2保持冻结，公共默认仍为旧版。固定数据：训练185图（133 PASS、52 FAIL），验证101图（24 PASS、77 FAIL）。历史test已接触，仅作冻结后的回归。',
      '', '## 版本与本轮规则', '',
      '- Baseline1_HOGMLP_v1_PASSOnly_Autoencoder：旧A2 seed42，原冻结阈值。',
      '- Baseline1_HOGMLP_v2_Supervised_CLAHE：原B2 seed42，原冻结阈值。',
      '- 本轮R0保留v2结构，R1缩小网络，R2加强L2；各3个种子，均训练200轮。验证集选择轮次与阈值：误拒≤5/24，先漏放少，再误拒少，轮次并列取早。组间比较三种子均值。',
      '- 所有变体仅用于诊断，不参与学习；未新增几何训练增强。',
      '', '## 稳定性诊断','',
      '每图原始设置加20种变体。相同尺寸与填充内比较位置；相同尺寸与位置内比较填充。原图对缩小居中的变化同时包含尺度、重采样和补白影响，不代表纯位置影响，也不能模拟斜拍近大远小。每图重复变体在逐图记录中标记，样本数始终按原图计算。',
      '', '| 分区/模型 | 至少一种变体判定改变/原图数 | 原图漏放 | 原图误拒 |', '|---|---:|---:|---:|']
    for key,s in stability.items():
        b=s['variants']['original'];lines.append(f"| {key} | {s['any_flip']}/{s['n']} ({s['any_flip']/s['n']:.2%}) | {b['fn']} | {b['fp']} |")
    lines+=['','### 每种变换（验证集）','','| 模型 | 变体 | 漏放 | 误拒 | F1 | 平衡准确率 | 相对原图翻转 | 正确变错误 | 错误变正确 |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in ('old','v2'):
        for variant,m in stability['valid_'+name]['variants'].items():lines.append(f"| {name} | {variant} | {m['fn']} | {m['fp']} | {pct(m['f1'])} | {pct(m['balanced_accuracy'])} | {m['flips']}/101 | {m['correct_to_wrong']} | {m['wrong_to_correct']} |")
    lines+=['','### 控制尺度后的配对比较（验证集）','','| 固定条件对照 | 旧版翻转/101 | v2翻转/101 |','|---|---:|---:|']
    for key,val in stability['valid_old']['paired_flips'].items():lines.append(f"| {key} | {val} | {stability['valid_v2']['paired_flips'][key]} |")
    lines+=['','## 9次训练与轮次选择','','| 配置 | 种子 | 最佳轮次 | 最佳漏放 | 最佳误拒 | 200轮漏放 | 200轮误拒 | 训练及逐轮验证秒 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    runs={}
    for group in CONFIGS:
        runs[group]=[]
        for seed in SEEDS:
            m=read(OUT/f'experiments/{group}/seed{seed}/metrics.json');runs[group].append(m)
            b,l=m['best'],m['last'];lines.append(f"| {group} | {seed} | {b['epoch']} | {b['fn']} | {b['fp']} | {l['fn']} | {l['fp']} | {m['fit_seconds']:.1f} |")
    lines+=['','| 配置 | 隐藏层 / L2 | 平均漏放±种子标准差 | 平均误拒±种子标准差 | 平均F1±标准差 | 平均平衡准确率±标准差 |','|---|---|---:|---:|---:|---:|']
    for group,rr in runs.items():
        def avg(key,percentage=False):
            a=np.array([r['best'][key] for r in rr])*(100 if percentage else 1)
            return f'{a.mean():.2f} ± {a.std(ddof=0):.2f}'+('%' if percentage else '')
        lines.append(f'| {group} | {CONFIGS[group]} | {avg("fn")} | {avg("fp")} | {avg("f1",True)} | {avg("balanced_accuracy",True)} |')
    lines+=['','标准差为三个种子的总体标准差（ddof=0），不是泛化置信区间。',
        f"\n入选组 **{frozen['group']} seed42，第{frozen['epoch']}轮**，阈值 `{frozen['threshold']}`。R0三个种子的第200轮权重、标准化器与v2 B2逐项精确复现。",'','## 共同验证与历史回归','',
        '| 分区/版本 | TP | FP误拒 | FN漏放 | TN | FAIL F1 | 平衡准确率 |','|---|---:|---:|---:|---:|---:|---:|']
    old=read(ROOT/'outputs/member04_hog/supervised_v2/assessment/comparison.json')
    for split in ('valid','test'):
        metrics=[('v1冻结',old['old_'+split]),('v2冻结',old['new_'+split])]
        metrics.append(('v3研究',summary['groups'][frozen['group']][0] if split=='valid' else read(OUT/'assessment/test_metrics.json')))
        for name,m in metrics:lines.append(f"| {split}/{name} | {m['tp']} | {m['fp']} | {m['fn']} | {m['tn']} | {pct(m['f1'])} | {pct(m['balanced_accuracy'])} |")
    best=summary['groups'][frozen['group']][0]
    lines+=['','冻结v1/v2沿用原阈值；本轮R0/R1/R2重新按相同误拒预算选择阈值和轮次。不可把所有提升归因于缩小网络或正则化，必须同时看R0和第200轮对照。',
      '', '## 错误分组和归因边界','', '| 验证分组 | v1漏放/误拒 | v2漏放/误拒 | v3漏放/误拒 | 张数 |','|---|---:|---:|---:|---:|']
    vp=csv_rows(OUT/f"experiments/{frozen['group']}/seed42/best_valid_predictions.csv")
    op=csv_rows(ROOT/'outputs/member04_hog/experiments/A2/seed42/valid_predictions.csv')
    npred=csv_rows(ROOT/'outputs/member04_hog/supervised_v2/experiments/B2/seed42/valid_predictions.csv')
    meta={Path(r['image_path']).name:r for r in read(DATA/'provenance.json')['sources'] if r['split']=='valid'}
    keys={id:('FAIL/干扰'+str(m['counts']['0']) if m['label']=='FAIL' else 'PASS/'+m['source_session']) for id,m in meta.items()}
    grouped={}
    for key in sorted(set(keys.values())):
        values=[score_metrics([r for r in rows if keys[r['image_id']]==key]) for rows in (op,npred,vp)]
        grouped[key]=values;lines.append('| '+key+' | '+' | '.join(f"{m['fn']}/{m['fp']}" for m in values)+f" | {sum(keys[i]==key for i in keys)} |")
    errors={}
    for name,pred in [('v1',op),('v2',npred)]:
        olderr={r['image_id'] for r in pred if r['label']!=r['predicted_label']};newerr={r['image_id'] for r in vp if r['label']!=r['predicted_label']}
        errors[name]=dict(common=sorted(olderr&newerr),fixed=sorted(olderr-newerr),introduced=sorted(newerr-olderr))
        lines.append(f"\n相对{name}：共有错误{len(olderr&newerr)}张，修复{len(olderr-newerr)}张，新增{len(newerr-olderr)}张。")
    write(OUT/'grouped_errors.json',dict(groups=grouped,changes=errors))
    lines+=['','## 解释与限制','',
      '- 如果固定尺度位置变化仍使预测翻转，可确认当前模型对这类画布位置变化敏感；不能由此断言真实物体移动或视角变化的误差完全相同。',
      '- 白色与边缘灰度补白的差异只反映合成边界条件，不证明真实背景是错误的唯一原因。',
      '- 来源、异常类型与标签混杂，物理场景独立性尚未核实。没有据此新增训练图片或修改原标注。',
      '- 已接触验证集用于逐轮、阈值及配置选择，存在选择乐观性；历史test不作为独立验收，更不据此回头挑选模型。',
      '- 本轮不自动替换公共默认入口；没有改善也保留负结果。',
      '', '## 复现与证据','',
      '从仓库根目录按顺序运行：','```powershell',
      'python -m members.member04_hog.supervised_v3_study.stability',
      'python -m members.member04_hog.supervised_v3_study.train',
      'python -m members.member04_hog.supervised_v3_study.assess',
      'python -m members.member04_hog.supervised_v3_study.report','```',
      '使用Python3.11及成员4锁定依赖。已有输出目录禁止覆盖，复现须在独立工作区恢复固定数据与v2参考权重。',
      '逐图诊断、逐轮验证分数、训练历史、最佳与末轮预测、冻结和检查记录位于 outputs/member04_hog/supervised_v3_study/；权重位于 weights/member04_hog/supervised_v3_study/。最新上传状态见模块progress.md。']
    # Standalone SVG curves: validation error counts, with selected epoch marked.
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="840" viewBox="0 0 1200 840"><rect width="1200" height="840" fill="white"/><text x="15" y="24" font-family="sans-serif" font-size="18">Validation FN (red), FP (blue); selected epoch dashed. Fixed FP budget = 5.</text>']
    for gi,group in enumerate(CONFIGS):
        for si,seed in enumerate(SEEDS):
            x=gi*400+40;y=si*260+65;w=330;h=180
            history=read(OUT/f'experiments/{group}/seed{seed}/history.json')
            epoch=runs[group][si]['best']['epoch']
            svg.append(f'<text x="{x}" y="{y-12}" font-family="sans-serif">{group} seed{seed}, chosen epoch {epoch}</text>')
            svg.append(f'<path d="M{x},{y} V{y+h} H{x+w}" fill="none" stroke="black"/>')
            for tick in (0,20,40,60,80):
                yy=y+h-tick*h/80
                svg.append(f'<text x="{x-27}" y="{yy+4}" font-size="11">{tick}</text><path d="M{x},{yy} H{x+w}" stroke="#ddd"/>')
            for key,color in [('fn','#d33'),('fp','#06c')]:
                points=' '.join(f"{x+(r['epoch']-1)*w/199:.2f},{y+h-r[key]*h/80:.2f}" for r in history)
                svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.5"/>')
            xx=x+(epoch-1)*w/199
            svg.append(f'<path d="M{xx},{y} V{y+h}" stroke="#555" stroke-dasharray="4 4"/><text x="{x}" y="{y+h+18}" font-size="12">1</text><text x="{x+w-20}" y="{y+h+18}" font-size="12">200</text>')
    svg.append('</svg>');(OUT/'validation_curves.svg').write_bytes(''.join(svg).encode('utf-8'))
    report='\n'.join(lines)+'\n';(OUT/'REPORT.md').write_bytes(report.encode('utf-8'))
    (ROOT/f'members/member04_hog/docs/{NAME}.md').write_bytes(report.encode('utf-8'))
    print('Report complete',flush=True)

if __name__=='__main__':run()
