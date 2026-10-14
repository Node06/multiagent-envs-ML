"""Compare V1/V2 honestly: development/regression versus fresh held-out cases."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE/'outputs'


def read(name):
    p = OUT/name
    return p, json.loads((p/'summary.json').read_text()), list(csv.DictReader((p/'results.csv').open()))


def main():
    dev, ds, dr = read('optimization_dev')
    reg, rs, rr = read('v2_regression_stratified')
    ordinary, os, ore = read('v2_regression_ordinary')
    fresh, fs, fr = read('v2_independent')
    old = json.loads((OUT/'stratified_20261013'/'summary.json').read_text())['mpc']
    signature = lambda c: tuple(c['hunter']+c['evader']+c['goal'])
    historical = set()
    for directory in [dev, reg, ordinary]:
        historical.update(signature(c) for c in json.loads((directory/'cases.json').read_text()))
    fresh_cases = json.loads((fresh/'cases.json').read_text())
    assert not historical.intersection(signature(c) for c in fresh_cases), 'Independent coordinates reused'
    assert len(fresh_cases)==60 and len(fr)==120 and len(rr)==120 and len(ore)==100
    previous = json.loads((OUT/'final_test'/'manifest.json').read_text())
    legacy_hash = hashlib.sha256((HERE/'controller_v1.py').read_bytes()).hexdigest()
    assert legacy_hash == previous['source_sha256']['controller.py'], 'Baseline changed'
    versions = ['mpc_v1', 'mpc']; indexed = {(r['case'], r['method']): r for r in fr}
    ids = {r['case'] for r in fr}
    better = [c for c in ids if indexed[c, 'mpc']['reason']=='success' and indexed[c, 'mpc_v1']['reason']!='success']
    worse = [c for c in ids if indexed[c, 'mpc_v1']['reason']=='success' and indexed[c, 'mpc']['reason']!='success']
    fig, ax = plt.subplots(figsize=(9, 5))
    rates = np.array([fs[v]['success_rate'] for v in versions])
    ci = np.array([fs[v]['success_rate_95ci'] for v in versions])
    ax.bar(['Frozen V1', 'Recovery V2'], rates*100, color=['#6699bb', '#38a67d'])
    ax.errorbar(np.arange(2), rates*100,
                yerr=np.maximum(0, np.stack((rates-ci[:, 0], ci[:, 1]-rates))*100), fmt='none', color='black', capsize=5)
    ax.set_ylim(0, 105); ax.set_ylabel('Success rate (%)')
    ax.set_title('New independent scenarios: paired V1/V2, 95% Wilson intervals')
    ax.grid(axis='y', alpha=.2); fig.tight_layout(); fig.savefig(fresh/'v1_v2_success.png', dpi=160); plt.close(fig)
    lines = ['# MPC V2 优化与验证报告', '', '## 改动与数据边界', '',
             '保留旧控制器 controller_v1.py，哈希与原测试控制器一致。mpc_v1 调用旧版；mpc 默认调用 V2。'
             '原物理、追击、终止和图像识别规则未修改，不按案例 ID 或真实坐标设置特例。', '',
             '旧版候选以朝全局路径的速度跟踪为中心，小扰动不一定能反向逃跑。'
             'V2 保留常规搜索，只有最优序列仍预测危险时才启动恢复搜索：'
             '相对当前视觉位置生成多方向、不同距离的临时避险点；允许先远离目标，'
             '若干步后重新接入 A* 路线；预测窗口扩展到 50 步。'
             '优先选预测安全候选；若全部危险，则优先延长预计存活时间。', '',
             '两个原失败案例参与了策略设计，原 120 场分层案例和 100 场普通案例均作为回归集，'
             '不再把它们作为 V2 的独立泛化证据。另用新种子 2026101302、每组 6 场共 60 场做独立配对测试；'
             '报告所有失败，不根据独立结果继续调参。', '',
             '## 已知失败修复', '', '| 案例 | V1 | V2 | V2 步数 | 恢复决策次数 |', '|---|---|---|---:|---:|']
    for r in dr:
        if r['method']=='mpc':
            baseline = next(x for x in dr if x['case']==r['case'] and x['method']=='mpc_v1')
            lines.append(f'| {r["case"]} | {baseline["reason"]} | {r["reason"]} | {r["steps"]} | {r["recovery_steps"]} |')
    lines += ['', '## 回归结果（不是独立测试）', '', '| 数据集 | V1 成功 | V2 成功 | V2 结束原因 |', '|---|---:|---:|---|',
              f'| 原分层 120 场 | {old["successes"]}/120 | {rs["mpc"]["successes"]}/120 | {rs["mpc"]["outcomes"]} |',
              f'| 原普通 100 场 | 100/100 | {os["mpc"]["successes"]}/100 | {os["mpc"]["outcomes"]} |', '',
              '## 新独立配对测试', '', '| 版本 | 成功 | 成功率 | 95% Wilson 区间 | 结束原因 |', '|---|---:|---:|---|---|']
    for version in versions:
        s = fs[version]; low, high = s['success_rate_95ci']
        lines.append(f'| {version} | {s["successes"]}/{s["episodes"]} | {s["success_rate"]:.1%} | {low:.1%}–{high:.1%} | {s["outcomes"]} |')
    lines += ['', f'配对中 V2 成功而 V1 失败 {len(better)} 场，V1 成功而 V2 失败 {len(worse)} 场。'
              '样本量有限，不根据置信区间重叠或两例修复就声称统计上必然优于旧版。', '',
              '![独立测试](v1_v2_success.png)', '', '## 计算开销', '',
              '| 版本 | 各回合中位耗时的中位数/ms | 有恢复决策的回合 |', '|---|---:|---:|']
    for version in versions:
        selected = [r for r in fr if r['method']==version]
        lines.append(f'| {version} | {np.median([float(r["decision_p50_ms"]) for r in selected]):.2f} | '
                     f'{sum(int(r["recovery_steps"])>0 for r in selected)} |')
    lines += ['', '恢复搜索增加计算量；常规回合保留旧算法。耗时包含图像渲染和首次规划，'
              '不保证 0.1 秒的硬实时截止。恢复搜索也不保证所有候选中一定存在安全轨迹。', '',
              '## 独立测试中的全部 V2 失败', '', '| 案例 | 分组 | 原因 | 步数 |', '|---|---|---|---:|']
    failures = [r for r in fr if r['method']=='mpc' and r['reason']!='success']
    for r in failures:
        lines.append(f'| {r["case"]} | {r["group"]} | {r["reason"]} | {r["steps"]} |')
    if not failures:
        lines.append('| 无 | — | — | — |')
    lines += ['', '## 复现', '', '在项目主目录、Linux multiagent-ml 环境中：', '', '```bash',
              'python wsl-support/vision_mpc/run.py --mode compare --methods mpc_v1 mpc \\',
              '  --cases wsl-support/vision_mpc/outputs/v2_independent/cases.json \\',
              '  --config wsl-support/vision_mpc/outputs/v2_independent/config.json \\',
              '  --output wsl-support/vision_mpc/outputs/v2_reproduction', '```', '',
              '重放旧版带 mpc 标签的历史实验时增加 --controller-version v1。旧 CSV、轨迹和报告保留。']
    (fresh/'优化验证报告.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    sources = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in HERE.glob('*.py')}
    (fresh/'version_manifest.json').write_text(json.dumps({'sources': sources, 'baseline_matches_original': True,
                                                        'default_method': 'mpc -> controller_v2',
                                                        'legacy_method': 'mpc_v1 -> controller_v1'}, indent=2), encoding='utf-8')
    print(fresh/'优化验证报告.md')


if __name__ == '__main__':
    main()
