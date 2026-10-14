"""Report challenge strata, paired outcomes and hunter contacts from real CSV."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from vision_mpc.stratified_cases import GROUPS
from vision_mpc.evaluator import wilson

LABELS = ['Ordinary', 'Hunter gap\n0.08-0.12', 'Hunter gap\n0.12-0.20', 'Hunter gap\n0.20-0.35',
          'Hunter blocks\nroute', 'Goal near\nedge', 'Evader near\nobstacle', 'Goal near\nobstacle',
          'Corner\nstart', 'Long\nroute']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('results', type=Path)
    args = p.parse_args(); out = args.results
    rows = list(csv.DictReader((out/'results.csv').open()))
    cases = {c['id']: c for c in json.loads((out/'cases.json').read_text())}
    if len(rows) != len(cases)*2 or any(m not in {r['method'] for r in rows} for m in ['astar', 'mpc']):
        raise RuntimeError('Expected complete paired A*/MPC results')
    groups = list(GROUPS)
    results = {}
    for group in groups:
        results[group] = {}
        for method in ['astar', 'mpc']:
            selected = [r for r in rows if r['group'] == group and r['method'] == method]
            n = len(selected); success = sum(r['reason'] == 'success' for r in selected)
            results[group][method] = {'n': n, 'success': success, 'rate': success/n,
                                     'ci': wilson(success, n),
                                     'outcomes': {reason: sum(r['reason'] == reason for r in selected)
                                                  for reason in sorted({r['reason'] for r in selected})},
                                     'hunter_contact_episodes': sum(int(r['hunter_obstacle_contact_frames']) > 0 for r in selected)}
    (out/'stratified_summary.json').write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
    fig, ax = plt.subplots(figsize=(15, 6)); x = np.arange(len(groups))
    for method, offset, color in [('astar', -.18, '#6699bb'), ('mpc', .18, '#38a67d')]:
        rate = np.array([results[g][method]['rate'] for g in groups])
        ci = np.array([results[g][method]['ci'] for g in groups])
        ax.bar(x+offset, rate*100, .34, label=method, color=color)
        ax.errorbar(x+offset, rate*100, yerr=np.maximum(0, np.stack((rate-ci[:, 0], ci[:, 1]-rate))*100),
                    fmt='none', color='#333333', capsize=3)
    ax.set_xticks(x, LABELS); ax.set_ylim(0, 105); ax.set_ylabel('Success rate (%)')
    ax.set_title('Frozen controller: paired challenge strata, 95% Wilson intervals')
    ax.legend(); ax.grid(axis='y', alpha=.2); fig.tight_layout()
    fig.savefig(out/'stratified_success.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(15, 5)); base = np.zeros(len(groups))
    for reason, color in [('success', '#38a67d'), ('caught', '#d75959'), ('obstacle_collision', '#e2aa51'),
                          ('timeout', '#8d9ca8'), ('perception_failure', '#836ab1'), ('ambiguous_terminal', '#444444')]:
        values = np.array([results[g]['mpc']['outcomes'].get(reason, 0) for g in groups])
        if values.any():
            ax.bar(x, values, bottom=base, label=reason, color=color); base += values
    ax.set_xticks(x, LABELS); ax.set_ylabel('Episodes'); ax.set_title('MPC outcomes by declared scenario group')
    ax.legend(); fig.tight_layout(); fig.savefig(out/'mpc_outcomes_by_group.png', dpi=160); plt.close(fig)
    mpc_rows = [r for r in rows if r['method'] == 'mpc']
    fig, ax = plt.subplots(figsize=(9, 5))
    for reason, color in [('success', '#38a67d'), ('caught', '#d75959'), ('obstacle_collision', '#e2aa51'),
                          ('timeout', '#8d9ca8'), ('perception_failure', '#836ab1')]:
        selected = [r for r in mpc_rows if r['reason'] == reason]
        if selected:
            ax.scatter([cases[r['case']]['initial_hunter_gap'] for r in selected],
                       [int(r['steps']) for r in selected], label=reason, c=color, alpha=.8)
    ax.set_xlabel('Initial hunter-evader center distance'); ax.set_ylabel('Episode steps')
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(out/'hunter_gap_outcomes.png', dpi=160); plt.close(fig)
    index = {(r['case'], r['method']): r for r in rows}
    improve = [c for c in cases if index[c, 'mpc']['reason'] == 'success' and index[c, 'astar']['reason'] != 'success']
    regress = [c for c in cases if index[c, 'astar']['reason'] == 'success' and index[c, 'mpc']['reason'] != 'success']
    failures = [r for r in mpc_rows if r['reason'] != 'success']
    lines = ['# 分层可行性测试报告（2026-10-13）', '',
             '## 设计与冻结范围', '',
             '在此前 100 个普通随机场景之外新增 10 类场景，每类 12 个，共 120 个不同初始位置。'
             '每个位置分别运行 A* 跟踪与完整 MPC，共 240 回合。使用旧正式测试的冻结 config.json；'
             '控制器与动力学代码未调参。新增的评测代码只记录分组与猎手碰障碍信息。', '',
             '案例在运行前一次生成并保存，固定种子 20261013。初始红绿间距大于 0.075，'
             '初始绿球距障碍大于 0.080，猎手距障碍大于 0.085，目标距障碍大于 0.078，'
             '目标距两球均大于 0.15；坐标不超出 [-0.80,0.80]。拒绝的是几何不符合生成条件的初始案例，'
             '并未删去运行失败案例。各分组是生成条件标签，其他风险因素可能同时存在。', '',
             '这是一组等额困难场景测试，整体成功率受分组比例影响，不代表真实考核位置的总体分布。'
             '每类仅 12 个样本，置信区间较宽，不能从 12/12 推断该类必然成功。', '',
             '## 分组结果', '',
             '| 场景类别 | A* 成功 | MPC 成功 | MPC 成功率及95%区间 | 被捕 | 撞障碍 | 超时 | 感知失败 |',
             '|---|---:|---:|---|---:|---:|---:|---:|']
    for group in groups:
        a, m = results[group]['astar'], results[group]['mpc']; o = m['outcomes']; lo, hi = m['ci']
        lines.append(f'| {GROUPS[group]} | {a["success"]}/{a["n"]} | {m["success"]}/{m["n"]} | '
                     f'{m["rate"]:.1%} [{lo:.1%}, {hi:.1%}] | {o.get("caught",0)} | '
                     f'{o.get("obstacle_collision",0)} | {o.get("timeout",0)} | {o.get("perception_failure",0)} |')
    lines += ['', '![分层成功率](stratified_success.png)', '', '![MPC结束原因](mpc_outcomes_by_group.png)', '',
              '![初始猎手间距与结果](hunter_gap_outcomes.png)', '', '## 配对比较', '',
              f'同一个初始位置中，MPC 成功而 A* 失败：{len(improve)} 个；A* 成功而 MPC 失败：{len(regress)} 个。'
              '这是直接配对计数，不是独立场景平均值之差，也未据此声称统计显著。', '',
              '## 猎手碰障碍与规则', '',
              '按原 simple_tag.py，猎手碰障碍不触发结束；绿球碰障碍、被捕以及到达目标才是相关结束事件。'
              '物理引擎仍会对猎手与障碍的接触施加碰撞力，没有删除障碍或赋予穿越能力。'
              '碰障碍诊断在决策之后由独立评估器读取真值，不返回控制器。', '',
              '接触的统计口径是每次 step 后猎手与任一障碍中心距离小于 0.075。'
              '视觉上看起来接触不一定达到这个物理阈值。这里只检查离散帧，不能排除帧间短暂接触。', '',
              '| 方法 | 有猎手接触的回合 | 其中成功 | 无接触回合 | 其中成功 |',
              '|---|---:|---:|---:|---:|']
    for method in ['astar', 'mpc']:
        selected = [r for r in rows if r['method'] == method]
        touch = [r for r in selected if int(r['hunter_obstacle_contact_frames']) > 0]
        no_touch = [r for r in selected if int(r['hunter_obstacle_contact_frames']) == 0]
        lines.append(f'| {method} | {len(touch)} | {sum(r["reason"]=="success" for r in touch)} | '
                     f'{len(no_touch)} | {sum(r["reason"]=="success" for r in no_touch)} |')
    lines += ['', '上述接触与成功的关系是观察性统计，场景难度不同，不能证明因果关系。'
              '猎手当前控制逻辑来自老师原环境，没有修改规则。', '', '## 全部 MPC 失败案例', '',
              '| 案例 | 分组 | 原因 | 步数 | 初始猎手间距 |', '|---|---|---|---:|---:|']
    for r in failures:
        c = cases[r['case']]
        lines.append(f'| {r["case"]} | {GROUPS[r["group"]]} | {r["reason"]} | {r["steps"]} | {c["initial_hunter_gap"]:.3f} |')
    if not failures:
        lines.append('| 无 | — | — | — | — |')
    lines += ['', '每个案例均有 trajectory.json、preview.png、final.png。失败可能来自距离过近、'
              '有限预测窗口、有限采样、视觉遮挡/误差或参考路线局限；不能仅凭失败轨迹断言问题数学上无解。', '',
              '## 适用性结论', '',
              '应依据分组表分别判断普通场景、近距离追击、靠边、靠障碍等情况。此前普通随机 100/100 '
              '只反映原范围，不替代本次分层测试，更不能保证考核满分。'
              '本次未根据结果修改控制参数，后续若优化需另建新的验证与测试清单。', '',
              '## 复现', '',
              '在项目主目录、已激活 multiagent-ml 的 Linux 终端运行：', '', '```bash',
              'python wsl-support/vision_mpc/run.py --mode compare --methods astar mpc \\',
              '  --cases wsl-support/vision_mpc/outputs/stratified_20261013/cases.json \\',
              '  --config wsl-support/vision_mpc/outputs/stratified_20261013/config.json \\',
              '  --output wsl-support/vision_mpc/outputs/stratified_reproduction',
              'python wsl-support/vision_mpc/stratified_report.py wsl-support/vision_mpc/outputs/stratified_reproduction', '```']
    (out/'分层测试报告.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    selections = [{'case': r['case'], 'reason': r['reason'], 'method': 'mpc'} for r in failures[:6]]
    if improve:
        selections.append({'case': improve[0], 'reason': 'paired_improvement', 'method': 'mpc'})
        selections.append({'case': improve[0], 'reason': 'paired_baseline', 'method': 'astar'})
    (out/'representative_replays.json').write_text(json.dumps(selections, indent=2), encoding='utf-8')
    print(out/'分层测试报告.md')


if __name__ == '__main__':
    main()
