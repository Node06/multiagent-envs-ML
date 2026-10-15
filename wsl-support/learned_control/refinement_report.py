"""Summarize both completed optimization rounds without changing the models."""
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


def main():
    out = HERE/'outputs/refined_20261014'
    summary = json.loads((out/'evaluation/summary.json').read_text())
    frozen = json.loads((out/'evaluation/frozen_models.json').read_text())
    train = json.loads((out/'training_summary.json').read_text())
    selected = frozen['selected_candidate']
    rows = list(csv.DictReader((out/'evaluation/results.csv').open()))
    assert len(rows) == 955
    methods = ['previous_main', 'optimized_v2', 'refined_main', 'refined_imitation', 'mpc']
    paired = {}
    for group in ('fresh_course_holdout', 'random_obstacles_stress', 'published_course_regression'):
        old = {r['case']: r for r in rows if r['method'] == 'previous_main' and r['group'] == group}
        new = {r['case']: r for r in rows if r['method'] == selected and r['group'] == group}
        repair = [k for k in old if old[k]['reason'] != 'success' and new[k]['reason'] == 'success']
        regress = [k for k in old if old[k]['reason'] == 'success' and new[k]['reason'] != 'success']
        n = len(repair)+len(regress)
        paired[group] = dict(repaired_cases=repair, regressed_cases=regress,
                             paired_mcnemar_exact_p=binomtest(len(repair), n, .5).pvalue if n else 1.)
    (out/'paired_comparison.json').write_text(json.dumps(paired, indent=2))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for ax, group in zip(axes, ('fresh_course_holdout', 'published_course_regression')):
        rates = [summary[group][m]['success_rate'] for m in methods]
        low = [max(0, rates[i]-summary[group][m]['success_rate_95ci'][0]) for i, m in enumerate(methods)]
        high = [max(0, summary[group][m]['success_rate_95ci'][1]-rates[i]) for i, m in enumerate(methods)]
        ax.bar(['Previous', 'Safety56 V2', 'Risk V3', 'V3 imitation', 'MPC'], rates, yerr=[low, high], capsize=3)
        ax.set_ylim(0, 1.15); ax.set_ylabel('Success fraction (Wilson 95% CI)'); ax.tick_params(axis='x', rotation=20)
        ax.set_title('Fresh holdout: seed2026101450' if group.startswith('fresh') else 'Published 60-case regression')
        for i, m in enumerate(methods):
            r = summary[group][m]; ax.text(i, rates[i]+.02, f"{r['successes']}/{r['episodes']}", ha='center')
    fig.savefig(out/'final_comparison.png', dpi=160); plt.close(fig)
    text = ['# 两轮 PPO 控制优化：最终实验报告', '',
            '## 做了哪些改动', '',
            '第一轮：原21维视觉特征扩充为56维，增加障碍净距离、接近速度、最近接距离与时间、目标方向几何、待执行动作风险和边界距离。训练近危险状态时加权，新增两轮困难 DAgger；每轮从验证最佳历史模型开始，保留全部阶段。PPO示范约束合入同次更新，减小学习率、clip和KL阈值。', '',
            '第一轮三个训练种子2411、2423、2437，各8192步PPO；仅依据新40场验证选择 seed2411 / ppo4096。其独立100场成功83/100，旧模型同批81/100；旧60场回归46/60，旧模型42/60；随机障碍18/30，旧模型20/30。提升有限且存在分布外退化。完整结果位于相邻 optimized_20261014/优化实验报告.md。', '',
            '第二轮：从第一轮冻结模型开始，新增60个困难固定布局与20个随机障碍布局的全专家示范，全部80场专家成功。训练可微四步风险损失，比较系数0.025/0.1/0.4；Actor保持低学习率，Critic使用3e-4与每轮5个预拟合epoch。风险预测忽略接触力且假设短期固定动作，仅作为训练近似；运行时没有风险过滤器或 MPC 兜底。', '',
            '教师环境、猎人、碰撞阈值和终止规则未变。公开控制接口仍只接受 RGB；CNN复用已训练模型，所有几何由图像估计和自身动作历史计算，真实状态仅用于事件奖励及事后诊断。不是可微端到端图像控制，也不能保证考核得分。', '',
            '## 第二轮训练与模型冻结', '',
            f"训练随机种子 {train['seed']}，示范 {train['samples']} 条，PPO {train['ppo_steps']} 步，耗时 {train['seconds']/60:.1f} 分钟。第二轮是单种子继续训练，不能冒称最终版本做了三个独立训练种子。", '',
            '| 阶段 | 80场验证成功 | 失败 |', '|---|---:|---|']
    for s in json.loads((out/'validation.json').read_text()):
        fails = ', '.join(f'{k}:{v}' for k, v in s['outcomes'].items() if k != 'success')
        text.append(f"| {s['stage']} | {s['outcomes'].get('success', 0)}/80 | {fails} |")
    text += ['', f"测试前依据同一80场验证选定候选 {selected}；模型文件哈希已冻结，测试不反馈阶段、种子或参数选择。新测试固定seed2026101450，共100场；随机障碍seed2026101451，共30场；原60场只回归，另加默认场景；五方法共955场。", '', '## 全新配对测试与回归', '']
    for group in ('fresh_course_holdout', 'random_obstacles_stress', 'published_course_regression'):
        text += ['### '+group, '', '| 方法 | 成功 | Wilson 95%区间 | 失败 |', '|---|---:|---:|---|']
        for method in methods:
            r = summary[group][method]; ci = r['success_rate_95ci']; label = method+'（验证预选候选）' if method == selected else method
            fails = ', '.join(f'{k}:{v}' for k, v in r['outcomes'].items() if k != 'success') or '无'
            text.append(f"| {label} | {r['successes']}/{r['episodes']} ({r['success_rate']:.1%}) | {ci[0]:.1%}–{ci[1]:.1%} | {fails} |")
        p = paired[group]
        text += ['', f"预选候选修复旧失败 {len(p['repaired_cases'])} 场，同时退化 {len(p['regressed_cases'])} 场；双侧精确McNemar p={p['paired_mcnemar_exact_p']:.4f}。", '']
    text += ['## 默认场景与性能边界', '']
    for method in methods:
        r = next(r for r in rows if r['method'] == method and r['group'] == 'default_demo')
        text.append(f"- {method}：{r['reason']}，{r['steps']}步；录像与轨迹均保留。")
    main = summary['fresh_course_holdout'][selected]
    text += ['', f"验证预选候选的新测试成功率 {main['success_rate']:.1%}；90%是目标而非保证，是否达到以实际成绩为准。不要把不同测试场景的百分比直接相减。新增数据、安全特征、损失和优化器共同变化，阶段结果不能证明某单项改动的因果收益；PPO收益应与同批模仿模型结果分开报告。", '',
             '模型仍会失败，默认录像不能隐瞒失败。老师固定障碍课程分布优先，随机障碍成绩单列。完整实验在WSL2 Ubuntu22.04 CPU完成，原生Windows图形运行未完整验证；模型与路径支持跨平台。未来模型比较使用新的测试场景，已公开两轮测试只作回归。', '',
             '## 运行与继续优化', '',
             '```bash',
             'python team_run.py learned --checkpoint wsl-support/learned_control/outputs/refined_20261014/primary_ppo.pt --show',
             'python wsl-support/learned_control/refine.py --output wsl-support/learned_control/outputs/team_refined',
             'python wsl-support/learned_control/refinement_evaluate.py --experiment wsl-support/learned_control/outputs/team_refined --seed 2026101704',
             '```', '',
             '保留旧模型、两轮所有阶段权重、示范数据、场景、逐场 CSV、验证日志、测试汇总、模型哈希和成功/失败录像。默认模型是否切换取决于发布检查，见 release_decision.json；不能用本测试集继续调参。', '']
    (out/'最终优化报告.md').write_text('\n'.join(text), encoding='utf-8')
    (out/'complete.json').write_text(json.dumps({'training': 'complete', 'evaluation': 'complete', 'report': 'complete'}))
    print(json.dumps({'candidate': selected, 'fresh': main, 'paired': paired['fresh_course_holdout']}, indent=2))


if __name__ == '__main__': main()
