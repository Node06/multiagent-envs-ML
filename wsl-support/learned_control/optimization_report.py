"""Report completed, frozen optimization results, including regressions."""
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
    out = HERE/'outputs/optimized_20261014'
    summary = json.loads((out/'evaluation/summary.json').read_text())
    training = json.loads((out/'training_summary.json').read_text())
    selection = json.loads((out/'primary_selection.json').read_text())
    selected = summary['selected_main']
    rows = list(csv.DictReader((out/'evaluation/results.csv').open()))
    seeds = [f"seed{r['seed']}" for r in training['seeds']]
    methods = ['previous_main', *seeds, 'optimized_imitation', 'mpc']
    comparison = {}
    for group in ('fresh_course_holdout', 'random_obstacles_stress', 'published_course_regression'):
        old = {r['case']: r for r in rows if r['group'] == group and r['method'] == 'previous_main'}
        new = {r['case']: r for r in rows if r['group'] == group and r['method'] == selected}
        improved = [k for k in old if old[k]['reason'] != 'success' and new[k]['reason'] == 'success']
        regressed = [k for k in old if old[k]['reason'] == 'success' and new[k]['reason'] != 'success']
        n = len(improved)+len(regressed)
        rates = [summary[group][m]['success_rate'] for m in seeds]
        comparison[group] = dict(repaired_cases=improved, regressed_cases=regressed,
                                  paired_mcnemar_exact_p=binomtest(len(improved), n, .5).pvalue if n else 1.,
                                  seed_success_mean=float(np.mean(rates)), seed_success_sample_std=float(np.std(rates, ddof=1)))
    (out/'paired_comparison.json').write_text(json.dumps(comparison, indent=2))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    labels = ['Previous', *seeds, 'Imitation', 'MPC']
    for ax, group in zip(axes, ('fresh_course_holdout', 'published_course_regression')):
        rates = [summary[group][m]['success_rate'] for m in methods]
        lo = [max(0, rates[i]-summary[group][m]['success_rate_95ci'][0]) for i, m in enumerate(methods)]
        hi = [max(0, summary[group][m]['success_rate_95ci'][1]-rates[i]) for i, m in enumerate(methods)]
        ax.bar(labels, rates, yerr=[lo, hi], capsize=3); ax.set_ylim(0, 1.15)
        ax.set_title('Fresh holdout (100)' if group.startswith('fresh') else 'Published regression (60)')
        ax.set_ylabel('Success fraction; Wilson 95% CI'); ax.tick_params(axis='x', rotation=25)
        for i, m in enumerate(methods): ax.text(i, rates[i]+.02, f"{summary[group][m]['successes']}/{summary[group][m]['episodes']}", ha='center', fontsize=8)
    fig.savefig(out/'paired_success.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    for seed in seeds:
        stages = json.loads((out/seed/'validation.json').read_text())
        ax.plot([s['stage'] for s in stages], [s['success_rate'] for s in stages], marker='o', label=seed)
    ax.set_ylim(0, 1); ax.set_ylabel('Validation success fraction (40 cases)'); ax.legend()
    fig.savefig(out/'validation_curve.png', dpi=160); plt.close(fig)
    text = ['# 安全特征与困难场景训练优化报告', '',
            '## 改动与边界', '',
            '保留旧 CNN 和主模型，老师原环境、猎人、碰撞规则未改。新增 safety56 模型：原21维图像特征 + 27维障碍安全特征 + 4维猎人相对运动特征 + 4维边界距离。净距离、接近速度、直线外推的最近接距离与时间、目标方向几何、待执行动作下一步距离均来自图像估计及自身动作。外推是特征，不能当作精确物理预测。推理仍是 RGB → CNN → 训练 Actor，没有 MPC/A* 兜底。', '',
            '复用原10833条训练示范，近障碍/接近猎人样本加权；新增两轮困难 DAgger，目标在障碍背后的新布局占一半，其他为普通随机布局。每轮从验证最佳历史模型开始。PPO 使用较小学习率1e-5、clip0.1、KL阈值0.01，将示范损失并入同一次更新；用图像安全势函数进行势能奖励塑形，终止规则不变。', '',
            '本轮是多项改动组合的分阶段实验，不能将提升单独归因于安全特征、加权数据或 PPO。CNN复用旧训练模型；未增加记忆网络。', '',
            '## 训练与选模', '',
            '3个训练种子分别独立训练，共用 seed2026101416 的40场新验证集。只按验证成绩从 PPO 阶段选择主模型，测试前保存模型哈希。新测试100场 seed2026101428；随机障碍30场 seed2026101429；旧60场仅回归。测试结果不反馈训练或跨种子选模。', '',
            '| 种子 | 示范条数 | PPO步数 | 最佳阶段验证率 | PPO最佳验证率 |', '|---|---:|---:|---:|---:|']
    for r in training['seeds']: text.append(f"| {r['seed']} | {r['samples']} | {r['ppo_steps']} | {r['best']:.1%} | {r['best_ppo']:.1%} |")
    text += ['', f"选定主模型：{selected} / {selection['stage']}；训练耗时 {training['seconds']/60:.1f} 分钟。", '', '## 冻结配对测试', '']
    for group in ('fresh_course_holdout', 'random_obstacles_stress', 'published_course_regression'):
        text += ['### '+group, '', '| 模型 | 成功 | Wilson 95%区间 | 失败 |', '|---|---:|---:|---|']
        for m in methods:
            r = summary[group][m]; ci = r['success_rate_95ci']
            fails = ', '.join(f'{k}:{v}' for k, v in r['outcomes'].items() if k != 'success') or '无'
            label = m+'（主模型）' if m == selected else m
            text.append(f"| {label} | {r['successes']}/{r['episodes']} ({r['success_rate']:.1%}) | {ci[0]:.1%}–{ci[1]:.1%} | {fails} |")
        c = comparison[group]
        text += ['', f"主模型修复旧模型失败 {len(c['repaired_cases'])} 场，同时新增退化 {len(c['regressed_cases'])} 场；双侧精确 McNemar p={c['paired_mcnemar_exact_p']:.4f}。三个训练种子成功率均值 {c['seed_success_mean']:.1%}，样本标准差 {c['seed_success_sample_std']:.1%}。", '']
    text += ['## 默认场景与局限', '']
    for m in ('previous_main', selected, 'optimized_imitation', 'mpc'):
        r = next(a for a in rows if a['group'] == 'default_demo' and a['method'] == m)
        text.append(f"- {m}：{r['reason']}，{r['steps']}步；录像保存在 evaluation/。")
    main = summary['fresh_course_holdout'][selected]
    text += ['', f"新独立测试主模型 {main['successes']}/{main['episodes']}。90%只是阶段目标，是否达到以本表为准，不能保证任意合法位置成功。PPO与优化后的模仿模型使用同批测试比较，不能把所有收益归给 PPO，也未完成同预算 SAC 比较。", '',
             '三种子和100场仍不足以覆盖所有考核布局；随机障碍不代表课程必考要求。原生 Windows 图形运行尚未完整验证，WSL CPU 为完整实验平台。已公开测试集后续只用于回归，下轮优化使用新测试种子。', '',
             '## 运行与资料', '',
             '- 指定模型：python team_run.py learned --checkpoint wsl-support/learned_control/outputs/optimized_20261014/primary_ppo.pt --show',
             '- 重训练：python wsl-support/learned_control/optimize.py --bc-epochs 80 --dagger-rounds 2 --dagger-cases 60 --ppo-steps 8192 --validation-count 40 --output wsl-support/learned_control/outputs/new_optimization',
             '- configuration.json、各种子validation.json、demonstrations.npz、所有检查点与主模型选模记录保留。',
             '- evaluation/cases.json、results.csv、summary.json、frozen_models.json保留全部场景和逐场结果；paired_comparison.json列出修复和退化案例。', '']
    (out/'优化实验报告.md').write_text('\n'.join(text), encoding='utf-8')
    print(json.dumps({'selected': selected, 'fresh': main, 'comparison': comparison['fresh_course_holdout']}, indent=2))


if __name__ == '__main__': main()
