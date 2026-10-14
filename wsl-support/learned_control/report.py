"""Summarize frozen evaluations without modifying checkpoints or test cases."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


def main():
    out = HERE/'outputs/main_20261014'
    summary = json.loads((out/'evaluation/summary.json').read_text())
    rows = list(csv.DictReader((out/'evaluation/results.csv').open()))
    names = {'astar': 'A*', 'mpc': 'MPC V2', 'old_ppo': 'Old PPO', 'imitation': 'Imitation', 'learned': 'CNN + PPO'}
    methods = list(names)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for ax, (group, results) in zip(axes, summary.items()):
        rates = [results[m]['success_rate'] for m in methods]
        low = [rates[i]-results[m]['success_rate_95ci'][0] for i, m in enumerate(methods)]
        high = [max(0, results[m]['success_rate_95ci'][1]-rates[i]) for i, m in enumerate(methods)]
        ax.bar(list(names.values()), rates, yerr=[low, high], capsize=4)
        ax.set_ylim(0, 1.13); ax.set_ylabel('Success fraction (Wilson 95% CI)')
        ax.set_title('Course: fixed obstacles (60)' if group.startswith('course') else 'Random obstacles stress (30)')
        ax.tick_params(axis='x', rotation=20)
        for i, m in enumerate(methods):
            ax.text(i, rates[i]+.02, f"{results[m]['successes']}/{results[m]['episodes']}", ha='center')
    fig.savefig(out/'success_comparison.png', dpi=160); plt.close(fig)
    text = ['# 训练图像控制主模型：冻结实验报告（2026-10-14）', '',
            '## 要求与实现', '',
            '教师原 readme 明确障碍位置固定；新 Course design PPT 第5页要求猎人、小球、目标初始位置随机，第6～8页要求训练机器学习模型，以图像作为输入。随机障碍并非已发现的必考要求，作为额外压力测试单独报告。原环境、猎人控制、碰撞和终止规则保持 ca0f9ab 基线，完整性检查通过。', '',
            '主路线：RGB → 训练 PixelNet CNN → 连通域和图像速度估计 → 训练 Actor → 二维有界加速度。CNN 为三层1×1卷积；Actor/Critic为两层256单元。模型接口仅接受 RGB，内部21维特征来自图像及自身动作历史，不从环境读取真实状态；真实坐标只用于事后误差诊断。该管线分阶段训练，不能宣称可微端到端。推理不调用 MPC 或规划器兜底。', '',
            '训练 seed1401：160场 MPC 图像专家示范成功；三轮 DAgger 后共10833条示范；CNN 1000步，平衡像素分类准确率99.707%；PPO 16384步微调，学习率3e-5，附加示范正则。固定40场验证；主模型仅在 PPO 阶段按验证成绩选择，选中 ppo16384（30/40），冻结后使用新的测试场景。训练约56.5分钟。', '',
            '## 配对独立测试', '',
            '测试 seed2026101404 固定障碍60场，seed2026101405 随机障碍30场；五个方法使用同批场景，共450场。成功需同时没有撞障碍、没有被捕并到达目标。下列置信区间为 Wilson 95%，不代表跨训练种子的波动。', '']
    for group, results in summary.items():
        text += ['### '+group, '', '| 方法 | 成功 | 95%区间 | 成功平均步数 | 失败 |', '|---|---:|---:|---:|---|']
        for m, r in results.items():
            ci = r['success_rate_95ci']
            failures = ', '.join(f'{k}:{v}' for k, v in r['outcomes'].items() if k != 'success') or '无'
            text.append(f"| {names[m]} | {r['successes']}/{r['episodes']} ({r['success_rate']:.1%}) | {ci[0]:.1%}–{ci[1]:.1%} | {r['mean_success_steps']:.2f} | {failures} |")
        text += ['']
    selected = [r for r in rows if r['method'] == 'learned']
    diagnostics = {}
    for group in summary:
        subset = [r for r in selected if r['group'] == group]
        contacts = [r for r in subset if int(r['hunter_obstacle_contact_frames']) > 0]
        free = [r for r in subset if int(r['hunter_obstacle_contact_frames']) == 0]
        diagnostics[group] = {
            'max_center_error_pixels': max(float(r['max_center_error_pixels']) for r in subset),
            'median_episode_mean_center_error_pixels': float(np.median([float(r['mean_center_error_pixels']) for r in subset])),
            'minimum_hunter_command_cosine': min(float(r['min_hunter_command_alignment_cosine']) for r in subset if r['min_hunter_command_alignment_cosine']),
            'no_hunter_obstacle_contact': {'episodes': len(free), 'successes': sum(r['reason'] == 'success' for r in free)},
            'with_hunter_obstacle_contact': {'episodes': len(contacts), 'successes': sum(r['reason'] == 'success' for r in contacts)}}
    (out/'diagnostics.json').write_text(json.dumps(diagnostics, indent=2))
    text += ['## 定位与猎人核验', '',
             '单帧阈值感知审计：100个课程布局 + 100个随机障碍布局，各物体中心误差最大均小于1个像素。此结果不能等同于动态 CNN 的全程误差；动态遮挡/接触会引入更大误差。训练 CNN 的8个随机布局定位测试要求小于3像素并通过。', '']
    for group, d in diagnostics.items():
        a, b = d['no_hunter_obstacle_contact'], d['with_hunter_obstacle_contact']
        text += [f"- {group}：主模型全程最大中心误差 {d['max_center_error_pixels']:.3f} 像素（400×400感知坐标）；每回合平均误差的中位数 {d['median_episode_mean_center_error_pixels']:.3f} 像素。猎人控制与猎物方向最小余弦 {d['minimum_hunter_command_cosine']:.6f}。猎人未接触障碍的回合成功 {a['successes']}/{a['episodes']}；接触障碍的回合成功 {b['successes']}/{b['episodes']}。这是观察性分组，不能作因果结论。"]
    text += ['', '猎物不动的原环境核验中，猎人在第29步捕获，控制方向余弦1.0。猎人不是刻意躲开猎物：原控制方向追向猎物，但一步延迟、惯性和障碍力可改变实际速度。', '',
             '## 限制与后续优化', '',
             'PPO主模型相对模仿模型课程测试多成功3场、压力测试多成功1场，不能据此证明显著提升，更不能证明优于未实现的 SAC。选择 PPO 是为了现有实现、连续有界动作及训练维护成本；当前主要收益来自专家示范与 DAgger。MPC更可靠但没有训练模型，作为对照和专家保留。', '',
             '默认场景主模型第22步被捕获，默认录像明确保存为失败。课程测试仍有5场被捕、13场撞障碍；主模型完成实现与实验，性能尚未达到所有场景可靠通过。下一轮优先增加近障碍/猎人接近的困难示范，改善遮挡感知并进行多训练种子验证；用全新测试种子确认收益。已公开场景只用于回归，不能继续拿本测试集选模型。', '',
             '验证在 WSL2 Ubuntu22.04 CPU 完成；Windows 支持同源代码、Python3.10及CPU模型，已有路径/入口检查，尚未完整验证原生 Windows OpenGL 图形运行。', '',
             '## 可复现资料', '',
             '- 本目录模型：primary_ppo.pt、imitation_best.pt、perception.pt及所有阶段检查点；训练配置位于检查点与 train.py，validation.json 保存选模过程。',
             '- evaluation/cases.json、results.csv、summary.json：冻结场景、450场逐场结果及汇总；diagnostics.json保存猎人接触分组。',
             '- representative_replays/：按冻结测试结果选择成功/被捕/撞障碍案例重新录像，绝不用于选模；default_demo/保存默认失败。',
             '- course_audit/：原代码 SHA256、200场单帧审计及静止猎物追击轨迹。',
             '- 重跑：python team_run.py learned-evaluate --seed 2026101504 --output wsl-support/learned_control/outputs/new_test', '']
    (out/'主模型实验报告.md').write_text('\n'.join(text), encoding='utf-8')
    cases = json.loads((out/'evaluation/cases.json').read_text())
    replay = []
    for reason in ('success', 'caught', 'obstacle_collision'):
        chosen = next(r for r in selected if r['reason'] == reason and r['group'].startswith('course'))
        replay.append(next(c for c in cases if c['id'] == chosen['case']))
    (out/'representative_cases.json').write_text(json.dumps(replay, indent=2))
    print(json.dumps(diagnostics, indent=2))


if __name__ == '__main__':
    main()
