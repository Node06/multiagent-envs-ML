"""Build plots and an evidence-based Chinese report from completed run files."""
import argparse
import csv
import json
import hashlib
import platform
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import cv2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('experiment', type=Path)
    args = p.parse_args()
    root = args.experiment
    spec = json.loads((root/'experiment.json').read_text())
    summary = json.loads((root/'holdout/summary.json').read_text())
    manifest = json.loads((root/'holdout/manifest.json').read_text())
    checkpoints = manifest['checkpoints']
    cfg = manifest['ppo_config']
    source_root = Path(__file__).resolve().parent
    source_files = list(source_root.glob('*.py')) + list(source_root.glob('*.json'))
    source_files += [source_root.parent/'vision_mpc'/name for name in
                     ('perception.py', 'state_estimator.py', 'controller_v1.py', 'controller_v2.py',
                      'global_planner.py', 'dynamics.py', 'evaluator.py', 'geometry.py')]
    source_files += [source_root.parent.parent/'multiagent'/name for name in
                     ('core.py', 'environment.py', 'scenarios/simple_tag.py')]
    (root/'reproducibility.json').write_text(json.dumps({
        'python': platform.python_version(), 'torch': torch.__version__,
        'numpy': np.__version__, 'opencv': cv2.__version__,
        'source_sha256': {str(f.relative_to(source_root.parent.parent)): hashlib.sha256(f.read_bytes()).hexdigest()
                          for f in source_files}}, indent=2))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    diagnostic_fig, diagnostic_axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    validations = []
    for seed in spec['seeds']:
        folder = root/f'seed{seed}'
        with (folder/'train.csv').open() as f:
            train = list(csv.DictReader(f))
        val = [json.loads(line) for line in (folder/'validation.jsonl').read_text().splitlines()]
        validations.append(val)
        axes[0].plot([float(r['steps']) for r in train], [float(r['train_recent_success']) for r in train], label=f'seed {seed}')
        axes[1].plot([r['steps'] for r in val], [r['success_rate'] for r in val], marker='o', label=f'seed {seed}')
        diagnostic_axes[0].plot([float(r['steps']) for r in train], [float(r['value_loss']) for r in train], label=f'seed {seed}')
        diagnostic_axes[1].plot([float(r['steps']) for r in train], [float(r['entropy']) for r in train], label=f'seed {seed}')
    for ax, title in zip(axes, ['Training success (last 100 episodes; changing curriculum)', 'Fixed validation success']):
        ax.set(xlabel='Environment steps', ylabel='Success rate', ylim=(0, 1), title=title)
        ax.grid(alpha=.2); ax.legend()
    fig.savefig(root/'learning_curves.png', dpi=160); plt.close(fig)
    for ax, title in zip(diagnostic_axes, ['Value loss', 'Bounded action differential entropy']):
        ax.set(xlabel='Environment steps', title=title)
        ax.grid(alpha=.2); ax.legend()
    diagnostic_fig.savefig(root/'optimization_diagnostics.png', dpi=160); plt.close(diagnostic_fig)
    methods = list(summary)
    rates = np.array([summary[m]['success_rate'] for m in methods])
    ci = np.array([summary[m]['success_rate_95ci'] for m in methods])
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    ax.bar(methods, rates, color=['#6c879c', '#319879']+['#c58a38']*(len(methods)-2))
    ax.errorbar(methods, rates, yerr=np.maximum(0, np.stack([rates-ci[:, 0], ci[:, 1]-rates])),
                fmt='none', ecolor='black', capsize=4)
    ax.set(ylabel='Success rate (95% Wilson CI)', ylim=(0, 1.05), title=f'Paired holdout: {spec["test_count"]} identical starts')
    fig.savefig(root/'holdout_success.png', dpi=160); plt.close(fig)
    with (root/'holdout/results.csv').open() as f:
        rows = list(csv.DictReader(f))
    # Select by validation only; ties use validation episode length.
    scores = {(f'ppo_seed{r["seed"]}', r['steps']):
              (r['success_rate'], -(r['mean_success_steps'] or float('inf')))
              for val in validations for r in val}
    selected = max(checkpoints, key=lambda c: scores[(c['method'], c['steps'])])
    chosen = selected['method']
    lines = ['# 视觉状态 PPO 实现与独立对照实验', '',
             '## 实验设置', '',
             f'- 随机种子：{spec["seeds"]}；每个种子训练 {spec["steps_per_seed"]:,} 步，总训练预算 {len(spec["seeds"])*spec["steps_per_seed"]:,} 步。',
             '- CPU PyTorch；独立 on-policy PPO，无 MPC 示范或预训练。',
             '- 800×800 原环境 RGB 图像；颜色分割及连续帧差分得到 21 维状态。真实坐标仅用于终止判定，不输入策略。',
             '- 两个 128 单元隐藏层，tanh 高斯策略；保留动作幅度、原一步延迟及原物理规则。',
             '- 训练前 35% 预算逐步增加初始场景难度，之后随机混合；没有修改追击规则。',
             '- 20 个固定验证场景选检查点；最终测试在全部模型冻结后生成。',
             f'- 最终测试：{spec["test_count"]} 个新随机场景，seed={spec["test_seed"]}；每回合上限 500 步。',
             '- 所有方法通过同一 VisualEnv，使用相同感知、初始化、动作约束和结果判定。',
             '- MPC 对照明确使用 controller_v1，对应原版算法；不将新实验与已有测试集混合。',
             f'- 奖励：成功 +{cfg["success_reward"]:g}，真实终止失败 {cfg["failure_reward"]:g}，每步 -{cfg["step_cost"]:g}，'
             f'目标距离势函数 Φ=-{cfg["potential_scale"]:g}×距离；使用 γΦ(s′)-Φ(s) 塑形。',
             f'- 超参数：学习率 {cfg["learning_rate"]}，γ={cfg["gamma"]}，GAE λ={cfg["gae_lambda"]}，'
             f'裁剪 {cfg["clip"]}，rollout={cfg["rollout_steps"]}，batch={cfg["batch_size"]}，每批最多 {cfg["epochs"]} 轮更新，'
             f'熵系数 {cfg["entropy_coef"]}，梯度范数上限 {cfg["max_grad_norm"]}，KL 提前停止阈值 {cfg["target_kl"]}。',
             '', '## 独立测试结果', '',
             '| 方法 | 成功 | 成功率（95%区间） | 被捕 | 撞障碍 | 超时 | 感知失败 | 其他失败 | 成功平均步数 | 控制中位耗时均值 ms | 含视觉单步中位耗时均值 ms |',
             '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for method, s in summary.items():
        low, high = s['success_rate_95ci']; outcomes = s['outcomes']
        steps = f'{s["mean_success_steps"]:.1f}' if s['mean_success_steps'] is not None else '—'
        other_failures = sum(v for k, v in outcomes.items() if k not in
                             ('success', 'caught', 'obstacle_collision', 'timeout', 'perception_failure'))
        lines.append(f'| {method} | {s["successes"]}/{s["episodes"]} | {s["success_rate"]:.1%}（{low:.1%}～{high:.1%}） | '
                     f'{outcomes.get("caught", 0)} | {outcomes.get("obstacle_collision", 0)} | {outcomes.get("timeout", 0)} | '
                     f'{outcomes.get("perception_failure", 0)} | {other_failures} | '
                     f'{steps} | {s["mean_episode_decision_p50_ms"]:.3f} | {s["mean_episode_step_with_vision_p50_ms"]:.3f} |')
    ppo_rates = [s['success_rate'] for method, s in summary.items() if method.startswith('ppo_')]
    seed_std = np.std(ppo_rates, ddof=1) if len(ppo_rates) > 1 else 0.
    lines += ['', f'{len(ppo_rates)} 个 PPO 训练种子的测试成功率均值为 {np.mean(ppo_rates):.1%}，种子间样本标准差为 {seed_std:.1%}。'
              '同一测试场景在不同种子中重复使用，不能把它们当作完全独立的三倍测试样本。', '',
              '## 模型选择与配对差异', '',
              f'仅依据验证集选择演示模型：**{chosen}**，检查点训练步数 {selected["steps"]:,}，验证成功率 {selected["validation_success"]:.1%}。',
              '各检查点路径及 SHA-256 位于 holdout/manifest.json。']
    selected_rows = {r['case']: r['reason'] == 'success' for r in rows if r['method'] == chosen}
    for baseline in ('astar', 'mpc_v1'):
        other = {r['case']: r['reason'] == 'success' for r in rows if r['method'] == baseline}
        wins = sum(selected_rows[k] and not other[k] for k in selected_rows)
        losses = sum(not selected_rows[k] and other[k] for k in selected_rows)
        lines.append(f'- 对 {baseline}：PPO 独有成功 {wins} 场，对照独有成功 {losses} 场，配对成功率差 {(wins-losses)/len(other):+.1%}。')
    selected_s, mpc_s = summary[chosen], summary['mpc_v1']
    conclusion = ('本轮 PPO 的独立测试成功率低于 MPC，未证明其能够替代或增强 MPC；它已形成可复现的深度强化学习对照。'
                  if selected_s['success_rate'] < mpc_s['success_rate'] else
                  '本轮 PPO 测试成功率达到或超过 MPC；此结果仅适用于本次场景分布，需更多独立测试确认。')
    lines += ['', '## 结果解释与限制', '', conclusion,
              '- 已完成固定预算训练；不据此声称策略已收敛或达到最优。',
              '- 成功平均步数只统计成功回合，不能脱离成功率单独比较效率。',
              '- 耗时为同一运行环境下各回合中位数的均值；训练并行、最终评估串行，视频和文件写入不计入控制计时。',
              '- 含视觉单步耗时包括策略、物理推进、渲染和识别；控制耗时仅包括策略或规划计算。',
              '- 上述单步耗时不含回合初始化、网络加载和初始地图构建。',
              '- 训练曲线采用不断变化的课程场景，泛化判断以固定验证与独立测试为准。',
              '- 图像识别失败计入结果，不使用真实状态补齐。',
              '', '## 图表', '', '![学习曲线](learning_curves.png)', '', '![优化诊断](optimization_diagnostics.png)', '', '![独立测试](holdout_success.png)', '',
              '## 复现', '', '在项目根目录、WSL multiagent-ml 环境运行：', '', '```bash',
              'python -m unittest discover -s wsl-support/vision_ppo/tests -v',
              'python wsl-support/vision_ppo/run_experiment.py --output wsl-support/vision_ppo/outputs/reproduction --steps 131072 --test-count 60',
              '```', '', '最终结果逐场数据：holdout/results.csv；训练日志：seed*/train.csv；验证日志：seed*/validation.jsonl。']
    (root/'实验报告.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    (root/'selected_model.json').write_text(json.dumps(selected, indent=2))
    print('REPORT '+str(root/'实验报告.md'), flush=True)


if __name__ == '__main__':
    main()
