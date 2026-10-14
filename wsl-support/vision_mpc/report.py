"""Generate figures and a factual Chinese report from saved evaluation CSV."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('results', type=Path)
    args = p.parse_args()
    out = args.results
    rows = list(csv.DictReader((out/'results.csv').open()))
    summary = json.loads((out/'summary.json').read_text())
    methods = list(summary)
    labels = {'direct': 'Direct', 'astar': 'A* tracking', 'mpc': 'Vision + A* + MPC', 'mpc_v1': 'MPC V1',
              'no_hunter': 'MPC without hunter', 'no_delay': 'MPC without delay'}
    rates = np.array([summary[m]['success_rate'] for m in methods])
    intervals = np.array([summary[m]['success_rate_95ci'] for m in methods])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([labels[m] for m in methods], rates*100, color=['#8d9ca8', '#6699bb', '#38a67d', '#e2aa51'][:len(methods)])
    ax.errorbar(np.arange(len(methods)), rates*100,
                yerr=np.maximum(0, np.stack((rates-intervals[:, 0], intervals[:, 1]-rates))*100),
                fmt='none', color='black', capsize=5)
    ax.set_ylim(0, 105); ax.set_ylabel('Success rate (%)')
    ax.set_title('Same scenarios, original environment, 95% Wilson intervals')
    ax.grid(axis='y', alpha=.2); fig.tight_layout()
    fig.savefig(out/'success_comparison.png', dpi=180); plt.close(fig)
    reasons = ['success', 'caught', 'obstacle_collision', 'timeout', 'perception_failure', 'ambiguous_terminal']
    fig, ax = plt.subplots(figsize=(10, 5)); base = np.zeros(len(methods))
    colors = ['#38a67d', '#d75959', '#e2aa51', '#8d9ca8', '#836ab1', '#444444']
    for reason, color in zip(reasons, colors):
        values = np.array([summary[m]['outcomes'].get(reason, 0) for m in methods])
        if values.any():
            ax.bar([labels[m] for m in methods], values, bottom=base, label=reason, color=color)
            base += values
    ax.set_ylabel('Episodes'); ax.set_title('Termination causes (collision is not success)')
    ax.legend(); fig.tight_layout(); fig.savefig(out/'outcomes.png', dpi=180); plt.close(fig)
    lines = ['# 视觉规划与 MPC：实验报告', '',
             '本报告由真实测试 CSV 和汇总文件生成，不将验证集或演示成功率冒充正式测试结果。', '',
             '本报告原 100 场与分层 120 场数据对应 V1。当前程序默认使用 V2，历史成绩不自动变成新版成绩。', '',
             '## 方法与数据来源', '',
             '图像颜色分割和连通域获得物体中心；连续图像估计速度；A* 生成绕障参考路线；'
             '采样式 MPC 用独立模型预测，包含半隐式积分、速度上限、软碰撞和一步动作延迟。'
             '搜索使用 CEM（交叉熵采样优化），并非最初设计中的束搜索。', '',
             '没有修改老师的 core.py、environment.py 或 simple_tag.py。每个回合重新创建环境。'
             '控制器不接收真实位置或原 observation；独立 evaluator 读取真实位置核对终止原因和视觉误差，'
             '该信息只用于指标记录，不返回控制器。', '',
             '正式场景由固定随机种子生成，起始坐标范围 [-0.72,0.72]，初始物体间距至少 0.25，'
             '距离障碍至少 0.13。因而结果不能推广为任意合法初始位置的成功率。'
             '所有方法使用同一 cases.json 和 config.json，回合最多 500 步。', '',
             '## 结果', '',
             '| 方法 | 回合数 | 成功 | 成功率 | 95% Wilson 区间 | 被捕 | 撞障碍 | 超时 |',
             '|---|---:|---:|---:|---|---:|---:|---:|']
    for m in methods:
        s = summary[m]; o = s['outcomes']; lo, hi = s['success_rate_95ci']
        lines.append(f'| {m} | {s["episodes"]} | {s["successes"]} | {s["success_rate"]:.1%} | '
                     f'{lo:.1%}–{hi:.1%} | {o.get("caught",0)} | {o.get("obstacle_collision",0)} | {o.get("timeout",0)} |')
    lines += ['', '![成功率与置信区间](success_comparison.png)', '', '![结束原因](outcomes.png)', '',
              '## 运行开销与视觉诊断', '',
              '| 方法 | 成功回合平均步数 | 各回合中位耗时的中位数/ms | 最大小球中心误差/像素 | 感知失败回合 |',
              '|---|---:|---:|---:|---:|']
    for m in methods:
        selected = [r for r in rows if r['method'] == m]
        med = np.median([float(r['decision_p50_ms']) for r in selected])
        errors = [float(r['max_center_error_pixels']) for r in selected if r['max_center_error_pixels']]
        mean = summary[m]['mean_success_steps']
        lines.append(f'| {m} | {mean if mean is not None else "无成功回合"} | {med:.2f} | '
                     f'{max(errors) if errors else "无数据"} | {summary[m]["outcomes"].get("perception_failure",0)} |')
    lines += ['', '耗时包括渲染、视觉、规划和控制；短回合的 95 分位数容易被首次规划开销放大。'
              '上述耗时是“回合内中位数”的跨回合中位数，不是全体帧的合并分位数。'
              '视觉误差仅统计检测成功时的红/绿球中心，不等于所有对象的检测召回率或鲁棒性证明。', '',
              '## 消融含义', '',
              '- direct：只朝目标控制，不利用规划避障。',
              '- astar：全局规划加速度跟踪，不进行猎手预测或 MPC。',
              '- no_hunter：保留 MPC 避障，去掉猎手风险和被捕约束；正式环境仍有猎手。',
              '- mpc：完整图像、规划、动态预测控制。', '',
              '## 局限与失败分析', '',
              '不承诺满分。红球最高速度大于绿球；近距离初始化、边缘目标、强遮挡、颜色改变等未被此随机测试集完整覆盖。'
              '采样 MPC 是有限预测窗口的近似优化，可能卡住、超时、找到不安全候选或漏掉更好的逃脱路线。'
              '候选碰撞使用独立软碰撞模型；视觉速度存在像素误差。', '',
              '异常检测采用明确失败退出，尚未实现长时间遮挡预测。A* 无路时使用直接目标参考，'
              '由 MPC 保留碰撞惩罚，但它不代表有可行全局路线。', '',
              '查看 results.csv 中所有非 success 行及对应 trajectory.json/final.png；保留失败，'
              '不以 reward 或 done_n[1] 代替到达指标。测试结束后不再根据本测试集修改控制参数。', '',
              '## 复现与评分对应', '',
              '使用 run.py --mode compare --cases 本目录/cases.json --config 本目录/config.json '
              '--output 新结果目录，随后运行 report.py 新结果目录。', '',
              '图像输入：展示检测叠加及数据隔离；避障/避猎手/到达：展示结束分类和配对对比；'
              '汇报：说明模型、延迟、对照、置信区间和失败案例。评分转换由老师决定，不能据本表直接计算官方分数。']
    stress_path = out.parent/'stress_test'/'summary.json'
    delay_path = out.parent/'delay_ablation'/'summary.json'
    if stress_path.exists():
        stress = json.loads(stress_path.read_text())['mpc']
        lines += ['', '## 独立压力测试（不混入正式 100 场）', '',
                  f'另选 {stress["episodes"]} 个近距离追击、边缘目标和通道场景，成功 '
                  f'{stress["successes"]} 次（{stress["success_rate"]:.1%}）。'
                  f'结束原因：{stress["outcomes"]}。', '',
                  '失败案例 near_pursuit：猎手与绿球初始距离 0.12，4 步后被捕；'
                  '保存了完整轨迹、终止图和回放视频。程序没有根据这组压力结果继续调参。'
                  '这说明不能把正式随机测试的 100% 当作近距离追击保证。']
    if delay_path.exists():
        delay = json.loads(delay_path.read_text())['no_delay']
        lines += ['', '## 动作延迟消融', '',
                  f'在原 12 个验证案例中，忽略动作延迟的版本成功 {delay["successes"]}/{delay["episodes"]}；'
                  '完整版本也为 12/12，因此此小样本没有证明延迟补偿提高成功率。'
                  '保留时序补偿的依据是原代码与单元测试明确存在一步延迟。']
    lines += ['', '## 显示验收', '',
              '原始与标注两个窗口使用 Pyglet，适配层在读取原图前选择正确 OpenGL 上下文，'
              '避免原渲染器的清屏顺序在多窗口情况下产生残影。带标注的默认完整演示 74 步到达，'
              '视频位于 ../demo_final/mpc_demo.mp4。显示与录制额外增加开销，不保证每一步都满足 0.1 秒硬实时期限。']
    stratified = out.parent/'stratified_20261013'
    if (stratified/'summary.json').exists():
        expanded = json.loads((stratified/'summary.json').read_text())
        a, m = expanded['astar'], expanded['mpc']
        lines += ['', '## 2026-10-13 新增分层测试', '',
                  f'冻结原控制参数，新增 10 类、每类 12 个、共 120 个初始位置，配对运行 A* 与 MPC，共 240 回合。'
                  f'MPC 成功 {m["successes"]}/{m["episodes"]}（{m["success_rate"]:.1%}），'
                  f'A* 成功 {a["successes"]}/{a["episodes"]}（{a["success_rate"]:.1%}）。'
                  '新增分组涵盖三档近距离追击、猎手挡路、靠边目标、靠障碍起点/目标、角落起点和长路线。', '',
                  'MPC 两次失败均为被捕，未发生绿球撞障碍、超时或感知失败。'
                  '最小追击间距组与目标靠障碍组各成功 11/12，其余八组各成功 12/12。'
                  '每组样本少，不能推断该组所有位置必然成功。', '',
                  '17 个 MPC 回合记录到猎手接触障碍，均成功；没有接触的 103 回合成功 101 次。'
                  '这是观察性统计，不能证明接触导致成功。猎手接触不违反当前原环境的终止规则。', '',
                  '[完整分层报告、接触统计与全部失败案例](../stratified_20261013/分层测试报告.md)', '',
                  '![分层成功率](../stratified_20261013/stratified_success.png)', '',
                  '旧 100 场、旧 8 场压力测试和新增 120 场来自不同生成条件，应分别汇报，'
                  '不能把它们合并成一个未经说明的总体成功率。本次控制器和动力学源码哈希与旧测试一致。']
    optimization = out.parent/'v2_independent'/'优化验证报告.md'
    if optimization.exists():
        lines += ['', '## V2 优化验证', '',
                  'V2 在预测危险时引入临时避险路线，允许先远离目标，并使用更长预测。两个原失败案例转为开发案例，'
                  '原测试集用于回归，新独立场景用于比较泛化。不得用开发案例成功直接推断总体效果提升。', '',
                  '[新版验证结果、回归与独立比较](../v2_independent/优化验证报告.md)']
    (out/'实验报告.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(out/'实验报告.md')


if __name__ == '__main__':
    main()
