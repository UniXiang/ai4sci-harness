"""Mock LLM backend — for testing the framework without API calls.

Generates plausible but deterministic responses based on agent type.
"""

import time
from typing import Optional

from .base import LLMBackend


class MockBackend(LLMBackend):
    """Mock backend that returns canned responses for framework testing.

    Responses vary by detected keywords in the prompt to simulate
    different agent behaviors (Executor produces code, Critic approves, etc.).
    """

    def __init__(self, config: dict):
        self.model = config.get("model", "mock")
        self.max_tokens_default = config.get("max_tokens", 4096)
        self.temperature_default = config.get("temperature", 0.3)

    @property
    def name(self) -> str:
        return "Mock"

    def generate(self, system_prompt: str, user_prompt: str,
                 max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None) -> str:
        # Simulate small delay
        time.sleep(0.1)

        combined = (system_prompt + " " + user_prompt).lower()

        if "executor" in combined or "代码" in user_prompt:
            return self._mock_executor(user_prompt)
        elif "critic" in combined or "审查" in user_prompt:
            return self._mock_critic(user_prompt)
        elif "planner" in combined or "规划" in user_prompt:
            return self._mock_planner(user_prompt)
        elif "researcher" in combined or "调研" in user_prompt:
            return self._mock_researcher(user_prompt)
        elif "writer" in combined or "撰写" in user_prompt:
            return self._mock_writer(user_prompt)
        else:
            return self._mock_generic(user_prompt)

    def _mock_executor(self, prompt: str) -> str:
        return """## 数值模拟实现

以下为实现 Task 任务的 Python 代码：

```python
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ============================================================
# RGE 方程组数值求解
# ============================================================

def rge_system(t, y, params):
    '''
    MSSM 中微子 sector RGE 方程组
    t = ln(mu / mu0), y = [theta12, theta13, theta23, delta, dm21, dm31]
    '''
    theta12, theta13, theta23, delta, dm21, dm31 = y

    # Beta functions (简化示例 — 完整版见正式运行)
    d_theta12 = -0.01 * np.sin(2 * theta12) * np.exp(t)
    d_theta13 = -0.005 * np.sin(2 * theta13) * np.exp(t)
    d_theta23 = -0.008 * np.sin(2 * theta23) * np.exp(t)
    d_delta = 0.002 * np.sin(delta) * np.exp(t)
    d_dm21 = 0.001 * dm21 * np.exp(t)
    d_dm31 = 0.001 * dm31 * np.exp(t)

    return [d_theta12, d_theta13, d_theta23, d_delta, d_dm21, d_dm31]

# 初始条件 (GUT 能标 ~ 2e16 GeV, t=0)
# TM 混合模式
y0_tm = [
    np.radians(34.0),   # theta12
    np.radians(8.5),    # theta13
    np.radians(45.0),   # theta23
    np.radians(230.0),  # delta_CP
    7.5e-5,             # dm21 [eV^2]
    2.5e-3,             # dm31 [eV^2]
]

# RGE 跑动: t 从 0 (GUT) 到 ~30 (EW scale)
t_span = (0, 30)
t_eval = np.linspace(0, 30, 500)

sol = solve_ivp(
    rge_system, t_span, y0_tm,
    args=(None,), method='RK45',
    t_eval=t_eval, rtol=1e-8, atol=1e-10
)

if sol.success:
    print("RGE 积分成功完成")
    print(f"积分步数: {len(sol.t)}")

    # 提取结果
    theta12_final = np.degrees(sol.y[0, -1])
    theta13_final = np.degrees(sol.y[1, -1])
    theta23_final = np.degrees(sol.y[2, -1])
    delta_final = np.degrees(sol.y[3, -1])

    print(f"\\n最终混合角 (低能标):")
    print(f"  theta12 = {theta12_final:.2f}°")
    print(f"  theta13 = {theta13_final:.2f}°")
    print(f"  theta23 = {theta23_final:.2f}°")
    print(f"  delta_CP = {delta_final:.2f}°")

    # ---- 绘图 ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    energy = np.exp(sol.t) * 2e16  # 换算回 GeV

    axes[0, 0].plot(energy, np.degrees(sol.y[0]), 'b-', label=r'$\\theta_{12}$')
    axes[0, 0].set_xscale('log')
    axes[0, 0].set_xlabel('Energy Scale [GeV]')
    axes[0, 0].set_ylabel(r'$\\theta_{12}$ [deg]')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(energy, np.degrees(sol.y[1]), 'r-', label=r'$\\theta_{13}$')
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_xlabel('Energy Scale [GeV]')
    axes[0, 1].set_ylabel(r'$\\theta_{13}$ [deg]')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(energy, np.degrees(sol.y[2]), 'g-', label=r'$\\theta_{23}$')
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_xlabel('Energy Scale [GeV]')
    axes[1, 0].set_ylabel(r'$\\theta_{23}$ [deg]')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(energy, np.degrees(sol.y[3]), 'm-', label=r'$\\delta_{CP}$')
    axes[1, 1].set_xscale('log')
    axes[1, 1].set_xlabel('Energy Scale [GeV]')
    axes[1, 1].set_ylabel(r'$\\delta_{CP}$ [deg]')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('RGE Running of Neutrino Mixing Parameters — TM Pattern')
    plt.tight_layout()
    plt.savefig('sandbox/rge_running_tm.png', dpi=150)
    plt.close()
    print("\\n图表已保存: sandbox/rge_running_tm.png")
else:
    print(f"RGE 积分失败: {sol.message}")

print("\\nExecution complete.")
```

## 数值结果摘要

| 参数 | GUT 能标 | 低能标 | 变化 |
|------|----------|--------|------|
| θ₁₂ | 34.00° | 33.85° | -0.15° |
| θ₁₃ | 8.50° | 8.47° | -0.03° |
| θ₂₃ | 45.00° | 44.82° | -0.18° |
| δ_CP | 230.00° | 231.20° | +1.20° |

（注：以上为框架示例模拟输出，实际运行需完整 RGE beta 函数）
"""

    def _mock_researcher(self, prompt: str) -> str:
        return """## 文献摘要

### 中微子质量模型与 RGE 辐射稳定性研究现状

中微子振荡实验已确认中微子具有非零质量，为超出标准模型的新物理提供了确凿证据。
在众多中微子质量模型中，基于味对称性的混合模式（如 TM 和 μ−τ reflection）因其预言能力而受到广泛关注。

#### 关键文献脉络

1. **TM 混合模式** [1-3]：TM（Tri-bimaximal Mixing 变体）模式预言 θ₁₂ ≈ 35.3°，
   θ₂₃ = 45°，θ₁₃ = 0。θ₁₃ 非零的实验发现要求引入修正，当前研究聚焦于 TM₁ 和 TM₂ 变体。

2. **μ−τ reflection 对称性** [4-6]：该对称性预言 θ₂₃ = 45° 和 δ_CP = ±90° 或 270°，
   在轻子混合中具有特殊的唯象学意义，与 JUNO 对 θ₂₃ 和 δ_CP 的灵敏度目标直接相关。

3. **RGE 跑动效应** [7-9]：MSSM 框架下，中微子 Yukawa 耦合在大能标区域可显著修正混合角。
   特别是 tan β 较大时，辐射修正可导致低能混合角对高能边界条件产生 O(1°) 量级的偏离。

4. **JUNO 实验灵敏度** [10-11]：JUNO 预计将 θ₁₂ 测量精度提升至 0.3%，对 Δm²₂₁ 精度达 0.5%。
   这将为区分不同混合模式提供前所未有的约束。

5. **NuFIT 5.x 全局拟合** [12]：最新 NuFIT 结果给出了 3σ 范围内的混合参数，是本研究数值模拟的基准输入。

## 关键理论方程

### MSSM 中微子 RGE

中微子质量矩阵的 RGE 跑动由以下方程描述：

\\[
16\\pi^2 \\frac{d m_\\nu}{dt} = \\alpha \\, m_\\nu + P^T m_\\nu + m_\\nu P
\\]

其中 $t = \\ln(\\mu/\\mu_0)$，在 MSSM 中：

\\[
\\alpha = 2\\,\\text{Tr}(3Y_u^\\dagger Y_u + 3Y_d^\\dagger Y_d + Y_e^\\dagger Y_e) - \\frac{6}{5}g_1^2 - 6g_2^2
\\]

\\[
P = C_e Y_e^\\dagger Y_e
\\]

### PMNS 参数化

\\[
U_{\\text{PMNS}} = R_{23}(\\theta_{23}) \\cdot \\text{diag}(1, 1, e^{i\\delta}) \\cdot R_{13}(\\theta_{13}) \\cdot \\text{diag}(1, e^{i\\alpha_2/2}, e^{i\\alpha_3/2}) \\cdot R_{12}(\\theta_{12})
\\]

## 参数空间与实验约束

| 参数 | NuFIT 5.x best-fit | 3σ 范围 | 单位 |
|------|-------------------|---------|------|
| sin²θ₁₂ | 0.308 | 0.275–0.345 | — |
| sin²θ₁₃ | 0.0222 | 0.0205–0.0240 | — |
| sin²θ₂₃ (NO) | 0.550 | 0.435–0.610 | — |
| δ_CP / ° | 232 | 144–350 | deg |
| Δm²₂₁ | 7.49×10⁻⁵ | 6.92–8.05×10⁻⁵ | eV² |
| Δm²₃₁ (NO) | 2.534×10⁻³ | 2.455–2.615×10⁻³ | eV² |

## 研究空白

1. JUNO 精度下 TM 与 μ−τ reflection 模式辐射稳定性的系统比较尚缺乏
2. 增强工具（PyTorch 替代模型、PySR 符号回归）在 RGE 分析中的应用未经探索
3. 高能标边界条件对低能预测的误差传播需定量评估

### 参考文献

[1] Harrison, Perkins, Scott, PLB 530 (2002) 167
[2] Xing, Phys. Rept. 854 (2020) 1
[3] Ding, King, Luhn, JHEP 06 (2024) 012
[4] Grimus, Lavoura, PLB 671 (2009) 456
[5] Mohapatra, Nishi, PRD 86 (2012) 073007
[6] Zhou, 1809.06460
[7] Antusch et al., JHEP 03 (2005) 024
[8] Luo, Xiao, Xing, PRD 96 (2017) 075035
[9] Gehrlein et al., EPJC 84 (2024) 101
[10] JUNO Collaboration, JPG 43 (2016) 030401
[11] JUNO Collaboration, PPNP 123 (2022) 103927
[12] Esteban et al. (NuFIT 5.3), nu-fit.org
"""

    def _mock_planner(self, prompt: str) -> str:
        return """## 理论框架制定与模拟策略

### 1. 完整 RGE 方程组

在 MSSM 框架下，从 GUT 能标到电弱能标的 RGE 跑动由以下耦合微分方程组描述：

**规范耦合**：
\\[
\\frac{d g_i}{dt} = \\frac{b_i}{16\\pi^2} g_i^3, \\quad (i=1,2,3)
\\]
其中 MSSM 系数 $(b_1, b_2, b_3) = (33/5, 1, -3)$。

**中微子质量矩阵**：
\\[
16\\pi^2 \\frac{d m_\\nu}{dt} = (\\alpha + C_e y_\\tau^2)\\, m_\\nu + C_e y_\\tau^2 \\left[ P^T m_\\nu + m_\\nu P \\right]
\\]

其中 $P$ 投影片到 τ 方向，$C_e = 1$ (MSSM)，$y_\\tau$ 为 τ Yukawa 耦合。

**混合角 RGE**（从 PMNS 分解导出）：
\\[
\\frac{d\\theta_{ij}}{dt} = f_{ij}(\\theta_{12}, \\theta_{13}, \\theta_{23}, \\delta, \\alpha_i; y_\\tau, \\Delta m^2_{ij})
\\]

### 2. 初始条件定义

**TM 混合模式** (GUT 能标 $\\Lambda_{\\text{GUT}} = 2\\times 10^{16}$ GeV)：
\\[
\\theta_{12}^0 = 35.26^\\circ \\quad (\\sin^2\\theta_{12} = 1/3)
\\]
\\[
\\theta_{23}^0 = 45^\\circ, \\quad \\theta_{13}^0 = 8.5^\\circ \\text{ (变量，扫描范围 8.0°–9.0°)}
\\]
\\[
\\delta^0 = 230^\\circ \\text{ (变量，扫描范围 180°–360°)}
\\]

**μ−τ reflection 混合模式** (GUT 能标)：
\\[
\\theta_{23}^0 = 45^\\circ, \\quad \\delta^0 = 270^\\circ \\text{ (或 90°)}
\\]
\\[
|U_{\\mu i}| = |U_{\\tau i}| \\quad (i=1,2,3) \\quad \\text{(reflection 条件)}
\\]

**质量平方差** (GUT 能标)：
\\[
\\Delta m^2_{21} = 7.5\\times 10^{-5} \\text{ eV}^2, \\quad \\Delta m^2_{31} = 2.5\\times 10^{-3} \\text{ eV}^2
\\]

### 3. 数值积分方案

**方案选择**: 从 GUT 能标向下跑动（top-down）。理由：(a) 高能理论在高能标自然定义，(b) 跑动方向与物理直觉一致，(c) 可自然处理阈值修正。

**积分器**: `scipy.integrate.solve_ivp` with `method='RK45'`, `rtol=1e-8`, `atol=1e-10`.

**能标范围**: $t \\in [0, \\ln(\\Lambda_{\\text{GUT}}/M_Z)] \\approx [0, 33.1]$

### 4. 参数扫描策略

| 参数 | 扫描范围 | 步长 | 网格点数 |
|------|----------|------|----------|
| $\\theta_{13}(\\Lambda_{\\text{GUT}})$ | 8.0°–9.0° | 0.1° | 11 |
| $\\delta_{\\text{CP}}(\\Lambda_{\\text{GUT}})$ | 180°–360° | 10° | 19 |
| $\\tan\\beta$ | 10, 30, 50 | — | 3 |

总扫描点: 11 × 19 × 3 × 2 (两种模式) = 1254 个

### 5. 伪代码

```
算法: RGE 辐射稳定性分析

输入: 混合模式类型 (TM 或 μτ-reflection), 高能标参数
输出: 低能标混合参数, RGE 跑动曲线, 稳定性判别

1. 初始化 MSSM 参数:
   - GUT 能标: Λ_GUT = 2e16 GeV
   - EW 能标: M_Z = 91.2 GeV
   - tanβ ∈ {10, 30, 50}
   - Yukawa 耦合: y_t, y_b, y_τ

2. 设置高能标混合参数:
   IF 模式 == TM:
       θ₁₂ ← 35.26°, θ₂₃ ← 45°, θ₁₃ ← 扫描值, δ ← 扫描值
   ELIF 模式 == μτ-reflection:
       θ₂₃ ← 45°, δ ← 270°, |U_μi| = |U_τi|, θ₁₂, θ₁₃ ← 扫描值

3. 构建 RGE 方程组:
   FOR each parameter set:
       y_init ← [g₁, g₂, g₃, y_t, y_b, y_τ, m_ν 的独立分量]
       调用 solve_ivp(rge_rhs, t_span, y_init)

4. 提取低能标 PMNS 参数:
   U(Λ_EW) = 对角化 m_ν(Λ_EW)
   提取 θ₁₂, θ₁₃, θ₂₃, δ, Δm²₂₁, Δm²₃₁

5. 辐射稳定性判别:
   FOR each 扫描点:
       IF 所有低能参数 ∈ NuFIT 3σ 范围:
           标记为"稳定"
       ELSE:
           标记为"不稳定"
   计算: 稳定比例 = N_stable / N_total

6. 增强工具使用:
   - PyTorch: 训练替代模型 f_θ: (高能参数) → (低能参数)
     交叉验证: 5-fold, R² > 0.99
     ODE 回溯: 替代模型 vs 完整 ODE 的相对误差 < 0.1%
   - REAP: 独立交叉验证，不用于最终结果

返回: 稳定性分析结果, RGE 跑动曲线, 图集
```

### 6. 增强工具使用规划与合规性

| 工具 | 使用时机 | 合规方案 |
|------|----------|----------|
| REAP | Task 3 交叉验证 | ✅ 仅用作比对参考 |
| PyTorch | 可选: 加速参数扫描 | ✅ 5-fold CV + ODE 回溯验证 |
| PySR | 可选: 提取近似解析关系 | ✅ 残差分析 + 量纲约束 |
| FrEIA | 可选: 逆问题 | ✅ 正向 RGE 映射嵌入 |
"""

    def _mock_critic(self, prompt: str) -> str:
        return """【APPROVED】

## 审查报告

### 1. 代码正确性 ✅
- **语法检查**: 通过 — 代码语法正确，自包含
- **导入检查**: numpy, scipy.integrate, matplotlib — 均在允许列表中
- **NaN/Inf 检查**: 未检测到 NaN 或 Inf 值
- **退出码**: 0 (正常退出)
- **执行时间**: 在合理范围内

### 2. 物理正确性 ✅
- **RGE beta 函数**: 函数形式与 MSSM 理论预期一致
- **幺正性**: PMNS 矩阵条件数正常，行列式 |det U| = 1.000 ± 10⁻¹²
- **CP 相位效应**: δ_CP 随能标跑动方向与理论预期一致（向 270° 收敛趋势）
- **边界条件**: 高能初始条件与 Planner 制定方案一致

### 3. 增强工具合规性 ✅
- **REAP 比对**: 确认仅用于交叉验证，差异在 5% 以内，合理
- **PyTorch**: 本次未使用（可接受 — 标记为非必需）
- **PySR/FrEIA**: 本次未使用（可接受）

### 4. 图表质量 ✅
- 清晰的坐标轴标签和单位
- 合适的分辨率 (150 dpi)
- log 尺度正确应用

### 5. 数值结果合理性 ✅
- 辐射修正幅度 O(0.1°–0.2°) 符合 MSSM 跑动预期
- τ Yukawa 耦合主导的修正模式正确（θ₂₃ 修正 > θ₁₂ 修正）

### 综合评价
代码正确，物理结果合理，推荐通过。当前为模拟示例代码 — 实际运行需替换为完整 MSSM RGE beta 函数。
"""

    def _mock_writer(self, prompt: str) -> str:
        return """## 现象学分析与物理讨论

### RGE 辐射稳定性分析

基于前序数值模拟结果，我们对 TM 与 μ−τ reflection 两种混合模式的
RGE 辐射稳定性进行了系统比较。以下讨论基于从 GUT 能标（$\\Lambda_{\\text{GUT}} \\approx 2\\times 10^{16}$ GeV）
到电弱能标（$M_Z = 91.2$ GeV）的完整 RGE 跑动。

#### 1. TM 混合模式的辐射稳定性

TM 混合模式的核心特征是 $\\theta_{12}^0 \\approx 35.3^\\circ$ 和 $\\theta_{23}^0 = 45^\\circ$。
数值结果表明，在 MSSM 框架下：

- **混合角跑动幅度**: $\\theta_{12}$ 从 $35.3^\\circ$ 降至约 $33.8^\\circ$（$\\Delta \\approx -1.5^\\circ$），
  在高 $\\tan\\beta$（≥30）区域，跑动幅度增大。这与 τ Yukawa 耦合主导的 RGE 修正机制一致：
  $\\theta_{12}$ 的 RGE 正比于 $y_\\tau^2 \\Delta m^2_{21} / \\Delta m^2_{31}$，在 MSSM 中为负贡献。

- **CP 相位稳定性**: $\\delta_{\\text{CP}}$ 表现出中度跑动（$\\Delta\\delta \\approx 1^\\circ$–$5^\\circ$），
  方向取决于初始值。当 $\\delta^0 \\approx 230^\\circ$ 时，低能 $\\delta$ 与 NuFIT 最佳拟合值
  $232^\\circ$ 高度吻合。

- **JUNO 兼容性**: 约 68% 的 TM 参数空间在 JUNO 3σ 范围内保持稳定。
  $\\theta_{12}$ 的跑动幅度（$-1.5^\\circ$）对 JUNO 预期的 0.3% 精度不构成显著威胁，
  但对 $\\sin^2\\theta_{12}$ 在 NuFIT 3σ 边界附近的参数点需谨慎对待。

#### 2. μ−τ Reflection 模式的辐射稳定性

μ−τ reflection 对称性的定义特征是 $\\theta_{23}^0 = 45^\\circ$ 和 $\\delta_{\\text{CP}}^0 = 270^\\circ$
（或 $90^\\circ$），且关联条件 $|U_{\\mu i}| = |U_{\\tau i}|$ 在高能标成立。

- **对称性破缺**: RGE 跑动在低能标产生可观测的 μ−τ 对称性破缺。
  $\\theta_{23}$ 偏离 $45^\\circ$ 约 $\\Delta\\theta_{23} \\approx -0.2^\\circ$ 至 $-1.0^\\circ$
  （取决于 $\\tan\\beta$），偏离方向朝向 Normal Ordering 的 NuFIT 偏好值。

- **JUNO 灵敏度**: JUNO 对 $\\theta_{23}$ 的灵敏度预计可达 $\\Delta\\theta_{23} \\sim 0.5^\\circ$（90% CL）。
  当 $\\tan\\beta \\geq 30$ 时，μ−τ reflection 的辐射破缺可超过这一阈值，
  使 JUNO 有能力在 $\\sim 2\\sigma$ 水平区分严格的 μ−τ reflection 对称性
  与其辐射修正后的变体。

#### 3. 两种模式的比较

| 特征 | TM 混合 | μ−τ reflection | 物理根源 |
|------|---------|----------------|----------|
| $\\theta_{12}$ 跑动 | 中等（$\\sim -1.5^\\circ$） | 较小（$\\sim -0.5^\\circ$） | TM 的 $\\theta_{12}$ 更高 |
| $\\theta_{23}$ 跑动 | 较小（$\\sim -0.2^\\circ$） | 中等（$\\sim -0.5^\\circ$） | reflection 对称性破缺 |
| $\\delta_{\\text{CP}}$ 跑动 | 中度（$\\sim 1^\\circ$–$5^\\circ$） | 较小（$\\sim 1^\\circ$–$2^\\circ$） | $\\delta = 270^\\circ$ 有部分 RGE 稳定性保护 |
| 整体稳定性评分 | ★★★☆☆ | ★★★★☆ | μ−τ reflection 在辐射修正下更稳定 |

#### 4. 物理结论

1. **μ−τ reflection 模式在 RGE 辐射修正下表现出更强的整体稳定性**，
   尤其在高 $\\tan\\beta$ 区域，其 CP 相位受到部分 RGE 保护。

2. TM 混合模式的 $\\theta_{12}$ 跑动幅度（$\\sim -1.5^\\circ$）虽然显著，
   但仍在 JUNO 灵敏度的可分辨范围内，为该模式提供了可检验的唯象学特征。

3. JUNO 对 $\\theta_{23}$ 和 $\\sin^2\\theta_{12}$ 的联合测量将构成对
   μ−τ reflection 对称性强有力的检验：若 $\\theta_{23} \\neq 45^\\circ$
   和/或 $\\delta_{\\text{CP}} \\neq 270^\\circ$ 在 JUNO 的高精度下得到确认，
   可区分"高能严格 reflection + RGE 修正"与"代际对称性破缺"两种物理图景。

4. **主要结论**: 在 JUNO 首批数据的背景下，μ−τ reflection 模式因其更强的辐射稳定性
   和更少的自由参数而略受偏好。但最终判决需要 JUNO $\\theta_{23}$ 和
   $\\delta_{\\text{CP}}$ 的第一性原理测量结果。

#### 5. 展望

- 结合 DUNE/HK 对 $\\delta_{\\text{CP}}$ 和 $\\theta_{23}$ 的互补灵敏度
- 扩展至 Type-II 跷跷板、线性跷跷板等其他中微子质量产生机制
- 利用 PySR 从数值结果中提取 RGE 修正的近似解析公式
"""

    def _mock_generic(self, prompt: str) -> str:
        return f"""## AI4Sci Harness — Generic Response

This is a mock response from the MockBackend.
No specific agent type was detected in the prompt.

Prompt length: {len(prompt)} characters.
System prompt and user prompt received.
"""
