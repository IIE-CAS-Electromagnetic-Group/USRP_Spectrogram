import numpy as np
from gnuradio import gr
import time


# ======== 工具：RRC 滤波器 ========
def rrc_taps(beta=0.35, sps=10, span=6):
    N = span * sps
    t = np.arange(-N / 2, N / 2 + 1) / sps
    taps = np.zeros_like(t)
    for i, x in enumerate(t):
        if np.isclose(x, 0.0):
            taps[i] = 1.0 - beta + 4 * beta / np.pi
        elif np.isclose(abs(x), 1 / (4 * beta)):
            taps[i] = (beta / np.sqrt(2)) * (
                    (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) +
                    (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
            )
        else:
            num = np.sin(np.pi * x * (1 - beta)) + 4 * beta * x * np.cos(np.pi * x * (1 + beta))
            den = np.pi * x * (1 - (4 * beta * x) ** 2)
            taps[i] = num / den
    taps /= np.sqrt(np.sum(taps ** 2))
    return taps


# ======== 各调制函数 ========

def mod_2FSK(bits, freq, t, df=5e3):
    N = len(t)
    if N == 0:
        return np.zeros(0, np.complex64)
    dt = (t[1] - t[0]) if N > 1 else 0.0
    freq_inst = freq + (2 * bits.astype(np.float64) - 1) * df
    phase = 2 * np.pi * np.cumsum(freq_inst) * dt
    return np.exp(1j * phase).astype(np.complex64)


def mod_GFSK(freq, t, samp_rate, bt=0.35, sym_rate=1e5):
    N = len(t)
    if N == 0:
        return np.zeros(0, np.complex64)

    # 用实际采样率计算每符号采样点
    sps = max(2, int(round(samp_rate / float(sym_rate))))

    # 生成符号并上采样到 >= N
    n_sym = max(1, int(np.ceil(N / sps)))
    data = np.random.choice([-1, 1], n_sym)
    up = np.repeat(data, sps)
    if len(up) < N:
        up = np.pad(up, (0, N - len(up)), mode="edge")
    up = up[:N]  # 保证长度 = N

    # 高斯脉冲成形
    L = 5
    gauss = np.exp(-0.5 * (np.linspace(-2, 2, L) ** 2) / (bt ** 2))
    phase_dev = np.convolve(up, gauss, mode="same")

    # 双保险：长度对齐
    if len(phase_dev) > N:
        phase_dev = phase_dev[:N]
    elif len(phase_dev) < N:
        phase_dev = np.pad(phase_dev, (0, N - len(phase_dev)), mode="edge")

    phase = np.cumsum(phase_dev) * (np.pi / sps)
    sig = np.exp(1j * (2 * np.pi * freq * t + phase))
    return sig.astype(np.complex64)


def mod_16QAM_rrc(freq, t, samp_rate, sym_rate=4e5, beta=0.4):
    """16QAM + RRC 成形，去掉周期伪谱"""
    N = len(t)
    if N == 0:
        return np.zeros(0, np.complex64)
    sps = max(2, int(round(samp_rate / sym_rate)))
    n_sym = int(np.ceil(N / sps)) + 8

    bits = np.random.randint(0, 2, size=(n_sym, 4))
    mI = (2 * bits[:, 0] + bits[:, 1]) * 2 - 3
    mQ = (2 * bits[:, 2] + bits[:, 3]) * 2 - 3
    sym = (mI + 1j * mQ) / np.sqrt(10)
    sym *= np.exp(1j * 2 * np.pi * np.random.rand(n_sym))  # 打散周期

    up = np.zeros(n_sym * sps, dtype=np.complex128)
    up[::sps] = sym
    taps = rrc_taps(beta=beta, sps=sps, span=6)
    shaped = np.convolve(up, taps, mode="full")
    if len(shaped) < N:
        shaped = np.pad(shaped, (0, N - len(shaped)))
    baseband = shaped[:N]

    # 微弱相位噪声防固定纹理
    phi_noise = np.cumsum(np.random.randn(N) * 1e-3)
    baseband *= np.exp(1j * phi_noise)
    return (baseband * np.exp(1j * 2 * np.pi * freq * t)).astype(np.complex64)


# ======== 主 Block ========

class blk(gr.sync_block):
    """
    业务调度：
      - FH 2FSK：hop_period=2 ms，占空50%，f_offset∈{-1MHz..+1MHz}按200kHz栅格随机
      - 宽带 16QAM：sym_rate=400 kHz，固定 f_offset=+300 kHz，20 ms ON / 10 ms OFF
      - 周期信标 GFSK：sym_rate=100 kHz，f_offset=-150 kHz，每 50 ms 发 5 ms
    """

    def __init__(self, samp_rate=10e6):
        gr.sync_block.__init__(
            self,
            name="Simple Scheduler",
            in_sig=None,
            out_sig=[np.complex64],
        )
        self.samp_rate = float(samp_rate)

        # 时间基准
        self.t0 = time.time()

        # --- 业务参数 ---
        # FH
        self.fh_hop_ms = 2.0
        self.fh_duty = 0.5
        self.fh_grid = np.arange(-1_000_000, 1_000_000 + 1, 200_000, dtype=np.int64)
        self.fh_curr = 0.0
        self._fh_last_win = -1  # 上一次窗口编号

        # 宽带 16QAM
        self.wide_sym_rate = 4e5
        self.wide_offset = +300e3
        self.wide_period_ms = 30.0
        self.wide_on_ms = 20.0

        # 信标 GFSK
        self.bcn_sym_rate = 1e5
        self.bcn_offset = -150e3
        self.bcn_period_ms = 50.0
        self.bcn_on_ms = 5.0

        # 幅度权重
        self.w_fh, self.w_wide, self.w_bcn = 0.25, 2.0, 0.2

    # --- 内部工具 ---
    def _safe_off(self, f_off):
        """频偏限幅到奈奎斯特内"""
        limit = 0.45 * self.samp_rate
        if f_off > limit:
            return limit
        if f_off < -limit:
            return -limit
        return f_off

    def _apply_fade(self, sig, fade_ratio=0.01):
        """为信号首尾加渐入渐出窗，避免横向条纹"""
        N = len(sig)
        if N < 8:
            return sig
        edge = max(4, int(fade_ratio * N))
        win = np.ones(N)
        taper = 0.5 * (1 - np.cos(np.linspace(0, np.pi, edge)))
        win[:edge] = taper
        win[-edge:] = taper[::-1]
        return sig * win

    # --- 主工作函数 ---
    def work(self, input_items, output_items):
        out = output_items[0]
        N = len(out)
        if N == 0:
            return 0

        t = np.arange(N, dtype=np.float64) / self.samp_rate
        now = time.time()
        elap_ms = (now - self.t0) * 1000.0

        # ===== 1) FH 2FSK 调度 =====
        fh_win = int(elap_ms // self.fh_hop_ms)
        if fh_win != self._fh_last_win:
            self._fh_last_win = fh_win
            self.fh_curr = float(np.random.choice(self.fh_grid))
            self.fh_curr = self._safe_off(self.fh_curr)
        fh_phase_ms = elap_ms % self.fh_hop_ms
        fh_on = fh_phase_ms < (self.fh_hop_ms * self.fh_duty)

        sr_fh = 1e3
        sps_fh = max(1, int(self.samp_rate // sr_fh))
        n_fh = int(np.ceil(N / sps_fh))
        bits_fh = np.repeat(np.random.randint(0, 2, n_fh), sps_fh)[:N]
        sig_fh = mod_2FSK(bits_fh, self.fh_curr, t, df=5e3) if fh_on else np.zeros(N, np.complex64)
        if fh_on:
            sig_fh = self._apply_fade(sig_fh)

        # ===== 2) 宽带 16QAM 调度 =====
        wide_phase_ms = elap_ms % self.wide_period_ms
        wide_on = (wide_phase_ms < self.wide_on_ms)
        wide_off = self._safe_off(self.wide_offset)

        if wide_on:
            sig_wide = mod_16QAM_rrc(wide_off, t, self.samp_rate, sym_rate=self.wide_sym_rate, beta=0.5)
            sig_wide = self._apply_fade(sig_wide)
        else:
            sig_wide = np.zeros(N, np.complex64)

        # ===== 3) 周期信标 GFSK 调度 =====
        bcn_phase_ms = elap_ms % self.bcn_period_ms
        bcn_on = (bcn_phase_ms < self.bcn_on_ms)
        bcn_off = self._safe_off(self.bcn_offset)

        if bcn_on:
            sig_bcn = mod_GFSK(bcn_off, t, self.samp_rate, bt=0.2, sym_rate=self.bcn_sym_rate)
            sig_bcn = self._apply_fade(sig_bcn)
        else:
            sig_bcn = np.zeros(N, np.complex64)

        # ===== 相位随机化，避免固定干涉 =====
        rot = np.exp(1j * 2 * np.pi * np.random.rand(3))
        sig_fh *= rot[0]
        sig_wide *= rot[1]
        sig_bcn *= rot[2]

        # ===== 叠加与整体平滑 =====
        sig = self.w_fh * sig_fh + self.w_wide * sig_wide + self.w_bcn * sig_bcn

        # 整体渐入渐出，平滑块边界
        sig = self._apply_fade(sig, fade_ratio=0.02)

        # 幅度限幅防溢出
        peak = np.max(np.abs(sig))
        if peak > 1.0:
            sig = sig / peak * 0.9

        out[:] = sig.astype(np.complex64)
        return N