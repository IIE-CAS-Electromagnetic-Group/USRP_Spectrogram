import numpy as np
from gnuradio import gr
import time


# --- 各调制函数 ---
def mod_BPSK(bits, freq, t):
    return np.exp(1j * (2 * np.pi * freq * t + np.pi * bits))


def mod_QPSK(bits, freq, t):
    # 保证比特数为偶数
    n_sym = len(bits) // 2
    bits = bits[:n_sym * 2]
    idx = bits[0::2] * 2 + bits[1::2]
    phases = np.pi / 4 + np.pi / 2 * idx
    # 生成每个符号的复数值
    sym = np.exp(1j * phases)
    # 重复符号以匹配采样点数量
    out = np.repeat(sym, int(np.ceil(len(t) / len(sym))))[:len(t)]
    return out * np.exp(1j * 2 * np.pi * freq * t)


def mod_GFSK(freq, t, bt=0.35):
    N = len(t)
    data = np.random.choice([-1, 1], N)
    samples_per_bit = max(2, int(len(t) / N))
    gauss = np.exp(-0.5 * (np.linspace(-2, 2, samples_per_bit * 2) ** 2) / bt ** 2)
    phase_dev = np.convolve(data, gauss, mode="same")
    phase = np.cumsum(phase_dev) * (np.pi / samples_per_bit)
    return np.exp(1j * (2 * np.pi * freq * t + phase))


def mod_16QAM(bits, freq, t):
    # 4 bit 一个符号
    n_sym = len(bits) // 4
    bits = bits[:n_sym * 4]
    b = bits.reshape(-1, 4)
    mI = (2 * b[:, 0] + b[:, 1]) * 2 - 3
    mQ = (2 * b[:, 2] + b[:, 3]) * 2 - 3
    sym = (mI + 1j * mQ) / np.sqrt(10)
    out = np.repeat(sym, int(np.ceil(len(t) / len(sym))))[:len(t)]
    return out * np.exp(1j * 2 * np.pi * freq * t)


def mod_2FSK(bits, freq, t, df=5e3):
    """二进制频移键控（2-FSK）"""
    # bits=0 → freq - df, bits=1 → freq + df
    freq_inst = freq + (2 * bits - 1) * df
    phase = np.cumsum(2 * np.pi * freq_inst / len(freq_inst))
    return np.exp(1j * phase)


def mod_OFDM(bits, freq, t, n_subcarriers=8, sub_spacing=1e3):
    """简化版 OFDM：多载波正交叠加"""
    # 生成多个子载波信号叠加
    carriers = []
    for k in range(n_subcarriers):
        sub_freq = freq + (k - n_subcarriers / 2) * sub_spacing
        phase = 2 * np.pi * sub_freq * t + np.pi * bits[k % len(bits)]
        carriers.append(np.exp(1j * phase))
    sig = np.sum(carriers, axis=0) / n_subcarriers
    return sig


# -------- 主 Block --------
class blk(gr.sync_block):
    def __init__(self):
        gr.sync_block.__init__(self,
                               name="Simple Scheduler",
                               in_sig=None,
                               out_sig=[np.complex64]
                               )
        self.samp_rate = 1e6
        self.freq_carrier = 1e6
        self.last_switch = time.time()
        self.state = "GFSK"  # 当前模式

        self.last_switch = time.time()
        self.fh_freq = 0.0
        self.next_hop_t = time.time()  # 下一次跳频触发时刻

    def work(self, input_items, output_items):
        out = output_items[0]
        N = len(out)
        t = np.arange(N) / self.samp_rate
        now = time.time()

        # ---- 三路各自的符号率（决定带宽与清晰度）----
        # FH 2FSK（较窄）
        sr_fh = 1e3
        sps_fh = max(1, int(self.samp_rate / sr_fh))
        n_fh = int(np.ceil(N / sps_fh))
        bits_fh = np.repeat(np.random.randint(0, 2, n_fh), sps_fh)[:N]

        # 宽带 16QAM（更宽）
        sr_wide = 4e5
        sps_wide = max(1, int(self.samp_rate / sr_wide))
        n_wide = max(1, int(np.ceil(N / sps_wide)))
        bits_wide = np.repeat(np.random.randint(0, 2, n_wide), sps_wide)[:N]

        # 周期信标 GFSK（中等）
        sr_bcn = 1e5
        sps_bcn = max(1, int(self.samp_rate / sr_bcn))
        n_bcn = int(np.ceil(N / sps_bcn))
        bits_bcn = np.repeat(np.random.randint(0, 2, n_bcn), sps_bcn)[:N]

        # ---- 三路调度门控（毫秒计时，单线程显式逻辑）----
        # 1) FH 短突发：hop_period=2ms，占空 50%
        hop_ms = 2.0
        # 到点就挑新频点（±1 MHz 栅格随机）
        if now >= self.next_hop_t:
            grid = 1e6
            self.fh_freq = float(np.random.choice([-1, 0, 1])) * grid  # 可改为更细栅格
            self.next_hop_t = now + hop_ms / 1000.0
        # 50% 占空：在每个 2ms 窗口前 1ms 开、后 1ms 关
        fh_phase_ms = ((now - self.last_switch) * 1000.0) % hop_ms
        fh_on = (fh_phase_ms < hop_ms / 2.0)

        # 2) 宽带 16QAM：20ms ON / 10ms OFF（周期 30ms）
        wide_period = 30.0
        wide_phase = ((now - self.last_switch) * 1000.0) % wide_period
        wide_on = (wide_phase < 20.0)
        wide_freq = +3e5

        # 3) 周期信标 GFSK：每 50ms 发 5ms（周期 50ms）
        bcn_period = 50.0
        bcn_phase = ((now - self.last_switch) * 1000.0) % bcn_period
        bcn_on = (bcn_phase < 5.0)
        bcn_freq = -1.5e5

        # ---- 逐路生成（关路就输出 0）----
        sig_fh = mod_2FSK(bits_fh, self.fh_freq, t, df=5e3) if fh_on else 0.0
        sig_wide = mod_16QAM(bits_wide, wide_freq, t) if wide_on else 0.0
        sig_bcn = mod_GFSK(bcn_freq, t, bt=0.15) if bcn_on else 0.0

        # ---- 叠加与缩放（防止过载）----
        # 相对功率：FH -6 dB，宽带 0 dB，信标 -10 dB（自行调）
        w_fh, w_wide, w_bcn = 0.5, 1.0, 0.32
        sig_sum = w_fh * sig_fh + w_wide * sig_wide + w_bcn * sig_bcn

        # 归一化（防溢出）；也可不归一化但降低各路权重
        if np.any(sig_sum):
            peak = np.max(np.abs(sig_sum))
            if peak > 1.0:
                sig_sum = sig_sum / peak

        out[:] = sig_sum
        # 可选：100ms 对齐一个“框架起点”，便于标注
        if ((now - self.last_switch) * 1000.0) > 100.0:
            self.last_switch = now
        return len(out)



