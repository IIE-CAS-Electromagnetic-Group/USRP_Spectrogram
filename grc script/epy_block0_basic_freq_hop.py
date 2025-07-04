'''
模拟基础跳频通信（单信号）
调制：GFSK（bt=0.2~0.35）或 2FSK（Δf 固定）。
采样率：samp_rate = 1e6
符号率：sym_rate = 100e3（samples/symbol=10）
跳频节拍：hop_period = 5 ms（可做 1/2/5/10ms 四档）
频点栅格：grid = 200 kHz；中心f0，范围±2 MHz → 21 个栅格。
频偏选择：每个 hop 从 {f0 + k*grid} 随机选（避免相邻重复）
时长：每段 2–5 s
功率：常数或轻微抖动±2 dB
标注（每 hop 一条）：{mode:'GFSK', hop_id, f_offset, bt/Δf, sym_rate, SNR, t_start, t_end}
实现要点：在 GFSK/2FSK 分支里每到 hop_period 随机更新 self.freq_carrier（±2 MHz 栅格）。
'''


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

    def work(self, input_items, output_items):
        out = output_items[0]
        N = len(out)
        t = np.arange(N) / self.samp_rate

        # --- 限制符号速率，插值平滑 ---
        symbol_rate = 1e3
        samples_per_symbol = max(1, int(self.samp_rate / symbol_rate))
        num_symbols = int(np.ceil(N / samples_per_symbol))
        bits = np.random.randint(0, 2, num_symbols)
        bits = np.repeat(bits, samples_per_symbol)[:N]

        elapsed = (time.time() - self.last_switch) * 1000  # 毫秒级控制

        # -------- 跳频逻辑 --------
        hop_period = 10.0  # 每次跳频周期（毫秒）
        hop_range = 2e6  # 跳频范围 ±2 MHz
        hop_step = 2e5  # 每个跳频步进 200 kHz

        if elapsed >= hop_period:
            # 超过跳频周期 → 选择新频点
            f_list = np.arange(-hop_range, hop_range + hop_step, hop_step)
            self.freq_carrier = float(np.random.choice(f_list))
            self.last_switch = time.time()
            print(f"[Hop] New carrier = {self.freq_carrier / 1e3:.1f} kHz")

        # -------- 发射信号 --------
        out[:] = mod_GFSK(self.freq_carrier, t, bt=0.1)
        return len(out)

        # -------- 调制调用，这里留下只是用来参考，我们干脆直接用里面的函数得了--------
        '''if self.state == "GFSK":
            out[:] = mod_GFSK(self.freq_carrier, t, bt=0.1)
        elif self.state == "QPSK":
            out[:] = mod_QPSK(bits, self.freq_carrier + 5e4, t)
        elif self.state == "16QAM":
            out[:] = mod_16QAM(bits, self.freq_carrier, t)
        elif self.state == "BPSK":
            out[:] = mod_BPSK(bits, self.freq_carrier, t)
        elif self.state == "2FSK":
            out[:] = mod_2FSK(bits, self.freq_carrier, t, df=2e3)
        elif self.state == "OFDM":
            out[:] = mod_OFDM(bits, self.freq_carrier, t, n_subcarriers=8, sub_spacing=2e3)
        elif self.state == "IDLE":
            out[:] = 0
        else:
            out[:] = 0

        return len(out)'''

