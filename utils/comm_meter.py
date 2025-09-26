from collections import defaultdict
import json, os


class CommMeter:
    """
    按方向（uplink/downlink）累加实际字节与baseline字节，并可按mode（dense/svd/aad/bkd/bkd_aad/…）细分。
    仅在训练结束时输出汇总。
    """

    def __init__(self, log_dir, baseline='dense_delta'):
        self.log_dir = log_dir
        self.baseline = baseline
        self.reset()

    def reset(self):
        self.rounds = 0
        self.stats = {
            'uplink':   {'actual': 0, 'baseline': 0},
            'downlink': {'actual': 0, 'baseline': 0},
            'by_mode': defaultdict(lambda: {
                'uplink': 0, 'uplink_base': 0, 'downlink': 0, 'downlink_base': 0
            })
        }

    def add(self, direction, actual_bytes, baseline_bytes, mode='unknown'):
        actual_bytes = int(actual_bytes)
        baseline_bytes = int(baseline_bytes)
        self.stats[direction]['actual'] += actual_bytes
        self.stats[direction]['baseline'] += baseline_bytes
        m = self.stats['by_mode'][mode]
        if direction == 'uplink':
            m['uplink'] += actual_bytes
            m['uplink_base'] += baseline_bytes
        else:
            m['downlink'] += actual_bytes
            m['downlink_base'] += baseline_bytes

    def tick_round(self):  # 仅计数，不打印
        self.rounds += 1

    @staticmethod
    def _ratio(base, act):
        return (base / act) if act > 0 else float('inf')

    @staticmethod
    def _savings(base, act):
        return (1.0 - (act / max(base, 1))) if base > 0 else 0.0

    def summary(self):
        up, dn = self.stats['uplink'], self.stats['downlink']
        total_actual = up['actual'] + dn['actual']
        total_baseline = up['baseline'] + dn['baseline']
        result = {
            'baseline': self.baseline,
            'rounds': self.rounds,
            'uplink_MB': up['actual'] / 1e6,
            'uplink_baseline_MB': up['baseline'] / 1e6,
            'uplink_cr': self._ratio(up['baseline'], up['actual']),
            'uplink_savings': self._savings(up['baseline'], up['actual']),
            'downlink_MB': dn['actual'] / 1e6,
            'downlink_baseline_MB': dn['baseline'] / 1e6,
            'downlink_cr': self._ratio(dn['baseline'], dn['actual']),
            'downlink_savings': self._savings(dn['baseline'], dn['actual']),
            'total_MB': total_actual / 1e6,
            'total_baseline_MB': total_baseline / 1e6,
            'total_cr': self._ratio(total_baseline, total_actual),
            'total_savings': self._savings(total_baseline, total_actual),
            'by_mode': {}
        }
        for mode, v in self.stats['by_mode'].items():
            result['by_mode'][mode] = {
                'uplink_MB': v['uplink'] / 1e6,
                'uplink_baseline_MB': v['uplink_base'] / 1e6,
                'uplink_cr': self._ratio(v['uplink_base'], v['uplink']),
                'downlink_MB': v['downlink'] / 1e6,
                'downlink_baseline_MB': v['downlink_base'] / 1e6,
                'downlink_cr': self._ratio(v['downlink_base'], v['downlink'])
            }
        return result

    def dump(self, logger=None, filename='comm_report.json'):
        os.makedirs(self.log_dir, exist_ok=True)
        path = os.path.join(self.log_dir, filename)
        with open(path, 'w') as f:
            json.dump(self.summary(), f, indent=2)
        msg = f'[CommMeter] saved final report to {path}'
        (logger.info(msg) if logger else print(msg))
