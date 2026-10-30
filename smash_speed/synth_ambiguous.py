"""種類の範囲を重ねた「あいまいな」合成ラリー（ショットの種類の判定の境目での間違い方を見る）。

``synth_rally.SHOT_PROFILES`` は種類ごとの範囲がほとんど重ならず、B4 のしきい値（``shot_types.RuleThresholds``）も
その範囲から決めたので、合成データでの判定の正解率は 0.98〜1.00 と高く出すぎる。実際には次のものが重なる。

- 速いドロップ（カット）と遅いスマッシュ（ハーフスマッシュ）、平らに速く飛ぶクリア（アタックロブ）と高いドリブン
- 高めのドライブと低いロブ、深く押し込むプッシュとドライブ
- 頭より少し低い打点のドロップ・クリアと、打点の高いプッシュ・ドライブ
- 少し深く落ちるネットと短いプッシュ

``AMBIGUOUS_PROFILES`` は、範囲をこの向きに少し広げた設定（サーブは同じ。判定はラリーの最初かどうかだけで決まるため）。
既定の ``SHOT_PROFILES`` とその合成データは変えず、``use_profiles`` の間だけ差し替えて作る。

``python -m smash_speed.synth_ambiguous --bench`` で、既定・あいまいな範囲のそれぞれで混同行列を出す。
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from typing import Any, Iterator

from smash_speed import synth_rally
from smash_speed.synth_rally import SHOT_PROFILES, ShotProfile

_P = SHOT_PROFILES
AMBIGUOUS_PROFILES: dict[str, ShotProfile] = {
    "serve_short": _P["serve_short"],
    "serve_long": _P["serve_long"],
    # 遅いスマッシュ（30 m/s〜）、ほぼ水平に打つもの、打点の少し低いもの
    "smash": replace(_P["smash"], speed_mps=(30.0, 110.0), elevation_deg=(-35.0, 0.0), hit_z_m=(2.0, 3.1),
                     hit_depth_m=(1.5, 6.3)),
    # 平らに速く飛ぶクリア（打ち出し角 8° から）、打点の低いもの
    "clear": replace(_P["clear"], elevation_deg=(8.0, 50.0), hit_z_m=(1.8, 3.0), hit_depth_m=(3.0, 6.6),
                     land_depth_m=(4.3, 6.6)),
    # 速いドロップ（カット）、打点の低いもの、少し深く落ちるもの
    "drop": replace(_P["drop"], speed_mps=(5.0, 50.0), elevation_deg=(-20.0, 15.0), hit_z_m=(1.8, 3.0),
                    land_depth_m=(0.5, 3.5)),
    # 高めに打ち上げる・遅いドライブ、打点の高いもの
    "drive": replace(_P["drive"], speed_mps=(15.0, 45.0), elevation_deg=(-10.0, 25.0), hit_z_m=(0.8, 2.1),
                     land_depth_m=(2.5, 6.6), max_apex_m=3.3),
    # 低いロブ（打ち出し角 22° から）、少し浅く落ちるもの
    "lob": replace(_P["lob"], elevation_deg=(22.0, 65.0), hit_z_m=(0.2, 1.5), land_depth_m=(4.0, 6.6)),
    # 少し深く・高く上がるネット
    "net": replace(_P["net"], hit_z_m=(0.2, 1.6), land_depth_m=(0.2, 2.0), max_apex_m=2.5),
    # 打点の高い・深く押し込むプッシュ、少し速いもの
    "push": replace(_P["push"], speed_mps=(6.0, 26.0), hit_z_m=(1.0, 2.2), land_depth_m=(1.5, 4.8),
                    max_apex_m=2.6),
}


@contextmanager
def use_profiles(profiles: dict[str, ShotProfile]) -> Iterator[None]:
    """この間だけ ``synth_rally.SHOT_PROFILES`` の中身を ``profiles`` にする（合成ラリーを作る関数がそれを使う）。

    キー（serve_short・smash など）と種類は同じでなければならない。終わったら元に戻す（同じプロセスの中だけ）。
    """
    cur = synth_rally.SHOT_PROFILES
    if set(profiles) != set(cur) or any(profiles[k].shot_type != cur[k].shot_type for k in cur):
        raise ValueError("キーと種類が SHOT_PROFILES と同じ設定にしてください")
    saved = dict(cur)
    cur.update(profiles)
    try:
        yield
    finally:
        cur.update(saved)


def ambiguous_benchmark(mode: str = "truth", n_rallies: int = 100, fps: float = 30.0, seed: int = 0,
                        profiles: dict[str, ShotProfile] | None = None, **kwargs: Any) -> dict[str, Any]:
    """あいまいな範囲（``profiles``。省略時は ``AMBIGUOUS_PROFILES``）の合成ラリーで、種類の判定の混同行列を出す。

    Args:
        mode: "truth" は正解の軌道から判定する（``shot_types.truth_benchmark``。ルールそのものの限界）。
            "fit" は横の合成ラリーを観測し、打球を見つけて当てはめた軌道から判定する（``shot_types.synthetic_benchmark``）

    Returns:
        ``matrix``（混同行列）、``scores``、``rows``（``shot_types`` と同じ）
    """
    from smash_speed.shot_types import synthetic_benchmark, truth_benchmark

    with use_profiles(AMBIGUOUS_PROFILES if profiles is None else profiles):
        if mode == "truth":
            return truth_benchmark(n_rallies, seed)
        return synthetic_benchmark(n_rallies, fps, seed, **kwargs)


def main(argv: list[str] | None = None) -> None:
    import argparse
    import time

    from smash_speed.shot_types import format_matrix, mistakes
    from smash_speed.shots import SHOT_TYPE_LABELS

    ap = argparse.ArgumentParser(description="種類の範囲を重ねた合成ラリーで、種類の判定の混同行列を出す")
    ap.add_argument("--bench", action="store_true", help="合成ラリーで測る")
    ap.add_argument("--mode", nargs="+", default=["truth"], choices=["truth", "fit"])
    ap.add_argument("--rallies", type=int, default=100)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if not a.bench:
        ap.error("--bench を指定してください")
    for mode in a.mode:
        for name, prof in (("既定の範囲", SHOT_PROFILES), ("あいまいな範囲", AMBIGUOUS_PROFILES)):
            t0 = time.time()
            r = ambiguous_benchmark(mode, a.rallies, a.fps, a.seed, dict(prof))
            sc = r["scores"]
            print(f"{name}・{'正解の軌道' if mode == 'truth' else f'{a.fps:.0f} fps の当てはめ'}："
                  f"正解率 {sc['accuracy']:.3f}（{sc['n']} 本、{time.time() - t0:.0f} 秒）")
            print(format_matrix(r["matrix"]))
            print("多い間違い：" + "、".join(f"{SHOT_TYPE_LABELS[t]} → {SHOT_TYPE_LABELS[p]} {n} 本"
                                        for t, p, n in mistakes(r["matrix"])))


if __name__ == "__main__":
    import sys

    if (sys.stdout.encoding or "").lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    main()
