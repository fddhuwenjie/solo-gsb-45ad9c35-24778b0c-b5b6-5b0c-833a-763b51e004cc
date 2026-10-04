"""并列解计数等价关系回归测试。

修复前：同一间隙段内“参考层出列(GA) / 待校核层出列(GB)”的不同先后次序
被当成不同方案计入 solution_count（1 层对 1 层、参数全 0 时返回 3）。
修复后：solution_count 与 solutions_listed 使用同一等价关系——
“匹配对集合 + 两侧未匹配层集合”，顺序重复在最优子图与多段组合中均被去除。

运行：python3 -m unittest discover -s tests -t . -v
"""
import unittest

from app.correlator import COUNT_CAP, MAX_SOLUTIONS, correlate, prepare_layers

ZERO_WEIGHTS = {"lithology": 0.0, "grain": 0.0, "water": 0.0,
                "elev": 0.0, "confidence": 0.0}
DEFAULT_WEIGHTS = {"lithology": 0.45, "grain": 0.15, "water": 0.10,
                   "elev": 0.20, "confidence": 0.10}


def make_layer(lid, seq, lith="细砂", top=None, bottom=None, conf=0.8):
    return {
        "id": lid, "seq": seq,
        "top": seq - 1 if top is None else top,
        "bottom": seq if bottom is None else bottom,
        "lithology": lith, "grain": "细粒", "water": "湿润",
        "confidence": conf,
    }


def run(ref_layers, hole_layers, anchors=(), weights=ZERO_WEIGHTS,
        gap_open=0.0, gap_extend=0.0, collar=100.0):
    ref = prepare_layers(ref_layers, collar)
    hole = prepare_layers(hole_layers, collar)
    return correlate(ref, hole, list(anchors), weights, 5.0, gap_open, gap_extend)


def signatures(res):
    """每个列出方案的等价类签名（匹配对集合 + 两侧未匹配层集合）。"""
    return {
        (tuple(sorted(s["pairs"])),
         tuple(sorted(s["unmatched_ref"])),
         tuple(sorted(s["unmatched_hole"])))
        for s in res["tie_solutions"]
    }


class TieCountEquivalenceTest(unittest.TestCase):
    """反例：五项权重与 gap_open/gap_extend 全为 0。"""

    def test_single_layer_counterexample(self):
        """参考孔与待校核孔各一层：只有“两层均匹配 / 两层均未匹配”2 个方案。"""
        res = run([make_layer(1, 1)], [make_layer(2, 1)])
        self.assertEqual(res["solution_count"], 2)
        self.assertEqual(res["solutions_listed"], 2)
        self.assertEqual(
            signatures(res),
            {(((1, 2),), (), ()), ((), (1,), (2,))},
        )
        # 歧义区间：该参考层既可匹配也可未匹配
        self.assertEqual(len(res["ambiguities"]), 1)
        amb = res["ambiguities"][0]
        self.assertEqual(amb["ref_layer_id"], 1)
        self.assertEqual(amb["outcome_count"], 2)
        self.assertTrue(amb["can_be_unmatched"])

    def test_gap_order_not_counted_2x2(self):
        """2 层对 2 层全并列：等价类数 = C(4,2) = 6，而非 Delannoy 数 13。"""
        res = run([make_layer(1, 1), make_layer(2, 2)],
                  [make_layer(3, 1), make_layer(4, 2)])
        self.assertEqual(res["solution_count"], 6)
        self.assertEqual(res["solutions_listed"], 6)
        self.assertEqual(len(signatures(res)), 6)

    def test_count_cap_and_listing_limit(self):
        """8 层对 8 层全并列：C(16,8)=12870，计数封顶 9999、最多列出 100 条。"""
        res = run([make_layer(100 + i, i + 1) for i in range(8)],
                  [make_layer(200 + i, i + 1) for i in range(8)])
        self.assertEqual(res["solution_count"], COUNT_CAP)
        self.assertEqual(res["solutions_listed"], MAX_SOLUTIONS)
        self.assertEqual(len(signatures(res)), MAX_SOLUTIONS)


class MultiSegmentCombinationTest(unittest.TestCase):
    """多标志层分段组合：段间笛卡尔积中同样去除顺序重复。"""

    def test_single_anchor_two_segments(self):
        """中间锁定 1 个锚点：两段各 1v1（各 2 个等价类），组合后恰 4 个。"""
        res = run([make_layer(11, 1), make_layer(12, 2), make_layer(13, 3)],
                  [make_layer(21, 1), make_layer(22, 2), make_layer(23, 3)],
                  anchors=[(1, 1)])
        self.assertEqual(res["solution_count"], 4)
        self.assertEqual(res["solutions_listed"], 4)
        # 锚点保留：主解中锚点对带 marker 标志，anchors 摘要不变
        self.assertEqual([(a["ref_layer_id"], a["hole_layer_id"])
                          for a in res["anchors"]], [(12, 22)])
        marked = {(p["ref_layer_id"], p["hole_layer_id"])
                  for p in res["pairs"] if p["marker"]}
        self.assertEqual(marked, {(12, 22)})
        # 4 个方案 = 前后两段“匹配/不匹配”的全部组合，锚点始终在列
        expect = set()
        for first in ((), ((11, 21),)):
            for last in ((), ((13, 23),)):
                pairs = tuple(sorted(first + ((12, 22),) + last))
                unref = tuple(sorted({11, 12, 13} - {p[0] for p in pairs}))
                unhole = tuple(sorted({21, 22, 23} - {p[1] for p in pairs}))
                expect.add((pairs, unref, unhole))
        self.assertEqual(signatures(res), expect)

    def test_two_anchors_three_segments(self):
        """两个锁定锚点切出三段 1v1：2×2×2 = 8 个等价类，顺序重复不计。"""
        res = run([make_layer(11, 1), make_layer(12, 2), make_layer(13, 3),
                   make_layer(14, 4), make_layer(15, 5)],
                  [make_layer(21, 1), make_layer(22, 2), make_layer(23, 3),
                   make_layer(24, 4), make_layer(25, 5)],
                  anchors=[(1, 1), (3, 3)])
        self.assertEqual(res["solution_count"], 8)
        self.assertEqual(res["solutions_listed"], 8)
        self.assertEqual(len(signatures(res)), 8)
        # 每个列出方案都必须包含两个锚点对
        for sig in signatures(res):
            self.assertIn((12, 22), sig[0])
            self.assertIn((14, 24), sig[0])

    def test_adjacent_anchors_empty_segments(self):
        """锚点相邻产生空段：只有 1 个方案，锚点不被穿越。"""
        res = run([make_layer(1, 1), make_layer(2, 2)],
                  [make_layer(3, 1), make_layer(4, 2)],
                  anchors=[(0, 0), (1, 1)])
        self.assertEqual(res["solution_count"], 1)
        self.assertEqual(res["solutions_listed"], 1)
        self.assertTrue(all(p["marker"] for p in res["pairs"]))


class RegressionUnchangedTest(unittest.TestCase):
    """常规参数下的既有行为保持不变。"""

    def test_unique_optimum_default_params(self):
        """默认权重 + 仿射间隙：唯一最优时计数与列出均为 1，总分不变。"""
        res = run([make_layer(1, 1, "细砂"), make_layer(2, 2, "黏土"),
                   make_layer(3, 3, "砾石")],
                  [make_layer(4, 1, "细砂"), make_layer(5, 2, "黏土"),
                   make_layer(6, 3, "砾石")],
                  weights=DEFAULT_WEIGHTS, gap_open=2.0, gap_extend=1.0)
        self.assertEqual(res["solution_count"], 1)
        self.assertEqual(res["solutions_listed"], 1)
        self.assertEqual([(p["ref_seq"], p["hole_seq"]) for p in res["pairs"]],
                         [(1, 1), (2, 2), (3, 3)])
        self.assertEqual(res["gap_count"], 0)
        self.assertGreater(res["total_score"], 90.0)

    def test_tie_with_real_gap_penalties(self):
        """非零间隙惩罚下的并列：两个高程对称的候选层，2 个等价类。"""
        ref = [make_layer(1, 1, top=0.0, bottom=2.0)]            # zmid = 99
        hole = [make_layer(2, 1, top=0.0, bottom=7.0),
                make_layer(3, 2, top=7.0, bottom=12.0)]
        # 待校核孔孔口 105.5：两层中高程 102 / 96，与参考层 99 等距对称
        res = correlate(prepare_layers(ref, 100.0),
                        prepare_layers(hole, 105.5),
                        [], DEFAULT_WEIGHTS, 5.0, 2.0, 1.0)
        self.assertEqual(res["solution_count"], 2)
        self.assertEqual(res["solutions_listed"], 2)
        self.assertEqual(
            signatures(res),
            {(((1, 2),), (), (3,)), (((1, 3),), (), (2,))},
        )
        self.assertEqual(len(res["ambiguities"]), 1)
        self.assertFalse(res["ambiguities"][0]["can_be_unmatched"])

    def test_listed_never_exceeds_count_or_limit(self):
        """不变量：solutions_listed ≤ min(solution_count, 100)，签名互异。"""
        cases = [
            run([make_layer(1, 1)], [make_layer(2, 1)]),
            run([make_layer(1, 1), make_layer(2, 2)],
                [make_layer(3, 1), make_layer(4, 2)]),
            run([make_layer(1, 1), make_layer(2, 2), make_layer(3, 3)],
                [make_layer(4, 1), make_layer(5, 2), make_layer(6, 3)],
                anchors=[(1, 1)]),
        ]
        for res in cases:
            self.assertLessEqual(res["solutions_listed"],
                                 min(res["solution_count"], MAX_SOLUTIONS))
            self.assertEqual(len(signatures(res)), res["solutions_listed"])


if __name__ == "__main__":
    unittest.main()
