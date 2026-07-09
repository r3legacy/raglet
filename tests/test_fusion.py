from raglet.fusion import reciprocal_rank_fusion


def test_fusion_combines_rankings():
    # Document 2 is ranked first in both lists, so it should win.
    rankings = [[2, 1, 3], [2, 1, 4]]
    fused = reciprocal_rank_fusion(rankings)
    assert fused[0][0] == 2
    ids = [doc_id for doc_id, _ in fused]
    assert ids == [2, 1, 3, 4]


def test_fusion_empty():
    assert reciprocal_rank_fusion([]) == []


def test_fusion_weighting():
    rankings = [[1, 2], [1, 3]]
    fused = dict(reciprocal_rank_fusion(rankings))
    # Document 1 is rank 0 in both rankings -> highest fused score.
    top = max(fused, key=fused.get)
    assert top == 1
    assert fused[1] > fused[2]
    assert fused[1] > fused[3]
