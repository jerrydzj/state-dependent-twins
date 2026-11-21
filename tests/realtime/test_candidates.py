from sdt.realtime.candidates import Settings, table


def test_the_plan_s_92_candidates_in_its_order():
    t = table()
    assert len(t) == 92 and t.candidate.is_unique and t.order.tolist() == list(range(92))
    assert (t.cls == 'twin').sum() == 32 and (t.cls == 'simple').sum() == 60
    assert t.groupby('model', sort=False).size().to_dict() == {'shift': 16, 'scale': 16, 'smooth': 10, 'ridge': 12,
                                                              'gb': 12, 'two-stage': 26}
    assert t.candidate.tolist()[:5] == ['shift/D/k1', 'shift/D/k3', 'shift/D/k6', 'shift/D/kall', 'shift/D+M/k1']
    assert t.candidate.tolist()[32:34] == ['smooth/-/a1.0', 'smooth/-/a0.9']
    assert t.candidate.tolist()[42:45] == ['ridge/D/hnone', 'ridge/D/h12', 'ridge/D/h36']


def test_two_stage_candidates_smooth_only_on_d_and_d_plus_p():
    t = table()
    two = t[t.model == 'two-stage'].groupby('input_set', sort=False).setting.apply(list).to_dict()
    smooth = [f'smooth-a{a:.1f}' for a in Settings().alphas]
    regressions = ['level-regression', 'anchored-change']
    assert two == {'D': smooth, 'D+M': regressions, 'D+P': smooth + regressions, 'D+M+P': regressions}
