def test_hyp3_water_mask(script_runner):
    ret = script_runner.run('hyp3_water_mask', '-h')
    assert ret.success


def test_proc_water_mask(script_runner):
    ret = script_runner.run('proc_water_mask', '-h')
    assert ret.success
