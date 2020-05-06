def test_hyp3_water_mask(script_runner):
    ret = script_runner.run('hyp3_water_mask', '-h')
    assert ret.success
