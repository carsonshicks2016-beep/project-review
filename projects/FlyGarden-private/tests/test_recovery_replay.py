from scripts.replay_recovery_interfaces import held_command

def test_neural_commands_only_act_after_their_window():
    rows=[{'candidate_motor':[.2,.8]},{'candidate_motor':[.7,.1]}]
    assert held_command(rows,0)[1]==[0.,0.]
    assert held_command(rows,.025)[1]==[.2,.8]
    assert held_command(rows,.05)[1]==[.7,.1]
