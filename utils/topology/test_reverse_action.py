from representation import *
import copy

before_state = AbstractState()
after_state = AbstractState()
after_state.Reide2(over_idx=0,under_idx=0,left=1,over_before_under=-1)
mid_state = copy.deepcopy(after_state)
after_state.cross(over_idx=0, under_idx=4, sign=-1)
action1={'move':'R2','over_idx':0,'under_idx':0,'left':1,'over_before_under':-1}
action2={'move':'cross','over_idx':0,'under_idx':4,'sign':-1}
reversed_action2=reverse_action(action2, mid_state, after_state)
print(reversed_action2)
reversed_action1=reverse_action(action1, before_state, mid_state)
print(reversed_action1)
rev_rev_action1=reverse_action(reversed_action1, mid_state, before_state)
print(rev_rev_action1)
rev_rev_action2=reverse_action(reversed_action2, after_state, mid_state)
print(rev_rev_action2)
after_state.undo_cross(head=reversed_action2['head'])
after_state.undo_Reide2(over_idx=reversed_action1['over_idx'],under_idx=reversed_action1['under_idx'])
assert(after_state==before_state)
