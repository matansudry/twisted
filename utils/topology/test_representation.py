from representation import AbstractState

def is_valid_state(state):
    # point over/under form pairs
    for i in range(1,state.pts+1):
        other_idx = state.points[i].over or state.points[i].under
        if other_idx is None:
            print('mid point does not have intersection!')
            return False
        if state.points[i].over is not None:
            self_idx = state.points[other_idx].under
        else:
            self_idx = state.points[other_idx].over
        if self.idx is None or self.idx != i:
            print('point intersection does not match!')
            return False
        if state.points[i].sign != state.points[other_idx].sign:
            print('intersecting points have different sign!')
            return False
    # edge next / prev form pairs
    for i, e in enumerate(state.edges):
        if e.next is None or e.prev is None:
            print('edge missing next / prev field!')
            return False
        if state.edges[e.next].prev != i or state.edges[e.prev].next != i:
            print('edge next/prev does not match!')
            return False
    # face point to correct edges, and uses edge exactly once.
    visited = set([])
    for i,f in enumerate(state.faces):
        next_edge = state.edges[f.edge]
        while next_edge not in visited:
            if next_edge.face != i:
                print('face pointer points to wrong edge!')
                return False
            visited.add(next_edge)
            next_edge = state.edges[next_edge.next]
    if len(visited) != len(state.edges):
        print('traversing all faces does not cover all edges!')
        return False
    return True

state = AbstractState()
#state.Reide1(idx=0, sign=1, left=1)
#print(state)
#state.Reide1(idx=2, sign=1, left=-1)
#print(state)

#state.Reide2(over_idx=0, under_idx=0, left=True)
#print(state)
#state.undo_Reide2(over_idx=1, under_idx=3)
#print(state)
#assert(is_valid_state(state))

#state.Reide1(idx=0, sign=1, left=1)
#print(state)
#state.cross(over_idx=0, under_idx=2, sign=1)
#print(state)
#state.undo_cross(head=False)
#print(state)
#state.undo_cross(head=False)
#print(state)

#state.Reide2(over_idx=0, under_idx=0, left=True)
#print(state)
#state.undo_cross(head=False)
#print(state)
#state.undo_cross(head=True)
#print(state)

state.Reide1(idx=0, sign=1, left=1)
print(state)
state.cross(over_idx=2, under_idx=0, sign=1)
print(state)

state = AbstractState()
state.Reide1(idx=0, sign=1, left=1)
print(state)
state.cross(over_idx=2, under_idx=0, sign=-1)
print(state)

