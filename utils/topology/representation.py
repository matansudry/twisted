# each intersecting point is a dictionary {'over', 'under', 'sign'},
# 'over' is the index of the intersecting point over it, if there is one,
# 'under' is the index of the intersecting point under it, if there is one,
# 'sign' is the sign of [over (cross) under (dot) z]
# Abstract rope state is represented by a list of intersecting points (in order).

# The 2D projection defined by the abstract rope state is also a directed graph.
# Inspired by the half edge representation (refer to CS268),
# each segment is also a pair of directed edge
# {'begin', 'end', 'face', 'next', 'prev'},
# where 'begin' and 'end' are indices of the points,
# 'face' is the index of face on the left.
# the other fields are indices of edges.
# for the open ends segments, the pair of edges point to the same face.
# edges are organized as a list. edge i->i+1 have index 2i, and edge i+1->i have index 2i+1

# each face (loop) is a dictionary {'idx', 'edge'}.
# where 'idx' is the index of the face, should always be in order.
# 'edge' is index of one of the edges.
# faces are also organized as a list (in order).

# open end segment is included twice in a face.
import pdb

class Point(object):
    def __init__(self, over=None, under=None, sign=None):
        self.over = over
        self.under = under
        self.sign = sign

    def __repr__(self):
        # defines printing behavior
        if self.over is None and self.under is None:
            return "End point\n"
        if self.over is not None:
            outstring = 'U %d ' % (self.over)
        else:
            outstring = 'O %d ' % (self.under)
        if self.sign == 1:
            outstring = outstring + '+\n'
        elif self.sign == -1:
            outstring = outstring + '-\n'
        return outstring


class Edge(object):
    def __init__(self, face=None, next=None, prev=None):
        self.face = face
        self.next = next
        self.prev = prev

    def __repr__(self):
        if self.face is not None:
            outstring = 'face %d, prev %d, next %d.\n' % (
                         self.face, self.prev, self.next)
            return outstring
        else:
            return 'Invalid edge\n'


class Face(object):
    def __init__(self, edge=None):
        self.edge = edge

    def __repr__(self):
        if self.edge is not None:
            return 'Face: e %d\n' % (self.edge)
        else:
            return 'empty face\n'


class AbstractState(object):
    def __init__(self):
        point1 = Point()
        point2 = Point()
        self.points = [point1, point2]
        edge1 = Edge(face=0, next=1, prev=1) # 0->1
        edge2 = Edge(face=0, next=0, prev=0) # 1->0
        self.edges = [edge1, edge2]
        self.faces = [Face(0)]

    def __repr__(self):
        if self.pts==0:
            return 'Trivial state\n'
        outstring = ''
        for i in range(1, self.pts+1):
            outstring = outstring + '%d: '%(i) + self.points[i].__repr__()
        outstring = outstring + '\n'
        visited = set([])
        for f in self.faces:
            next_edge = f.edge
            while next_edge not in visited:
                begin_p = (next_edge // 2) + (next_edge % 2)
                end_p = (next_edge // 2) + 1 - (next_edge % 2)
                outstring = outstring + '%d-%d/' % (begin_p, end_p)
                visited.add(next_edge)
                next_edge = self.edges[next_edge].next
            outstring = outstring + '\n'
        return outstring

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            raise NotImplementedError()
        if len(self.points)!=len(other.points):
            return False
        for p,po in zip(self.points, other.points):
            if p.over != po.over or p.under != po.under or p.sign != po.sign:
                return False
        for e,eo in zip(self.edges, other.edges):
            if e.next != eo.next or e.prev != eo.prev:
                return False
        return True

    def point_intersect(self, over_idx, under_idx, sign):
        # fill in point dictionaries for intersecting point pair
        # This function does not update edges or faces accordingly!
        self.points[over_idx].under = under_idx
        self.points[over_idx].sign = sign
        self.points[under_idx].over = over_idx
        self.points[under_idx].sign = sign

    def link_edges(self, idx1, idx2):
        # link idx1.next = idx2, idx2.prev = idx1
        self.edges[idx1].next = idx2
        self.edges[idx2].prev = idx1

    def reindex_face(self, start_edge_idx, new_face_idx):
        next_edge = self.edges[start_edge_idx]
        while next_edge.next != start_edge_idx:
            next_edge.face = new_face_idx
            next_edge = self.edges[next_edge.next]
        next_edge.face = new_face_idx

    @property
    def pts(self):
        return len(self.points)-2

    def addPoint(self, idx):
        # insert one point in self.points.
        # split the segment into two segments (creates two new edges)
        # update edge and face pointers to maintain the same valid graph state.
        if idx==0 or idx==len(self.points)+1:
            raise ValueError("invalid insert position")
        point = Point()
        prev_edge1 = self.edges[2*(idx-1)]
        prev_edge2 = self.edges[2*idx-1]
        new_edge1 = Edge(face=prev_edge1.face) # idx->idx+1
        new_edge2 = Edge(face=prev_edge2.face) # idx+1->idx
        for p in self.points:
            if p.over is not None and p.over >= idx:
                p.over += 1
            if p.under is not None and p.under >= idx:
                p.under += 1
        for e in self.edges:
            if e.next >= 2*idx:
                e.next += 2
            if e.prev >= 2*idx:
                e.prev += 2
        for f in self.faces:
            if f.edge >= 2*idx:
                f.edge += 2
        self.points.insert(idx, point)
        self.edges.insert(2*idx, new_edge1)
        self.edges.insert(2*idx+1, new_edge2)

        if prev_edge1.next == 2*idx-1: # inserted the last segment
            self.link_edges(2*idx, 2*idx+1)
        else:
            self.link_edges(2*idx, prev_edge1.next)
            self.link_edges(prev_edge2.prev, 2*idx+1)
        self.link_edges(2*(idx-1), 2*idx)
        self.link_edges(2*idx+1, 2*idx-1)


    def removePoint(self, idx):
        # test two neighboring segments have the same left and right faces.
        # merge two segments (remove two edges)
        # remove point in self.points.
        # update indices.
        if idx==0 or idx==len(self.points)+1:
            raise ValueError("invalid remove position")
        prev_edge1 = self.edges[2*(idx-1)]
        prev_edge2 = self.edges[2*idx-1]
        new_edge1 = self.edges[2*idx]
        new_edge2 = self.edges[2*idx+1]
        if prev_edge1.next != 2*idx or new_edge1.prev != 2*(idx-1):
            raise RuntimeError("Something is wrong with data structure")
        if prev_edge2.prev != 2*idx+1 or new_edge2.next != 2*idx-1:
            raise RuntimeError("Something is wrong with data structure")
        if prev_edge1.face != new_edge1.face or prev_edge2.face != new_edge2.face:
            raise RuntimeError("Something is wrong with data structure")
        self.link_edges(2*(idx-1), new_edge1.next)
        self.link_edges(new_edge2.prev, 2*idx-1)
        self.edges = self.edges[:2*idx]+self.edges[2*(idx+1):]
        self.points = self.points[:idx]+self.points[idx+1:]

        for p in self.points:
            if p.over is not None and p.over >= idx:
                p.over -= 1
            if p.under is not None and p.under >= idx:
                p.under -= 1
        for e in self.edges:
            if e.next >= 2*idx:
                e.next -= 2
            if e.prev >= 2*idx:
                e.prev -= 2
        for f in self.faces:
            if f.edge >= 2*idx:
                f.edge -= 2


    def cross(self, over_idx, under_idx, sign=1):
        # make a cross move on prev_state, where segment [over_idx, over_idx+1]
        # cross over segment [under_idx, under_idx+1]
        # at least one of the idx needs to be the first or last segment.
        if (over_idx > 0) and (under_idx > 0) and (over_idx < self.pts) and (under_idx < self.pts):
            #print("Invalid R1 move")
            return False

        if over_idx == under_idx:
            #print("Use R1 instead")
            return False

        # the two segments must share a loop for this move to be feasible
        prev_over_left = self.edges[2*over_idx].face
        prev_over_right = self.edges[2*over_idx+1].face
        prev_under_left = self.edges[2*under_idx].face
        prev_under_right = self.edges[2*under_idx+1].face
        if (prev_over_left != prev_under_left) and (prev_over_right != prev_under_left) and \
           (prev_over_left != prev_under_right) and (prev_over_right != prev_under_right):
            #print("cross is not possible without crossing another first.")
            return False

        if over_idx < under_idx:
            self.addPoint(over_idx+1)
            self.addPoint(under_idx+2)
            over_idx += 1
            under_idx += 2
        else:
            self.addPoint(under_idx+1)
            self.addPoint(over_idx+2)
            over_idx += 2
            under_idx += 1

        # sign is determined by over/under, and if the common loop is on left/right of the middle segment.
        if (over_idx in [1, self.pts]) and (under_idx in [1, self.pts]):
            sign = sign
            from_face = self.edges[0].face
            enter_face = from_face
            if (sign==1)==(over_idx==self.pts):
                from_face_edges = [2*(under_idx-1), 2*under_idx]
                enter_face_edges = [2*under_idx+1, 2*under_idx-1]
            else:
                enter_face_edges = [2*(under_idx-1), 2*under_idx]
                from_face_edges = [2*under_idx+1, 2*under_idx-1]
            end_idx = over_idx
        elif over_idx in [1, self.pts]:
            input_sign = sign
            sign = -1 if (prev_under_left==prev_over_left)==(over_idx==1) else 1
            if input_sign != sign:
                self.removePoint(max(over_idx,under_idx))
                self.removePoint(min(over_idx,under_idx))
                return False
            from_face = self.edges[2*over_idx].face
            if prev_over_left==prev_under_left:
                enter_face = self.edges[2*under_idx+1].face
                enter_face_edges = [2*under_idx+1, 2*under_idx-1]
                from_face_edges = [2*(under_idx-1), 2*under_idx]
            else:
                enter_face = self.edges[2*under_idx].face
                enter_face_edges = [2*(under_idx-1), 2*under_idx]
                from_face_edges = [2*under_idx+1, 2*under_idx-1]
            end_idx = over_idx
        elif under_idx in [1, self.pts]:
            input_sign = sign
            sign = 1 if (prev_under_left==prev_over_left)==(under_idx==1) else -1
            if input_sign != sign:
                self.removePoint(max(over_idx,under_idx))
                self.removePoint(min(over_idx,under_idx))
                return False
            from_face = self.edges[2*under_idx].face
            if prev_under_left==prev_over_left:
                enter_face = self.edges[2*over_idx+1].face
                enter_face_edges = [2*over_idx+1, 2*over_idx-1]
                from_face_edges = [2*(over_idx-1), 2*over_idx]
            else:
                enter_face = self.edges[2*over_idx].face
                enter_face_edges = [2*(over_idx-1), 2*over_idx]
                from_face_edges = [2*over_idx+1, 2*over_idx-1]
            end_idx = under_idx
        if end_idx == 1:
            end_edges = [3, 1, 0, 2]
        else:
            end_edges = [2*self.pts-2, 2*self.pts, 2*self.pts+1, 2*self.pts-1]

        self.point_intersect(over_idx, under_idx, sign)

        self.link_edges(enter_face_edges[0], end_edges[1])
        self.link_edges(end_edges[2], enter_face_edges[1])
        self.link_edges(from_face_edges[0], end_edges[3])
        self.link_edges(end_edges[0], from_face_edges[1])

        self.edges[end_edges[1]].face = enter_face
        self.edges[end_edges[2]].face = enter_face
        new_face_idx = len(self.faces)
        self.reindex_face(end_edges[0], new_face_idx)
        self.faces.append(Face(end_edges[0]))
        old_face_idx = self.edges[end_edges[3]].face
        if self.edges[self.faces[old_face_idx].edge].face != old_face_idx:
            self.faces[old_face_idx].edge=end_edges[3]
        return True


    def Reide1(self, idx, sign, left):
        # idx: do a R1 movement on segment [idx, idx+1].
        # sign: sign of the new intersection.
        # left: 1 of the new loop is on the left side of the original segment, else -1.
        self.addPoint(idx+1)
        self.addPoint(idx+2)
        if sign == left:
            self.point_intersect(idx+2, idx+1, sign)
        else:
            self.point_intersect(idx+1, idx+2, sign)

        if left==1:
            old_face_edges = [2*idx, 2*(idx+1)+1, 2*(idx+2), 2*(idx+2)+1, 2*idx+1]
            new_face_edge = 2*(idx+1)
        else:
            old_face_edges = [2*(idx+2)+1, 2*(idx+1), 2*idx+1, 2*idx, 2*(idx+2)]
            new_face_edge = 2*(idx+1)+1

        self.link_edges(old_face_edges[0], old_face_edges[1])
        self.link_edges(old_face_edges[1], old_face_edges[2])
        self.link_edges(old_face_edges[3], old_face_edges[4])
        self.link_edges(new_face_edge, new_face_edge)
        self.edges[old_face_edges[1]].face = self.edges[old_face_edges[0]].face
        self.edges[new_face_edge].face = len(self.faces)
        self.faces.append(Face(new_face_edge))

        return True

    def Reide2(self, over_idx, under_idx, left=1, over_before_under=1):
        # move segment [over_idx, over_idx+1] on top of segment [under_idx, under_idx+1]
        # to create two new intersections.
        # if over_idx==under_idx, move the first half over second half if over_before_under==1.

        # Both open ends must be in the same loop to define a R2 move.
        if self.edges[0].face != self.edges[-1].face:
            #print('Both open ends must be in the same loop for R2')
            return False
        # Add a virtual edge to connect open ends for ease of the process
        self.link_edges(2*self.pts, 0)
        self.link_edges(1, 2*self.pts+1)
        temp_face_idx = len(self.faces)+5
        self.reindex_face(0, temp_face_idx)

        def undo():
            self.reindex_face(0, self.edges[1].face)
            self.link_edges(1,0)
            self.link_edges(2*self.pts, 2*self.pts+1)

        # To be a valid move, the two segments must share a loop.
        # It is possible the two segments share both left and right loop.
        # In this case the left argument is True to cross from left of the over segment.
        faces = [self.edges[2*over_idx].face, self.edges[2*over_idx+1].face,
                 self.edges[2*under_idx].face, self.edges[2*under_idx+1].face]
        if len(set(faces)) == 4:
            #print('invalid R2 move')
            undo()
            return False
        if len(set(faces)) < 2:
            #print('something is wrong in state')
            undo()
            return False

        shared_face = None
        if faces[0]==faces[2] or faces[0]==faces[3]:
            shared_face = faces[0]
        if faces[1]==faces[2] or faces[1]==faces[3]:
            if shared_face is None or (left==-1):
                shared_face = faces[1]

        if shared_face == faces[0] and left==-1:
            undo()
            return False
        if shared_face == faces[1] and left==1:
            undo()
            return False

        if over_idx > under_idx and over_before_under == 1:
            undo()
            return False
        if over_idx < under_idx and over_before_under == -1:
            undo()
            return False

        if over_before_under==1:
            self.addPoint(over_idx+1)
            self.addPoint(over_idx+2)
            self.addPoint(under_idx+3)
            self.addPoint(under_idx+4)
            under_idx += 2
        else:
            self.addPoint(under_idx+1)
            self.addPoint(under_idx+2)
            self.addPoint(over_idx+3)
            self.addPoint(over_idx+4)
            over_idx += 2


        # which two points form a pair and sign of intersections
        # depend on the diretion of segment relative to the common loop.
        if shared_face==faces[0] and shared_face==faces[3]:
            self.point_intersect(over_idx+1, under_idx+1, -1)
            self.point_intersect(over_idx+2, under_idx+2, 1)
            f1_edges = [2*over_idx, 2*under_idx+1]
            f2_edges = [2*under_idx, 2*(over_idx+1), 2*(under_idx+2)]
            f3_edges = [2*(over_idx+2)+1, 2*(under_idx+1)+1, 2*(over_idx)+1]
            f4_edges = [2*(under_idx+2)+1, 2*(over_idx+2)]
            f5_edges = [2*(under_idx+1), 2*(over_idx+1)+1]
        elif shared_face==faces[0] and shared_face==faces[2]:
            self.point_intersect(over_idx+1, under_idx+2, 1)
            self.point_intersect(over_idx+2, under_idx+1, -1)
            f1_edges = [2*over_idx, 2*(under_idx+2)]
            f2_edges = [2*(under_idx+2)+1, 2*(over_idx+1), 2*(under_idx)+1]
            f3_edges = [2*(over_idx+2)+1, 2*(under_idx+1), 2*(over_idx)+1]
            f4_edges = [2*(under_idx), 2*(over_idx+2)]
            f5_edges = [2*(under_idx+1)+1, 2*(over_idx+1)+1]
        elif shared_face==faces[1] and shared_face==faces[2]:
            self.point_intersect(over_idx+1, under_idx+1, 1)
            self.point_intersect(over_idx+2, under_idx+2, -1)
            f1_edges = [2*under_idx, 2*over_idx+1]
            f2_edges = [2*(under_idx+2)+1, 2*(over_idx+1)+1, 2*(under_idx)+1]
            f3_edges = [2*(over_idx), 2*(under_idx+1), 2*(over_idx+2)]
            f4_edges = [2*(over_idx+2)+1, 2*(under_idx+2)]
            f5_edges = [2*(under_idx+1)+1, 2*(over_idx+1)]
        elif shared_face==faces[1] and shared_face==faces[3]:
            self.point_intersect(over_idx+1, under_idx+2, -1)
            self.point_intersect(over_idx+2, under_idx+1, 1)
            f1_edges = [2*(over_idx+2)+1, 2*(under_idx)+1]
            f2_edges = [2*(under_idx), 2*(over_idx+1)+1, 2*(under_idx+2)]
            f3_edges = [2*(over_idx), 2*(under_idx+1)+1, 2*(over_idx+2)]
            f4_edges = [2*(under_idx+2)+1, 2*over_idx+1]
            f5_edges = [2*(under_idx+1), 2*(over_idx+1)]
        else:
            pdb.set_trace()
        self.link_edges(f1_edges[0], f1_edges[1])
        self.link_edges(f2_edges[0], f2_edges[1])
        self.link_edges(f2_edges[1], f2_edges[2])
        self.link_edges(f3_edges[0], f3_edges[1])
        self.link_edges(f3_edges[1], f3_edges[2])
        self.link_edges(f4_edges[0], f4_edges[1])
        self.link_edges(f5_edges[0], f5_edges[1])
        self.link_edges(f5_edges[1], f5_edges[0])

        self.edges[f2_edges[1]].face = self.edges[f2_edges[0]].face
        self.edges[f3_edges[1]].face = self.edges[f3_edges[0]].face
        f4_idx = len(self.faces)
        self.reindex_face(f4_edges[0], f4_idx)
        self.faces.append(Face(f4_edges[0]))
        f5_idx = len(self.faces)
        self.reindex_face(f5_edges[0], f5_idx)
        self.faces.append(Face(f5_edges[0]))
        f1_idx = self.edges[f1_edges[0]].face
        if f1_idx == temp_face_idx:
            if self.edges[0].face != f1_idx:
                # switch f4 and f1
                self.reindex_face(f4_edges[0], f1_idx)
                self.reindex_face(f1_edges[0], f4_idx)
                self.faces[f4_idx].edge = f1_edges[0]
            if self.edges[0].face != f1_idx:
                raise RuntimeError("something is wrong with R2 switch face")
        else:
            if self.edges[self.faces[f1_idx].edge].face != f1_idx:
                self.faces[f1_idx].edge=f1_edges[0]
        # undo the virtual link between open ends.
        undo()
        return True


    def undo_cross(self, head):
        # if head=True, remove intersection point 1, otherwise remove prev_state.pts
        if self.pts == 0:
            return False
        if head:
            end_idx = 1
            mid_idx = self.points[1].over or self.points[1].under
            end_edges = [3, 1, 0, 2]
        else:
            end_idx = self.pts
            mid_idx = self.points[-2].over or self.points[-2].under
            end_edges = [2*self.pts-2, 2*self.pts, 2*self.pts+1, 2*self.pts-1]
        if end_idx == mid_idx+1 or end_idx == mid_idx-1:
            return False  # should call undo_Reide1
        f1_edges = [end_edges[0], self.edges[end_edges[0]].next]
        f2_edges = [self.edges[end_edges[1]].prev, end_edges[1], end_edges[2], self.edges[end_edges[2]].next]
        f3_edges = [self.edges[end_edges[3]].prev, end_edges[3]]

        f1_idx = self.edges[end_edges[0]].face
        f2_idx = self.edges[end_edges[1]].face
        f3_idx = self.edges[end_edges[3]].face

        self.link_edges(f2_edges[0], f2_edges[3])
        self.link_edges(f3_edges[0], f1_edges[1])
        self.link_edges(end_edges[0], end_edges[1])
        self.link_edges(end_edges[2], end_edges[3])

        if f1_idx < f3_idx:
            self.reindex_face(end_edges[1], f1_idx)
        else:
            self.reindex_face(f1_edges[1], f3_idx)
        if self.edges[self.faces[f2_idx].edge].face != f2_idx:
            self.faces[f2_idx].edge = f2_edges[0]

        remove_idx = max(f1_idx, f3_idx)
        self.faces = self.faces[:remove_idx]+self.faces[remove_idx+1:]
        for e in self.edges:
            if e.face >= remove_idx:
                e.face -= 1
        self.removePoint(max(end_idx, mid_idx))
        self.removePoint(min(end_idx, mid_idx))
        return True

    def undo_Reide1(self, idx):
        if idx == 0 or idx > self.pts:
            return False
        other_idx = self.points[idx].over or self.points[idx].under
        if other_idx != idx+1:
            #print("invalid undo R1 move")
            return False

        left = self.edges[2*(idx-1)].face
        right = self.edges[2*idx-1].face
        if self.edges[2*idx].face not in [left, right]:
            remove_face_idx = self.edges[2*idx].face
        else:
            remove_face_idx = self.edges[2*idx+1].face
        self.edges[2*idx].face = left
        self.edges[2*idx+1].face = right
        self.link_edges(2*(idx-1), 2*idx)
        self.link_edges(2*idx, 2*(idx+1))
        self.link_edges(2*(idx+1)+1, 2*idx+1)
        self.link_edges(2*idx+1, 2*idx-1)
        if self.edges[self.faces[left].edge].face != left:
            self.faces[left].edge=2*(idx-1)
        if self.edges[self.faces[right].edge].face != right:
            self.faces[right].edge=2*idx-1
        self.faces = self.faces[:remove_face_idx]+self.faces[remove_face_idx+1:]
        for e in self.edges:
            if e.face >= remove_face_idx:
                e.face -= 1

        self.removePoint(idx+1)
        self.removePoint(idx)
        return True

    def undo_Reide2(self, over_idx, under_idx):
        if (over_idx==0) or (over_idx >= self.pts) or (under_idx==0) or (under_idx >= self.pts):
            return False
        if (self.points[over_idx].under is None) or (self.points[over_idx+1].under is None):
            #print("invalid undo R2 move")
            return False
        if (self.points[over_idx].under != under_idx) and (self.points[over_idx].under != under_idx+1):
            #print("invalid undo R2 move")
            return False
        if (self.points[over_idx+1].under != under_idx) and (self.points[over_idx+1].under != under_idx+1):
            #print("invalid undo R2 move")
            return False

        if self.edges[2*(under_idx-1)].face == self.edges[2*(under_idx+1)].face:
            f2_idx = self.edges[2*(under_idx-1)].face
            f1_idx = self.edges[2*under_idx-1].face
            f4_idx = self.edges[2*(under_idx+1)+1].face
            f3_idx = self.edges[2*under_idx+1].face
            f5_idx = self.edges[2*under_idx].face
        else:
            f2_idx = self.edges[2*under_idx-1].face
            f1_idx = self.edges[2*(under_idx-1)].face
            f4_idx = self.edges[2*(under_idx+1)+1].face
            f3_idx = self.edges[2*under_idx].face
            f5_idx = self.edges[2*under_idx+1].face
        if self.faces[f2_idx].edge==2*over_idx or self.faces[f2_idx].edge==2*over_idx+1:
            self.faces[f2_idx].edge=self.edges[self.faces[f2_idx].edge].next
        if self.faces[f3_idx].edge==2*under_idx or self.faces[f3_idx].edge==2*under_idx+1:
            self.faces[f3_idx].edge=self.edges[self.faces[f3_idx].edge].next

        if f4_idx > f1_idx:
            self.reindex_face(self.faces[f4_idx].edge, f1_idx)
            remove_face_idx1 = f4_idx
        else:
            self.reindex_face(self.faces[f1_idx].edge, f4_idx)
            remove_face_idx1 = f1_idx

        self.edges[2*under_idx].face = self.edges[2*(under_idx-1)].face
        self.edges[2*under_idx+1].face = self.edges[2*under_idx-1].face
        self.edges[2*over_idx].face = self.edges[2*(over_idx-1)].face
        self.edges[2*over_idx+1].face = self.edges[2*over_idx-1].face

        self.link_edges(2*(under_idx-1), 2*under_idx)
        self.link_edges(2*under_idx, 2*(under_idx+1))
        self.link_edges(2*(under_idx+1)+1, 2*under_idx+1)
        self.link_edges(2*under_idx+1, 2*under_idx-1)
        self.link_edges(2*(over_idx-1), 2*over_idx)
        self.link_edges(2*over_idx, 2*(over_idx+1))
        self.link_edges(2*(over_idx+1)+1, 2*over_idx+1)
        self.link_edges(2*over_idx+1, 2*over_idx-1)

        if remove_face_idx1 > f5_idx:
            remove_face_idx2 = remove_face_idx1
            remove_face_idx1 = f5_idx
        else:
            remove_face_idx2 = f5_idx

        self.faces = self.faces[:remove_face_idx1] + \
                     self.faces[remove_face_idx1+1:remove_face_idx2] + \
                     self.faces[remove_face_idx2+1:]
        for e in self.edges:
            if e.face > remove_face_idx2:
                e.face -= 2
            elif e.face > remove_face_idx1:
                e.face -= 1

        if over_idx <= under_idx:
            self.removePoint(under_idx+1)
            self.removePoint(under_idx)
            self.removePoint(over_idx+1)
            self.removePoint(over_idx)
        else:
            self.removePoint(over_idx+1)
            self.removePoint(over_idx)
            self.removePoint(under_idx+1)
            self.removePoint(under_idx)
        return True


def reverse_action(action, before_state, after_state):
    # action is a dictionary of function name and arguments.
    # states are AbstractState objects.
    if action['move']=='cross':
        reverse_action = {'move':'undo_cross'}
        if action['over_idx']==0 or action['under_idx']==0:
            reverse_action['head']=True
        else:
            reverse_action['head']=False
    if action['move']=='R1':
        reverse_action = {'move':'undo_R1'}
        reverse_action['idx'] = action['idx']+1
    if action['move']=='R2':
        reverse_action = {'move':'undo_R2'}
        if action['over_idx'] < action['under_idx'] or \
            (action['over_idx'] == action['under_idx'] and action['over_before_under']==1):
            reverse_action['over_idx'] = action['over_idx']+1
            reverse_action['under_idx'] = action['under_idx']+3
        else:
            reverse_action['over_idx'] = action['over_idx']+3
            reverse_action['under_idx'] = action['under_idx']+1
    if action['move']=='undo_cross':
        reverse_action = {'move':'cross'}
        if action['head']:
            reverse_action['sign'] = before_state.points[1].sign
            if before_state.points[1].over is not None:
                reverse_action['under_idx']=0
                reverse_action['over_idx']=before_state.points[1].over-2
            else:
                reverse_action['over_idx']=0
                reverse_action['under_idx']=before_state.points[1].under-2
        else:
            reverse_action['sign'] = before_state.points[-2].sign
            if before_state.points[-2].over is not None:
                reverse_action['under_idx']=after_state.pts
                reverse_action['over_idx']=before_state.points[-2].over-1
            else:
                reverse_action['over_idx']=after_state.pts
                reverse_action['under_idx']=before_state.points[-2].under-1
    if action['move']=='undo_R1':
        reverse_action = {'move':'R1'}
        reverse_action['idx'] = action['idx'] - 1
        reverse_action['sign'] = before_state.points[action['idx']].sign
        if before_state.points[action['idx']].over == action['idx']+1:
            reverse_action['left'] = reverse_action['sign']
        elif before_state.points[action['idx']].under == action['idx']+1:
            reverse_action['left'] = -reverse_action['sign']
        else:
            pdb.set_trace()
    if action['move']=='undo_R2':
        reverse_action = {'move':'R2'}
        if action['over_idx']<action['under_idx']:
            reverse_action['over_before_under'] = 1
            reverse_action['over_idx'] = action['over_idx']-1
            reverse_action['under_idx'] = action['under_idx']-3
        else:
            reverse_action['over_before_under'] = -1
            reverse_action['over_idx'] = action['over_idx']-3
            reverse_action['under_idx'] = action['under_idx']-1
        reverse_action['left'] = before_state.points[action['under_idx']+1].sign
    return reverse_action
