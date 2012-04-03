function [pp nnextleaf nnextnode nkids pos] = reorder(thisNode, t, nextleaf, nextnode, opp, okids, opos)

nnextleaf = nextleaf;
nnextnode = nextnode - 1;
nkids = okids;

pp = opp;
pos = opos;

kids = t.kids(:,thisNode);
kids = kids(find(kids));

for k = 1:2
    isLeafNodeKids = t.isLeafnode(kids(k));
    if isLeafNodeKids
        pp(nnextleaf) = nextnode;
        nkids(nextnode,k) = nnextleaf;
        pos(nnextleaf) = t.pos(kids(k));
        
        nnextleaf = nnextleaf+1;
    else
        pp(nnextnode) = nextnode;
        nkids(nextnode,k) = nnextnode;
        if kids(k) <= length(t.pos)
            pos(nnextnode) = t.pos(kids(k));
        end
        [pp nnextleaf nnextnode nkids pos] = reorder(kids(k), t, nnextleaf, nnextnode, pp, nkids, pos);
    end
end