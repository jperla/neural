function [inc numnode newt] = reformatTree(thisNode, t, upnext)

% binarize

kids = t.kids(:,thisNode);
kids = kids(find(kids));
isLeafNodeKids = t.isLeafnode(kids(1));

while length(kids) == 1 && isLeafNodeKids ~= 1
    kkids = t.kids(:,kids(1));
    kkids = kkids(find(kkids));
    
    t.pp(kids(1)) = -1;
    t.pp(kkids) = thisNode;
    t.kids(1:length(kkids),thisNode) = kkids;
    
    kids = kkids;
    isLeafNodeKids = t.isLeafnode(kids(1));
end

numnode = 0;
isLeafNodeKids = t.isLeafnode(kids(1));
if length(kids) == 1 && isLeafNodeKids
    t.isLeafnode(thisNode) = 1;
    t.pp(kids(1)) = -1;
    t.kids(:,thisNode) = 0;
    inc = 0;
    numnode = 1;    
else
    inc = 0;

    for k = 1:length(kids);
        isLeafNodeKids = t.isLeafnode(kids(k));
        if ~isLeafNodeKids
            [thisinc thisnumnode newt] = reformatTree(kids(k), t, upnext+inc);
            inc = inc+ thisinc;
            t = newt;
            numnode = numnode+thisnumnode;
        else
            numnode = numnode+1;
        end
    end

    next = upnext + inc;
    n = length(kids);
    last = kids(end);
    start = n-1;
    while n >= 2
        if (n == 2)
            next = thisNode;
        else
            next = next + 1;
            inc = inc+1;
        end
        
        t.pp(last) = next;
        t.pp(kids(start)) = next;
        
        t.kids(:, next) = 0;
        t.kids(1, next) = kids(start);
        t.kids(2, next) = last;


        last = next;
        start = start-1;
        n = n - 1;
    end
end

newt = t;