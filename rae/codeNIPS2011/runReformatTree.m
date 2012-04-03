%% reformat the tree structures for use


numinstance = length(allSTree);

allSKids = cell(1,numinstance);
empty = [];
for instance = 1:numinstance
    if mod(instance,1000) == 0
        instance
    end
    % get embeddings
    n = length(allSTree{instance});
    
    cnt = 0;
    for j = 1:length(allSStr{instance})
        if ~isempty(allSStr{instance}{j})
            cnt = cnt+1;
        end
    end
    if cnt < 2 % words in sentence
        
        if cnt == 1
            allSNum{instance} = allSNum{instance}(end);
            allSStr{instance} = allSStr{instance}(end);
            allSTree{instance} = allSTree{instance}(end);
            allSPOS{instance} = allSPOS{instance}(end);
        else
            empty = [empty instance];
        end
        continue
    end
    
    t = tree();
    t.pp = zeros(1,n);
    t.pp(1:n) = allSTree{instance};
    mostkids = length(find(allSTree{instance}==mode(allSTree{instance}))); % largest number of kids one node has
    t.kids = zeros(mostkids,n);
    for i = 1:n
        tempkids = find(allSTree{instance}==i);
        t.kids(1:length(tempkids),i) = tempkids;
    end
    
    t.leafFeatures = zeros(1,n);
    leafs = find(allSNum{instance}>0);
    t.isLeafnode = zeros(1,2*n);
    t.isLeafnode(leafs) = 1;
    
    t.pos = allSPOS{instance};
    
    for i = 1:length(leafs)
        t.leafFeatures(leafs(i)) = allSNum{instance}(leafs(i));
    end
    
    % binarize
    [inc numnode newt] = reformatTree(1, t, n+1);
    
    opp = zeros(1,2*numnode-1);
    okids = zeros(2*numnode-1,2);
    opos = cell(2*numnode-1,1);
    
    % reorder for trainRAE
    [pp nnextleaf nnextnode nkids pos] = reorder(1, newt, 1, 2*numnode-1, opp, okids, opos);
    
    newnum = zeros(1,numnode);
    newstr = cell(1,numnode);
    next = 1;
    for i=1:length(allSNum{instance})
        if (allSNum{instance}(i) > 0)
            newnum(next) = allSNum{instance}(i);
            newstr(next) = allSStr{instance}(i);
            next = next + 1;
        end
    end
    
    allSNum{instance} = newnum;
    allSStr{instance} = newstr;
    allSTree{instance} = pp;
    allSKids{instance} = nkids;
    allSPOS{instance} = pos;
    
end

allSNum(empty) = [];
allSStr(empty) = [];
allSTree(empty) = [];
allSKids(empty) = [];
allSPOS(empty) = [];