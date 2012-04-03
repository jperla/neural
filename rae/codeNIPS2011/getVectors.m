pid = fopen(phraseFile,'w');

load('vars.normalized.100.mat')

We_orig = We;

load('params.mat');

hiddenSize = 100;

for ii = 1:length(allSNum)
    instance = [];
    if (mod(ii,100) == 0)
        ii
    end
    
    instance.allSNum = allSNum{ii};
    instance.allSStr = allSStr{ii};
    instance.allSTree = allSTree{ii};
    instance.allSKids = allSKids{ii};
    
    data = instance.allSNum;
    words_indexed = data;
    %L is only the part of the embedding matrix that is relevant for this sentence
    %L is deltaWe
    if ~isempty(We)
        L = We(:, words_indexed);
        words_embedded = We_orig(:, words_indexed) + L;
    else
        words_embedded = We_orig(:, words_indexed);
    end
    
    
    [~, sl] = size(words_embedded);
    
    Tree = tree;
    Tree.pp = zeros((2*sl-1),1);
    Tree.nodeScores = zeros(2*sl-1,1);
    Tree.nodeNames = 1:(2*sl-1);
    Tree.kids = zeros(2*sl-1,2);
    
    Tree.nodeFeatures = zeros(hiddenSize, 2*sl-1, sl);
    Tree.nodeFeatures(:,1:sl,sl) = words_embedded;
    
    for i = sl+1:2*sl-1
        kids = instance.allSKids(i,:);
        if kids(1) <= sl
            ind1 = sl;
        else
            ind1 = kids(1)-sl;
        end
        if kids(2) <= sl
            ind2 = sl;
        else
            ind2 = kids(2)-sl;
        end
        c1 = Tree.nodeFeatures(:,kids(1),ind1);
        c2 = Tree.nodeFeatures(:,kids(2),ind2);
        
        p = tanh(W1*c1 + W2*c2 + b1);
       
        
        Tree.nodeFeatures(:,i,i-sl) = p;
        Tree.nodeFeatures(:,i,sl) = p;
    end
    
    fprintf(pid, '%s\n', cell2str(allSStr{ii},' '));
    if ii == 1
        dlmwrite(outVectors, Tree.nodeFeatures(:,end)');
    else
        dlmwrite(outVectors, Tree.nodeFeatures(:,end)', '-append');
    end
end


