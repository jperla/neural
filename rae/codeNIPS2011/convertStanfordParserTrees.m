load('vars.normalized.100.mat');

global wordMap;
wordMap = containers.Map(words,1:length(words));

allSNum = {};
allSStr = {};
allSOStr = {};
allSPOS = {};
allSTree = {};


fid = fopen(inputFile, 'r');
fileLines = textscan(fid, '%s', 'delimiter', '\n', 'bufsize', 100000);
fclose(fid);
fileLines=fileLines{1};


sNum=[];
c = [];
cc = [];
for i=1:length(fileLines)
    if mod(i,1000) == 0
        display([num2str(i) ' of ' num2str(length(fileLines))])
    end
    if (isempty(fileLines{i}))
        continue
    end
    
    if strcmp(fileLines{i}, 'SENTENCE_SKIPPED_OR_UNPARSABLE') || strcmp(fileLines{i}(1), '<')
        allSNum{end+1} = [];
        allSStr{end+1} = [];
        allSOStr{end+1} = [];
        allSPOS{end+1} = [];
        allSTree{end+1} = [];
        c = [c length(allSNum)];
        continue
    end
    
    if strcmp(fileLines{i}, 'Sentence skipped: no PCFG fallback.')
        cc = [cc length(allSNum)];
        continue
    end
    
    line = regexp(fileLines{i},' ','split');
    if isempty(line)
        continue
    end
    %if strcmp(line{1},'((SINV') || (length(line) >= 2 && strcmp(line{1},'(') && (strcmp(line{2},'(S') || strcmp(line{2},'(FRAG')))
    % too many freakin special cases! see:  grep -hE "^\(" * | sort | uniq
    if isempty(sNum)
        sNum = [-1]; % -1 for internal nodes
        sStr = {''};
        sOStr = {''};
        
        posTag = regexp(fileLines{i}, '([A-Z]+)', 'match');
        sPOS = {posTag{1}};
        %             disp(['Starting new phrase (POS: ' sPOS{1} '). Full line is: ' fileLines{i}])
        
        if strcmp(fileLines{i}(1),'(') && strcmp(fileLines{i}(2),'(')
            line = ['(' line{1}(2:end) line(2:end)];
        end
        
        
        sTree= [0];
        lastParents = [1];
        currentParent = 1;
        if length(line)>2
            line = line(3:end);
        else
            continue;
        end
        
    end
    
    lineLength = length(line);
    s=1;
    if isstr(line)
        line={line};
    end
    
    while s<=lineLength
        startsBranch = strcmp(line{s}(1),'(');
        %             nextIsWord = s<lineLength && strcmp(line{s+1}(end),')');
        nextIsWord = s<lineLength && (strcmp(line{s+1}(end),')') || (~strcmp(line{s+1}(1),'(') && s<lineLength-1));
        % internal nodes
        if startsBranch && ~nextIsWord
            sTree=[sTree currentParent];
            sStr{end+1}='';
            sOStr{end+1}='';
            sPOS{end+1}=line{s}(2:end);
            sNum = [sNum -100];
            currentParent=length(sNum);
            lastParents = [lastParents currentParent];
            s=s+1;
            continue;
        end
        
        if startsBranch && nextIsWord
            numWords = 1;
            mm = regexp(line{s+numWords},'(');
            m = regexp(line{s+numWords},')');
            while length(m) <= length(mm)%isempty(m)
                word = line{s+numWords};
                word = lower(word);
                thisNum = WordLookup(word);
                sStr{end+1} = [words{thisNum}];
                sOStr{end+1} = [word];
                sTree=[sTree currentParent];
                sPOS{end+1} = line{s}(2:end);
                sNum = [sNum thisNum];
                
                numWords = numWords+1;
                assert(s+numWords <= lineLength);
                m = regexp(line{s+numWords},')');
                mm = regexp(line{s+numWords},'(');
            end
            
            if ~isempty(mm)
                word = line{s+numWords}(mm+1:m-1);
            else
                word = line{s+numWords}(1:m-1);
            end
            %                 word = regexprep(word, '[0-9]', '2'); % replace all digits with 2
            
            word = lower(word);
            thisNum = WordLookup(word);
            sStr{end+1} = [words{thisNum}];
            sOStr{end+1} = [word];
            sTree=[sTree currentParent];
            sPOS{end+1} = line{s}(2:end);
            sNum = [sNum thisNum];
            s=s+numWords+1;
            lastParents=lastParents(1:(end-(length(m)-length(mm))+1));
            if isempty(lastParents)
                assert(length(sNum)==length(sStr));
                assert(length(sNum)==length(sPOS));
                assert(length(sNum)==length(sTree));
                
                allSNum{end+1} = sNum;
                allSStr{end+1} = sStr;
                allSOStr{end+1} = sOStr;
                allSPOS{end+1} = sPOS;
                allSTree{end+1} = sTree;
                
                s=s+1;
                
                sNum = [];
                sStr = {};
                sOStr = {};
                sPOS = {};
                sTree= [];
                
                continue
            end
            currentParent = lastParents(end);
            continue
        end
    end
end

runReformatTree

