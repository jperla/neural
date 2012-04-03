% =====================================
% Input file

% Parse input with Stanford Parser
% From Stanford Parser README:
% On a Unix system you should be able to parse the English test file with the
% following command:
%    ./lexparser.sh input.txt > parsed.txt

inputFile = 'parsed.txt';

convertStanfordParserTrees

% =====================================
% data
outVectors = 'outVectors.txt';      % the nth line is the vector for the nth phrase
phraseFile = 'phrases.txt';         % what our RAE sees


getVectors