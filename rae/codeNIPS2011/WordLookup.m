function index = WordLookup(InputString)
global wordMap
if wordMap.isKey(InputString)
    index = wordMap(InputString);
else
    index=wordMap('*UNKNOWN*');
end
