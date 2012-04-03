function string = cell2str(cellstr,delim)
% Cell string to string conversion
% Richard Socher, 2010

if nargin ==2
    delim = delim;
else
    delim = ', ';
end

string = [];
for i=1:length(cellstr),
	string = [string cellstr{i}];
	if i~=length(cellstr)
		string = [string delim];
	end
end
