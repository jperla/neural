#!/bin/bash

./stanford-parser-2011-09-14/lexparser.sh input.txt > parsed.txt

echo run | matlab -nodesktop
