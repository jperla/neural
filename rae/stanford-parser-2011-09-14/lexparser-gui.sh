#!/usr/bin/env bash
#
if [ ! $# -ge 2 ]; then
  echo Usage: `basename $0` [parserDataFilename [textFileName]]
  echo
  exit
fi

scriptdir=`dirname $0`

# Loading classpath
source $scriptdir/classpath.def
if [ -z $jarpath ]; then
  echo "Warning: install.sh has not been run. I'll try to infer the CLASSPATH..."
  CLASSPATH="$CLASSPATH":"$scriptdir"/stanford-parser.jar
else
  CLASSPATH="$CLASSPATH":"$jarpath"
fi

java -mx800m edu.stanford.nlp.parser.ui.Parser $*
