#!/usr/bin/env bash

# Memory limit
mem=10g

if [ ! $# -ge 5 ]; then
   echo Usage: `basename $0` lang len train_file test_file out_file features
   echo
   echo '  lang       : Language to parse (Arabic, English, Chinese, German, French)'
   echo '  len        : Maximum length of the sentences to parse'
   echo '  train_file : Training treebank file'
   echo '  test_file  : Test treebank file (for evaluation)'
   echo '  out_file   : Prefix for the output filename'
   echo '  features   : Variable length list of optional parser features'
   echo
   echo 'Parser memory limit is currently:' "$mem"
   echo   
   exit
fi

# Setup command-line options
lang=$1
len=$2
train_path=$3
test_file=$4
out_name=$5

shift 5

# Language-specific configuration
scriptdir=`dirname $0`
source $scriptdir/lexparser_lang.def

# Loading classpath
source $scriptdir/classpath.def
if [ -z $jarpath ]; then
  echo "Warning: install.sh has not been run. I'll try to infer the CLASSPATH..."
  CLASSPATH="$CLASSPATH":"$scriptdir"/stanford-parser.jar
else
  CLASSPATH="$CLASSPATH":"$jarpath"
fi

# Run the Stanford parser
java -server -Xmx"$mem" -Xms"$mem" edu.stanford.nlp.parser.lexparser.LexicalizedParser -v -maxLength "$len" \
-tLPP "$tlp" $lang_opts $* -writeOutputFiles \
-outputFilesExtension "$out_name"."$len".stp -outputFormat "penn" \
-outputFormatOptions "removeTopBracket,includePunctuationDependencies" -train "$train_path" -test "$test_file"

