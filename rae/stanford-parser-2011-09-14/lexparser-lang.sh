#!/usr/bin/env bash

# Memory limit
mem=10g

if [ ! $# -ge 5 ]; then
   echo Usage: `basename $0` lang len grammar out_file in_files
   echo
   echo '  lang       : Language to parse (Arabic, English, Chinese, German, French)'
   echo '  len        : Maximum length of the sentences to parse'
   echo '  grammar    : Serialized grammar file'
   echo '  out_file   : Prefix for the output filename'
   echo '  in_files   : List of files to parse'
   echo
   echo 'To set additional parser options, modify parse_opts in parse_lang.defs'
   echo 
   echo 'Parser memory limit is currently:' "$mem"
   echo   
   exit
fi

# Setup command-line options
lang=$1
len=$2
grammar=$3
out_file=$4

shift 4

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
-tLPP "$tlp" $lang_opts $parse_opts -writeOutputFiles \
-outputFilesExtension "$out_name"."$len".stp -outputFormat "penn" \
-outputFormatOptions "removeTopBracket,includePunctuationDependencies" $grammar $*

