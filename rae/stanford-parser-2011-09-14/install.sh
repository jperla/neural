#!/usr/bin/env bash
#
# Stanford Parser setup script
#
#   Sets the classpath for the scripts in /bin
#

# Absolute path to this script.
jardir=`dirname $0`

if [ ! -e "$jardir/stanford-parser.jar" ]; then
  echo "Could not find stanford-parser.jar!"
  echo "Please run this script in the root directory of the Stanford Parser distribution"
  echo "Terminating setup..."
  exit
fi

echo "Setting classpath to: $jardir/stanford-parser.jar"
echo "Modify classpath.def if this is not correct."
echo "jarpath=$jardir/stanford-parser.jar" > classpath.def
