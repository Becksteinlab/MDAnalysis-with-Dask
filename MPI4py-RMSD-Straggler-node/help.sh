# To grep required lines from error file you can do:
grep -o '^\[.*]' slurm.10967865.err > rank_location.txt

# Where:
#The command to match a regex: .......use grep
#To match only the regex written, ........use -o
#To match the start of a line, .................use ^: ^\[
#To match any characters in between, .use .: .*
#To match the end of a line, ...................use $: apal$
#A file that contains what you want to match.: file.txt
