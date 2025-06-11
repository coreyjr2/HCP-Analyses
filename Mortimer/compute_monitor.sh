#!/bin/bash

# Set default options
DURATION=300
LABEL="01"
TARGET_USER=$USER
OUTDIR=./
INTERVAL=5
verbose_mode=false

# Define help function
display_help() 
{
  echo
  echo "Script to monitor CPU and memory usage for the user running the command"
  echo
  echo "Usage: $0 [-d|n|o|u|i|v|h] "
  echo "Options:"
  echo "  -d | --duration     Duration to run in seconds. Default=${DURATION}"
  echo "  -n | --name         Run label (name) for the file. Default=\"${LABEL}\""
  echo "  -o | --output       Output directory. Should end in \\. Default is ${OUTDIR}"
  echo "  -u | --user         User to filter by. Only compute usage by the user is recorded. Default is \$USER"   
  echo "  -i | --interval     Interval at which to collect data. Default is ${INTERVAL}"
  echo "  -v | --verbose      Verbose mode"
  echo "  -h | --help         Display this help message"
  echo
  # echo some stuff here for the -a or --add-options 
}

# Read in Options
while [ "$1" != "" ]; do
  case $1 in
    -d | --duration)
      shift
      DURATION=$1
      ;;
    -n | --name)
      shift
      LABEL="$1"
      ;;
    -o | --output)
      shift
      OUTDIR=$1
      ;;
    -u | --user)
      shift
      TARGET_USER=$1
      ;;
    -i | --interval)
      shift
      INTERVAL=$1
      ;;
    -h | --help)
      display_help  
      exit 1
      ;;
    -v | --verbose)
      verbose_mode=true
      ;;
    *)
      echo "Invalid option: $1"
      display_help
      exit 1
      ;;
  esac
  shift
done


# Set default functional values
date=$(date '+%Y-%m-%d')
cpus=$(lscpu | grep "^CPU(s):" | awk '{print $2}')
outfile=${OUTDIR}compute-monitor_${TARGET_USER}_${date}_${LABEL}.tsv

if test -f "$outfile"; then
  if $verbose_mode; then
    echo "${outfile} already exists.\nAppending compute usage for ${USER} every ${INTERVAL} secondsfor the next ${DURATION} seconds."
  fi
else
    printf 'CPU_Usage   Memory_Usage  Time\n' > $outfile
    if $verbose_mode; then
      echo "Writing compute usage for ${USER} every ${INTERVAL} seconds for the next ${DURATION} seconds to ${outfile}."
    fi
fi


SECONDS=0
if $verbose_mode; then
  echo "DateTime              MEM(GB)  CPU(%)"
fi
while (( $SECONDS < $DURATION )); do 
  datetime=$(date '+%Y-%m-%d %H:%M:%S')
  #datetime=$(date '+%H:%M:%S')
  cpu_usage=$(top -b -n 1 -u $TARGET_USER | awk 'NR>7 { sum += $9; } END {print sum; }')
  mem_usage=$(ps haux | awk -v user=${TARGET_USER} '$1 ~ user { sum += $4} END { print sum; }')
  printf '%s\n' $cpu_usage $mem_usage $datetime | paste -sd ' ' >> $outfile
  if $verbose_mode; then
    echo "${datetime}   ${mem_usage}     ${cpu_usage}"
  fi
  sleep $INTERVAL
done