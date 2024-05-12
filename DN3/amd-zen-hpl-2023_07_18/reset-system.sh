#!/bin/bash
if [[ $(id -u) -ne 0 ]] ; then echo \"Please run as sudo. ex. sudo ./reset-system.sh\" ; exit 1 ; fi

    echo 3  > /proc/sys/vm/drop_caches
    echo 1  > /proc/sys/vm/compact_memory
    echo 'always' > /sys/kernel/mm/transparent_hugepage/enabled
    echo 'always' > /sys/kernel/mm/transparent_hugepage/defrag
    echo 0 > /proc/sys/kernel/numa_balancing
    cpupower frequency-set -g performance
    echo 1 > /sys/devices/system/cpu/cpufreq/boost
