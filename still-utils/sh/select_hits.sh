#!/bin/bash

grep -A 1 "Image filename" "$1" | grep "Image\|Event" | sed 'N;s/\n/ /' | awk '{print $3,$5}'
