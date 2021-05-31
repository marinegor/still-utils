#!/bin/bash

CURRENTFILE="$1"
cp "$CURRENTFILE" "$CURRENTFILE.backup"

sed -i -e 's/\],\s/)\n/g' "$CURRENTFILE"
sed -i -e 's/\[/(/g' "$CURRENTFILE"
sed -i -e 's/\]/)/g' "$CURRENTFILE"
sed -i -e 's/,//g' "$CURRENTFILE"
sed -i -e s/\'/\"/g "$CURRENTFILE"
