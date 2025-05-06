#!/bin/bash

find . -type f -name '*_wholeds.best' | while read -r file; do
    newname="${file/_wholeds.best/.best}"
    mv "$file" "$newname"
done
