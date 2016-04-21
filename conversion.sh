#!/bin/bash

# convert the cheatsheet sheet.md in a nice pdf doc using pandoc
# options don't always work depending on your version or pandoc
pandoc --variable classoption=twocolumn \
       --variable papersize=a4paper \
       --variable fontsize=10pt \
       -s sheet.md \
       -o sheet-twocolumn.pdf

pandoc --variable classoption=twocolumn \
       --variable papersize=a4paper \
       --variable fontsize=10pt \
       -s sheet.md \
       -o sheet-twocolumn.tex

pandoc --variable papersize=a4paper \
       -s sheet.md \
       -o sheet.pdf


