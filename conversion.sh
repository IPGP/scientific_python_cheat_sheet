#!/bin/bash

pandoc --variable classoption:twocolumn \
       --variable papersize:a4paper \
       -s sheet.md \
       -o sheet.pdf



