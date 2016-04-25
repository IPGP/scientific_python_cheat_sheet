#!/bin/bash

# let's say you just modified sheet.md and want to regenerate the site http://ipgp.github.io/scientific_python_cheat_sheet/
# just execute this script

# requirements:
# python: markowdn, beautifulsoup
# pandoc
# git

# get revision number
rev=$(git rev-parse --short HEAD)

# recreate index.html
git checkout master
python create-index-html.py
git commit -am "regenerated index.html revision: ${rev}"

git push origin master

git checkout gh-pages
git checkout master index.html
git commit -am "regenerad website revision: ${rev}"
git push origin gh-pages

