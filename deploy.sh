#!/usr/bin/env bash

# deploy script for travis-ci, inspired by: https://github.com/steveklabnik/automatically_update_github_pages_with_travis_example

set -o errexit -o nounset

if [ "$TRAVIS_BRANCH" != "master" ]
then
  echo "This commit was made against the $TRAVIS_BRANCH and not the master! No deploy!"
  exit 0
fi

rev=$(git rev-parse --short HEAD)

cp index.html _site/
cd _site/

git init
git config user.name "Thomas Belahi"
git config user.email "belahi@ipgp.fr"

git remote add upstream "https://$GH_TOKEN@github.com/IPGP/scientific_python_cheat_sheet.git"
git fetch upstream
git reset upstream/gh-pages

touch .

git add -A .
git commit -m "rebuild pages at ${rev}"
git push -q upstream HEAD:gh-pages
