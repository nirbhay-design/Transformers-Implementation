#!/bin/bash

if [ "$#" -eq 0 ]; then
    echo "commit message is required"
    exit 1
fi

commit_msg=$1

git add .
git commit -m "$commit_msg"
git push origin master 