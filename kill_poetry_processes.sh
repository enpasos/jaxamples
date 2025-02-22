#!/bin/bash

# Find and kill all processes matching the given pattern
ps aux | grep "/home/enpasos/.cache/pypoetry/virtualenvs/" | grep -v grep | awk '{print $2}' | xargs -r kill -9

echo "Killed all processes running from Poetry virtual environments."
