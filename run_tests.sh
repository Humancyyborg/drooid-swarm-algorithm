#!/bin/bash

python -m unittest
STATUS_ALL_TESTS=$?

echo "Status: $STATUS_ALL_TESTS"
echo "0 means success (no errors), non-zero status indicates failed tests"