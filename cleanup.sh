#!/bin/bash
cd /opt/hayai
# Delete *.log files older than 7 days
find . -name "*.log" -type f -mtime +7 -delete
