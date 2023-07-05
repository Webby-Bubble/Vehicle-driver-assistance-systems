# coding:utf-8

readDir = "over_speed_record.txt"
writeDir = "new_file.txt"
outfile = open(writeDir, "w")
f = open(readDir, "r")

lines_seen = set()  # Build an unordered collection of unique elements.

for line in f:
    line = line.strip('\n')
    if line not in lines_seen:
        outfile.write(line + '\n')
        lines_seen.add(line)