

import sys
selectionFileName = sys.argv[1]
dataFileName = sys.argv[2]
outputFileName = sys.argv[3]

selected = set()
with open(selectionFileName) as select:
    for line in select:
        line = line.rstrip('\n')
        selected.add (line)

print ("read selection, found %i" % len(selected))
print (selected)

with open(dataFileName) as data:
    with open(outputFileName, 'w+') as output:
        counter = 0
        for line in data:
            counter+=1
            if counter % 100000 == 0:
                print (counter)
            name = line.split(None, 1)[0]
            if name in selected:
                output.write(line)
