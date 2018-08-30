
data = open('/data/backup_home_28_06_2018/home/garpel/evaluation/dbpediaVectors/uniform.txt')
output = open ('output', 'w+')
for embedding in data:
    parts = embedding.split('\t', 1)
    name = parts[0]
    vector = parts[1]
    name = '<'+name+'>'
    output.write(name)
    output.write('\t')
    output.write(vector)
    # output.write('\n')
output.close()
data.close()

