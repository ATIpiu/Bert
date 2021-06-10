import csv


def content():
    csvFile = open('dev.csv', 'r', encoding="utf-8")
    reader = csv.reader(csvFile)
    file = open('dev.txt', 'w+', encoding="utf-8-sig")
    csvFile.readline()
    for line in reader:
        file.writelines(line[1][0:65] + '\t@' + line[2] + '\n')
    csvFile = open('train.csv', 'r', encoding="utf-8")
    reader = csv.reader(csvFile)
    file = open('train.txt', 'w+', encoding="utf-8-sig")
    csvFile.readline()
    for line in reader:
        file.writelines(line[1][0:65] + '\t@' + line[2] + '\n')
    csvFile = open('test.csv', 'r', encoding="utf-8")
    reader = csv.reader(csvFile)
    file = open('test.txt', 'w+', encoding="utf-8-sig")
    csvFile.readline()
    for line in reader:
        file.writelines(line[1][0:65] + '\t@' + line[2] + '\n')


if __name__ == '__main__':
    content()
