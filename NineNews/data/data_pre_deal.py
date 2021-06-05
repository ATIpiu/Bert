import csv

if __name__ == '__main__':

    csvFile =open('dev.csv', 'r',encoding="utf-8")
    reader = csv.reader(csvFile)

    file=open("E:/Bert/NineTitleNews/data/dev.txt", 'w+', encoding='utf-8')
    csvFile.readline()
    for line in reader:
        if len(line)==3:
            file.writelines(line[0]+' \t@'+line[2]+'\n')
        else:print(line)
