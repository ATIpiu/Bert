import csv

if __name__ == '__main__':

    csvFile =open('dev.csv', 'r',encoding="utf-8")
    reader = csv.reader(csvFile)
    file=open("dev.txt", 'w+', encoding='utf-8')
    csvFile.readline()
    for line in reader:
        if len(line)==3:
            file.writelines(line[2]+' \t@'+line[1]+'\n')
        else:print(line)
