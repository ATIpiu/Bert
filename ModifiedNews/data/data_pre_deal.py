import csv

if __name__ == '__main__':

    csvFile =open('test_data.csv', 'r',encoding="utf-8")
    reader = csv.reader(csvFile)
    file=open("test_data.txt", 'w+', encoding='utf-8')
    csvFile.readline()
    for line in reader:
        if len(line)==3:
            file.writelines(line[0][50:150]+' \t@'+line[1]+'\n')
        else:print(line)
