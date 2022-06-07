import csv
import random


def main():
    with open('bbc_dataset.csv', 'rt', encoding='unicode_escape') as bbc_fp, \
            open('rt_articles.csv', 'rt') as rt_fp, \
            open('train_dataset.csv', 'w') as train_fp:
        bbc_reader = csv.reader(bbc_fp)
        next(bbc_reader, None)

        rt_reader = csv.reader(rt_fp)
        next(rt_reader, None)

        writer = csv.writer(train_fp)

        writer.writerow(['text', 'is_propaganda'])

        for i, row in enumerate(bbc_reader):
            writer.writerow([row[0], 0])
            if i == 4200:
                break

        for i, row in enumerate(rt_reader):
            writer.writerow([row[2], 1])


if __name__ == '__main__':
    main()
