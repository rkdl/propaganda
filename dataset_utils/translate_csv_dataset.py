import logging
import csv
import googletrans


log = logging.getLogger(__name__)


def main():
    translator = googletrans.Translator(raise_exception=True)

    with open('twitter_dataset.csv', 'r') as in_file, \
            open('twitter_dataset_translated.csv', 'w') as out_file:

        reader = csv.DictReader(
            in_file,
            fieldnames=[None, 'id', 'created_at', 'text', 'is_propaganda'],
        )

        writer = csv.DictWriter(
            out_file,
            fieldnames=['id', 'created_at', 'text', 'is_propaganda']
        )

        for row in reader:
            try:
                translated = translator.translate(row['text'], 'uk')
            except Exception as e:
                log.exception('Something went wrong')
                continue

            writer.writerow({
                'id': row['id'],
                'created_at': row['created_at'],
                'is_propaganda': row['is_propaganda'],
                'text': translated.text,
            })

            log.info('processing...')


if __name__ == '__main__':
    main()
