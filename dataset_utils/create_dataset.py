import pandas as pd
from sklearn.utils import shuffle

pd.set_option('display.max_colwidth', 100)


def load_twitter_messages() -> pd.DataFrame:
    ok_messages = pd.concat([
        pd.read_csv('BBCWorld_tweets.csv', encoding='utf-8'),
        pd.read_csv('bpolitics_tweets.csv', encoding='utf-8'),
        pd.read_csv('KyivIndependent_tweets.csv', encoding='utf-8'),
    ])
    ok_messages['is_propaganda'] = False

    rus_media_messages = pd.concat([
        pd.read_csv('rt_com_tweets.csv', encoding='utf-8'),
        pd.read_csv('mfa_russia_tweets.csv', encoding='utf-8'),
    ])
    rus_media_messages['is_propaganda'] = True

    ok_messages_fraction = min(1.0, len(rus_media_messages) / len(ok_messages))

    messages = pd.concat([
        ok_messages.sample(frac=ok_messages_fraction),
        rus_media_messages,
    ])

    messages = shuffle(messages)

    return messages


def main():
    messages = load_twitter_messages()
    messages.to_csv('twitter_dataset.csv')


if __name__ == '__main__':
    main()
