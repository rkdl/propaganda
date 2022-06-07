import csv
import json
import logging
from typing import Generator, Iterable

import requests
import bs4

log = logging.getLogger(__name__)

SITEMAP_FNAME = 'sitemap_2022.xml'
REQUEST_HEADERS_FNAME = 'request_headers.json'
OUTPUT_FNAME = 'rt_articles.csv'


def load_request_headers() -> dict:
    with open(REQUEST_HEADERS_FNAME, 'r') as fp:
        return json.load(fp)


REQUEST_HEADERS = load_request_headers()


def get_sitemap_urls() -> list[str]:
    # response = requests.get('https://www.rt.com/sitemap_2022.xml')
    # content = response.text
    with open(SITEMAP_FNAME, 'rt') as f:
        content = f.read()

    soup = bs4.BeautifulSoup(content, 'lxml')

    urls = []

    for loc in soup.select('loc'):
        url = loc.text
        urls.append(url)

    return urls


def get_article_content(body: str) -> dict:
    html_parser = bs4.BeautifulSoup(body, 'html.parser')

    header_node = html_parser.select_one('.article__heading')
    summary_node = html_parser.select_one('.article__summary')
    body_node = html_parser.select_one('.article__text')

    title_inner_text = (
        header_node.get_text(strip=True)
        if header_node
        else None
    )
    summary_inner_text = (
        summary_node.get_text(strip=True)
        if summary_node
        else None
    )
    body_inner_text = (
        body_node.get_text(strip=True)
        if body_node
        else None
    )

    return {
        'title': title_inner_text,
        'summary': summary_inner_text,
        'body': body_inner_text,
    }


def get_articles(articles_urls: Iterable[str]) -> Generator[dict, None, None]:
    for i, _url in enumerate(articles_urls):

        response = requests.get(_url, headers=REQUEST_HEADERS)
        log.info(response)
        if not response.ok:
            log.error(
                f'got response with error: {response.status_code=}. '
                'trying to skip...'
            )
            continue

        log.info(f'got {i} article out of {len(articles_urls)}')
        content = get_article_content(response.text)

        yield content


def write_articles_to_file(articles: Iterable[dict]) -> None:
    log.info('start writing tweets...')

    with open(OUTPUT_FNAME, 'w') as fp:
        writer = csv.writer(fp)

        header = ['title', 'summary', 'body']
        writer.writerow(header)

        for article in articles:
            writer.writerow(
                [article['title'], article['summary'], article['body']]
            )

    log.info('finished writing tweets')


def main() -> None:
    urls = get_sitemap_urls()

    log.info('got urls, processing articles...')

    articles_stream = get_articles(urls)
    write_articles_to_file(articles_stream)

    log.info('job done!')


if __name__ == '__main__':
    main()
