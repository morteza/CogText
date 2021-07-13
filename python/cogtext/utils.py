import pandas as pd


def select_relevant_articles(corpus: pd.DataFrame) -> pd.DataFrame:
    """Remove certain irrelevant articles from the corpus.

    """
    def is_relevant(article):
        _is_relevant = (
            pd.notna(article['title']) and pd.notna(article['abstract']) and (
                'cognit' in article['abstract'] or
                'psych' in article['abstract'] or
                'psych' in article['journal_title'] or
                'cognit' in article['journal_title'] or
                'cognit' in article['title'] or
                'psych' in article['title']
            )
        )
        return _is_relevant
    return corpus[corpus.apply(is_relevant, axis=1)]
