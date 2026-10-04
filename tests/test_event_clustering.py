from src.engine.event_clustering import EventClusterer
from src.ingestion.demo import load_articles


def test_related_articles_are_clustered():

    articles = load_articles()

    clusterer = EventClusterer()

    clusters = clusterer.cluster(
        articles
    )

    article_to_cluster = {}

    for cluster in clusters:

        for article_id in cluster.article_ids:
            article_to_cluster[
                article_id
            ] = cluster.cluster_id

    # Rate hike articles.
    assert (
        article_to_cluster["art_001"]
        == article_to_cluster["art_002"]
    )

    # NovaBank regulatory articles.
    assert (
        article_to_cluster["art_003"]
        == article_to_cluster["art_004"]
        == article_to_cluster["art_005"]
    )

    # Energy disruption articles.
    assert (
        article_to_cluster["art_007"]
        == article_to_cluster["art_008"]
    )

    # MetroCredit credit articles.
    assert (
        article_to_cluster["art_010"]
        == article_to_cluster["art_011"]
    )

    # Distinct events must remain separate.
    assert (
        article_to_cluster["art_006"]
        != article_to_cluster["art_009"]
    )

    assert (
        article_to_cluster["art_001"]
        != article_to_cluster["art_003"]
    )

    assert (
        article_to_cluster["art_007"]
        != article_to_cluster["art_010"]
    )


def test_clusters_have_expected_size():

    articles = load_articles()

    clusterer = EventClusterer()

    clusters = clusterer.cluster(
        articles
    )

    sizes = sorted(
        [
            len(cluster.article_ids)
            for cluster in clusters
        ]
    )

    assert sizes == [
        1,
        1,
        1,
        2,
        2,
        2,
        3,
    ]