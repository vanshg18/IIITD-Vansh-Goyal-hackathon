"""
Common interface for all EventPulse data sources.

The rest of the pipeline interacts with BaseSource rather than knowing
anything about a specific API.
"""

from abc import ABC, abstractmethod

from src.engine.schemas import Article


class BaseSource(ABC):
    """
    Abstract interface implemented by every ingestion source.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable source name."""
        raise NotImplementedError

    @property
    @abstractmethod
    def source_type(self) -> str:
        """Source category: news, social, official, demo."""
        raise NotImplementedError

    @abstractmethod
    def fetch(self, limit: int = 20, **kwargs) -> list[Article]:
        """
        Fetch and convert source data into canonical Article objects.

        Parameters
        ----------
        limit:
            Maximum number of articles requested.

        Returns
        -------
        list[Article]
            Canonical EventPulse articles.
        """
        raise NotImplementedError