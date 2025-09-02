from .build_daily_post import main as build_daily_post
from .build_daily_arxiv import main as build_daily_arxiv


if __name__ == "__main__":
    build_daily_post()
    build_daily_arxiv()
    
    