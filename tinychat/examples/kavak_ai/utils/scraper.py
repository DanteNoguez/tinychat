import json
import os
import re

import requests
from bs4 import BeautifulSoup, NavigableString


def clean_text(text):
    # Remove extra whitespace and newlines
    return re.sub(r"\s+", " ", text).strip()


def scrape_blog_content(url):
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        blog_post_content_inner = soup.find("div", id="blog-post-content-inner")

        if blog_post_content_inner:
            return str(blog_post_content_inner)
        else:
            print(
                "The div with id 'blog-post-content-inner' was not found on the page."
            )
            return None
    else:
        print(f"Failed to retrieve the page. Status code: {response.status_code}")
        return None


def chunk_content(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    chunks = []
    current_chunk = []

    def process_element(element, level=0):
        if element.name in ["h2", "h3"]:
            text = element.get_text().strip()
            if text:  # Only process non-empty headers
                if current_chunk:
                    chunks.append(clean_text(" ".join(current_chunk)))
                current_chunk.clear()
                current_chunk.append(f"{'#' * level} {text}")
        elif element.name in ["p", "ul", "ol"]:
            text = []
            for child in element.children:
                if isinstance(child, NavigableString):
                    text.append(child.strip())
                elif child.name == "a":
                    text.append(f"{child.get_text().strip()} ({child.get('href', '')})")
                elif child.name == "img":
                    text.append(f"[Image: {child.get('alt', 'No description')}]")
                elif child.name in ["strong", "em", "b", "i"]:
                    text.append(child.get_text().strip())
                elif child.name in ["ul", "ol"]:
                    text.append(process_element(child, level + 1))
            processed_text = " ".join(text).strip()
            if processed_text:  # Only add non-empty processed text
                current_chunk.append(processed_text)
        elif element.name == "li":
            return f"- {process_element(element, level)}"
        return " ".join(text).strip() if "text" in locals() else ""

    for element in soup.find_all(["h2", "h3", "p", "ul", "ol"]):
        process_element(element, 2 if element.name == "h2" else 3)

    if current_chunk:
        chunks.append(clean_text(" ".join(current_chunk)))

    return [
        chunk
        for chunk in chunks
        if chunk.strip() != "#" * 2 and chunk.strip() != "#" * 3
    ]


if __name__ == "__main__":
    # url = "https://www.kavak.com/mx/blog/sedes-de-kavak-en-mexico"

    current_dir = os.path.dirname(os.path.abspath(__file__))

    blog_content_path = os.path.join(current_dir, "blog_content.html")

    with open(blog_content_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    print("Raw HTML read from 'blog_content.html'")

    chunks = chunk_content(html_content)

    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}:")
        print(chunk)
        print("-" * 50)

    json_file_path = os.path.join(current_dir, "text_chunks.json")
    with open(json_file_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print("Text chunks saved to 'text_chunks.json'")
