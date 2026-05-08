"""
app.py — Flask backend for Webpage MindMap Explorer.

This version parses webpages into a layout-based hierarchy using headings
(h1-h6) instead of extracting random noun phrases as first-level nodes.

Setup:
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
    set GEMINI_API_KEY=your_key_here        # Windows CMD
    # or $env:GEMINI_API_KEY="your_key"     # PowerShell
    python app.py
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import re
from urllib.parse import urlparse
from typing import List, Dict, Any

import requests as req
from bs4 import BeautifulSoup
from bs4.element import Tag, NavigableString
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import networkx as nx

app = Flask(__name__)
CORS(app)

print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("Models ready.")

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

NOISY_HEADINGS = {
    "contents", "references", "external links", "further reading", "see also",
    "notes", "bibliography", "sources", "navigation menu", "personal tools",
    "appearance", "article", "talk", "read", "view source", "view history",
}

NOISY_TEXT_PREFIXES = (
    "jump to", "toggle", "edit", "coordinates:", "retrieved from", "categories:",
)

# ── Basic helpers ─────────────────────────────────────────────────────────────

def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def slugify(text: str, fallback: str = "node") -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return text[:80] or fallback


def is_valid_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
    except Exception:
        return False


def create_node(title: str) -> Dict[str, Any]:
    return {"title": normalize_space(title) or "Untitled Section", "text": "", "subnodes": []}


def append_text_to_node(node: Dict[str, Any], text: str) -> None:
    cleaned = normalize_space(text)
    if not cleaned:
        return
    if node["text"]:
        node["text"] += " " + cleaned
    else:
        node["text"] = cleaned


def is_noise_text(text: str) -> bool:
    text = normalize_space(text)
    if not text or len(text) < 3:
        return True
    low = text.lower()
    if low in NOISY_HEADINGS:
        return True
    if any(low.startswith(prefix) for prefix in NOISY_TEXT_PREFIXES):
        return True
    if re.fullmatch(r"\[?edit\]?", low):
        return True
    return False

# ── Fetch and clean webpage ───────────────────────────────────────────────────

def fetch_html(url: str) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }
    response = req.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    return response.text


def clean_soup(html: str) -> BeautifulSoup:
    soup = BeautifulSoup(html, "lxml")

    # Remove non-content/layout elements.
    for tag in soup([
        "script", "style", "noscript", "svg", "canvas", "iframe", "form", "input",
        "button", "nav", "footer", "header", "aside"
    ]):
        tag.decompose()

    # Wikipedia-specific cleanup and common clutter.
    for selector in [
        ".mw-editsection", ".reference", ".reflist", ".navbox", ".infobox",
        ".metadata", ".toc", "#toc", ".sidebar", ".hatnote", ".shortdescription",
        ".printfooter", ".catlinks", ".mw-jump-link", ".noprint"
    ]:
        for tag in soup.select(selector):
            tag.decompose()

    return soup


def get_content_root(soup: BeautifulSoup):
    # Prefer actual article body over full webpage chrome.
    return (
        soup.select_one("#mw-content-text .mw-parser-output")
        or soup.find("article")
        or soup.find("main")
        or soup.body
        or soup
    )


def get_page_title(soup: BeautifulSoup, url: str) -> str:
    h1 = soup.find("h1")
    if h1:
        title = normalize_space(h1.get_text(" ", strip=True))
        if title:
            return title[:90]
    if soup.title and soup.title.string:
        title = normalize_space(soup.title.string)
        if title:
            return title[:90]
    return urlparse(url).netloc

# ── Layout-based HTML parser ──────────────────────────────────────────────────

def parse_html_to_nodes(html_string: str) -> List[Dict[str, Any]]:
    """
    Convert HTML into a hierarchy based on visible heading layout.

    Rules:
    - h1 creates a root node.
    - h2 becomes child of latest h1.
    - h3 becomes child of latest h2, and so on.
    - Text between headings belongs to the current active node.
    - Text before the first heading becomes an Introduction node.
    """
    if not html_string or not html_string.strip():
        return []

    soup = clean_soup(html_string)
    content_root = (soup)

    root_nodes: List[Dict[str, Any]] = []
    stack: List[tuple[int, Dict[str, Any]]] = []
    intro_node = None
    seen_content = set()

    def add_node_by_heading(level: int, title: str) -> Dict[str, Any]:
        nonlocal stack
        title = normalize_space(re.sub(r"\[.*?\]", "", title))
        if not title or title.lower() in NOISY_HEADINGS:
            title = "Untitled Section"

        new_node = create_node(title)

        # Pop until nearest heading of lower level becomes parent.
        while stack and stack[-1][0] >= level:
            stack.pop()

        if stack:
            stack[-1][1]["subnodes"].append(new_node)
        else:
            root_nodes.append(new_node)

        stack.append((level, new_node))
        return new_node

    def get_current_node() -> Dict[str, Any]:
        nonlocal intro_node
        if stack:
            return stack[-1][1]
        if intro_node is None:
            intro_node = create_node("Introduction")
            root_nodes.append(intro_node)
        return intro_node

    def add_text(text: str) -> None:
        text = normalize_space(re.sub(r"\[\d+\]", "", text))
        if is_noise_text(text):
            return
        # Avoid repeated list/nav text.
        key = text.lower()[:200]
        if key in seen_content:
            return
        seen_content.add(key)
        append_text_to_node(get_current_node(), text)

    def walk(element):
        for child in element.children:
            if isinstance(child, NavigableString):
                # Ignore loose whitespace; meaningful text is usually captured by parent tags.
                continue

            if not isinstance(child, Tag):
                continue

            tag_name = child.name.lower()

            if tag_name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
                heading_text = child.get_text(" ", strip=True)
                heading_text = normalize_space(re.sub(r"\[.*?\]", "", heading_text))
                if heading_text and heading_text.lower() not in NOISY_HEADINGS:
                    add_node_by_heading(int(tag_name[1]), heading_text)
                continue

            if tag_name in {"p", "li", "blockquote", "figcaption", "td", "th"}:
                add_text(child.get_text(" ", strip=True))
                continue

            # Tables often create noisy mindmap nodes; keep text only from simple article tables if needed.
            if tag_name == "table":
                continue

            walk(child)

    walk(content_root)

    def prune(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        pruned = []
        for node in nodes:
            node["title"] = normalize_space(node.get("title", "")) or "Untitled Section"
            node["text"] = normalize_space(node.get("text", ""))
            node["subnodes"] = prune(node.get("subnodes", []))

            # Remove empty Introduction if actual heading nodes exist.
            if node["title"] == "Introduction" and not node["text"] and root_nodes:
                continue

            if node["text"] or node["subnodes"] or node["title"]:
                pruned.append(node)
        return pruned

    return prune(root_nodes)


def ensure_single_root(nodes: List[Dict[str, Any]], title: str) -> Dict[str, Any]:
    """
    UI expects one root graph title. If webpage already has one main h1, use it.
    Otherwise wrap multiple roots under page title.
    """
    if not nodes:
        return create_node(title)

    # If there is one meaningful h1-like root, use page title as display but keep its children.
    if len(nodes) == 1:
        root = nodes[0]
        root["title"] = title or root["title"]
        return root

    intro_texts = []
    subnodes = []
    for node in nodes:
        if node["title"].lower() == "introduction":
            if node.get("text"):
                intro_texts.append(node["text"])
            subnodes.extend(node.get("subnodes", []))
        else:
            subnodes.append(node)

    return {
        "title": title,
        "text": " ".join(intro_texts),
        "subnodes": subnodes,
    }

# ── Graph/chunk conversion ────────────────────────────────────────────────────

def flatten_nodes_for_chunks(node: Dict[str, Any], chunks: list, path: str = ""):
    section_path = f"{path} > {node['title']}" if path else node["title"]
    text = normalize_space(node.get("text", ""))
    if len(text) >= 40:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=70,
            separators=["\n\n", "\n", ". ", " "]
        )
        for i, chunk_text in enumerate(splitter.split_text(text)):
            chunks.append({
                "id": len(chunks),
                "section": node["title"],
                "path": section_path,
                "page": 1,
                "text": chunk_text,
                "chunk_index": i,
            })

    for child in node.get("subnodes", []):
        flatten_nodes_for_chunks(child, chunks, section_path)


def build_graph_from_node(root_node: Dict[str, Any]):
    G = nx.DiGraph()
    used_ids = set()

    def unique_id(prefix: str, title: str) -> str:
        base = f"{prefix}_{slugify(title)}"
        candidate = base
        counter = 1
        while candidate in used_ids:
            counter += 1
            candidate = f"{base}_{counter}"
        used_ids.add(candidate)
        return candidate

    def add_recursive(node: Dict[str, Any], parent_id: str | None, depth: int):
        if parent_id is None:
            node_id = "root"
            level = "root"
            used_ids.add(node_id)
        else:
            node_id = unique_id(f"h{min(depth + 1, 6)}", node.get("title", "section"))
            level = "section"

        children = node.get("subnodes", []) or []
        G.add_node(
            node_id,
            label=node.get("title", "Untitled Section"),
            level=level,
            page=1,
            body_text=normalize_space(node.get("text", ""))[:1200],
        )
        if parent_id:
            G.add_edge(parent_id, node_id)

        for child in children:
            add_recursive(child, node_id, depth + 1)

    add_recursive(root_node, None, 0)

    for node_id in G.nodes:
        children = list(G.successors(node_id))
        # Any node with children should be explorable in a heading-based graph.
        G.nodes[node_id]["dynamic"] = len(children) > 0
        G.nodes[node_id]["child_count"] = len(children)

    return G


def graph_to_json(G):
    nodes = [
        {
            "id": node_id,
            "label": data.get("label", node_id),
            "level": data.get("level", "section"),
            "page": data.get("page", 1),
            "dynamic": data.get("dynamic", False),
            "child_count": data.get("child_count", 0),
            "body_text": data.get("body_text", "")[:1200],
        }
        for node_id, data in G.nodes(data=True)
    ]
    edges = [{"from": source, "to": target} for source, target in G.edges()]
    return {"nodes": nodes, "edges": edges}

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def home():
    return send_from_directory(".", "index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/process", methods=["POST"])
def process():
    data = request.get_json(silent=True) or {}
    url = (data.get("url") or request.form.get("url") or "").strip()

    if not url:
        return jsonify({"error": "No webpage URL provided."}), 400
    if not is_valid_url(url):
        return jsonify({"error": "Please enter a valid http/https webpage URL."}), 400

    try:
        print(f"\nProcessing webpage: {url}")
        html = fetch_html(url)
        soup_for_title = clean_soup(html)
        title = get_page_title(soup_for_title, url)

        parsed_roots = parse_html_to_nodes(html)
        root_node = ensure_single_root(parsed_roots, title)

        if not root_node.get("subnodes") and not root_node.get("text"):
            return jsonify({"error": "Could not extract enough readable heading-based content from this webpage."}), 400

        chunks = []
        flatten_nodes_for_chunks(root_node, chunks)

        G = build_graph_from_node(root_node)
        graph_data = graph_to_json(G)

        section_count = sum(1 for n in graph_data["nodes"] if n["level"] == "section")
        dynamic_count = sum(1 for n in graph_data["nodes"] if n["dynamic"])

        print(f"  ✓ {section_count} heading nodes")
        print(f"  ✓ {len(chunks)} chunks")
        print(f"  ✓ {len(graph_data['nodes'])} nodes, {len(graph_data['edges'])} edges")

        return jsonify({
            "graph": graph_data,
            "chunks": chunks,
            "title": title,
            "url": url,
            "stats": {
                "pages": 1,
                "sections": section_count,
                "concepts": 0,
                "dynamic": dynamic_count,
            }
        })

    except req.exceptions.Timeout:
        return jsonify({"error": "This webpage took too long to respond. Try another URL."}), 504
    except req.exceptions.RequestException as e:
        return jsonify({"error": f"Could not fetch webpage: {str(e)}"}), 502
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/explain", methods=["POST"])
def explain():
    data = request.get_json() or {}
    label = normalize_space(data.get("label", ""))
    body_text = normalize_space(data.get("body_text", ""))
    level = data.get("level", "section")

    gemini_key = os.getenv("GEMINI_API_KEY") or data.get("gemini_key", "")

    if not label:
        return jsonify({"explanation": "No node label was provided."}), 400

    # Fallback keeps the demo usable even when the API key is missing/quota is exhausted.
    if not gemini_key:
        if body_text:
            fallback = body_text[:500]
            if len(body_text) > 500:
                fallback += "..."
        else:
            fallback = f"{label} is a node extracted from the webpage hierarchy."
        return jsonify({"explanation": fallback})

    context_limit = 1800 if level in {"root", "section"} else 1400
    context = body_text[:context_limit]

    if context:
        prompt = f"""
You are explaining a node from an interactive webpage mindmap.

Node title: {label}
Node type: {level}
Webpage context:
{context}

Task:
Write a clear explanation of this node in 3 to 4 complete lines.
Use the webpage context as the main source.
Explain the main meaning, important details, and why it matters in this topic.
Use simple beginner-friendly language.
Do not use bullet points.
Do not use markdown.
Do not write a heading.
Do not stop mid-sentence.
End with a complete sentence.
"""
    else:
        prompt = f"""
You are explaining a node from an interactive webpage mindmap.

Node title: {label}
Node type: {level}

Task:
Write a clear explanation of this node in 3 to 4 complete lines.
Use simple beginner-friendly language.
Do not use bullet points.
Do not use markdown.
Do not write a heading.
Do not stop mid-sentence.
End with a complete sentence.
"""

    try:
        resp = req.post(
            f"{GEMINI_URL}?key={gemini_key}",
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.35,
                    "maxOutputTokens": 420,
                },
            },
            timeout=45,
        )

        if resp.status_code != 200:
            try:
                msg = resp.json().get("error", {}).get("message", "Unknown error")
            except Exception:
                msg = resp.text
            return jsonify({"explanation": f"Gemini error: {msg}"}), 500

        text = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()

        # Small safety cleanup for rare markdown-style responses.
        text = text.replace("```", "").strip()

        return jsonify({"explanation": text})

    except Exception as e:
        return jsonify({"explanation": f"Error: {str(e)}"}), 500


@app.route("/search", methods=["POST"])
def search():
    data = request.get_json() or {}
    query = data.get("query", "")
    chunks = data.get("chunks", [])
    k = int(data.get("k", 5))

    if not query or not chunks:
        return jsonify({"results": []})

    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=False).astype("float32")
    q_vec = embedder.encode([query]).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    distances, indices = index.search(q_vec, min(k, len(chunks)))

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        c = chunks[int(idx)].copy()
        c["score"] = round(float(dist), 4)
        results.append(c)

    return jsonify({"results": results})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
