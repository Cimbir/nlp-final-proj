import re
import random
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
import json

# Data Augmentation
templates = [
    "what is {item}",
    "whats {item}",
    "how {item} works",
    "define {item}",
    "explain {item}",
    "{item}",
]

PDF_PATH = "data/raw/jm.pdf"
OUT_PATH = "data/processed/{name}"
RANDOM_SEED = 42
INDEX_START = 618
INDEX_END = 625
PAGE_OFFSET = 7
SUMMARY_START = 2
SUMMARY_END = 7
CONTENT_START = 8
CONTENT_END = 587
MAX_WORD_WINDOW = 100
TRAIN_VAL_SPLIT_POINT = 200

random.seed(RANDOM_SEED)


def save_jsonl(data, filename):
    with open(filename, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    print("File saved", filename)


pdf = PdfReader(PDF_PATH)


def is_valid_index(s: str):
    return re.match(r"(.*?), (\d+)(, (\d+))?", s)


def to_page(s: str):
    try:
        return int(s)
    except ValueError:
        return -1


def extract_indexes(text: str):
    data = []
    text = text.replace("\n", " ")
    m = re.finditer(r"(.*?), (\d+)(, (\d+))?", text)

    for item in m:
        page = to_page(item.group(2))
        if page == -1:
            print("Page ValueError", item)
            continue
        data.append({"title": item.group(1).strip(), "page": page + PAGE_OFFSET})

    return data


def read_all_indexes():
    indexes = []
    for page in range(INDEX_START, INDEX_END):
        print("Processing Page: ", page)
        indexes += extract_indexes(pdf.pages[page].extract_text())
    # print(len(indexes), indexes)
    return indexes


def read_summary_page(text: str):
    sections = []

    lines = text.splitlines()

    for line in lines:
        m = re.match(r"(.*?) (\d+)$", line)
        if not m:
            continue
        name = m.group(1).strip(" .")
        sections.append(name)

    return sections


def read_all_sections():
    sections = []
    for page in range(SUMMARY_START, SUMMARY_END + 1):
        print("Processing Page: ", page)
        sections += read_summary_page(pdf.pages[page].extract_text())
    # print(len(sections), sections)
    return sections


def is_para_start(sections, line: str):
    return line in sections


def format_text(s: str):
    form_text = s.strip()
    form_text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", form_text)
    form_text = re.sub(r"\s+", " ", form_text).strip()
    return form_text


def read_section_content(sections):
    sect = []

    buffer = {"section": "", "pages": []}
    for page in range(CONTENT_START, CONTENT_END + 1):
        content = pdf.pages[page].extract_text()

        lines = content.splitlines()

        if len(lines) == 0:
            continue

        if "CHAPTER" in lines[0]:
            # Remove the chapter name top of the every page
            lines = lines[1:]

        page_buf = ""
        for line in lines:
            line = line.strip()

            if is_para_start(sections, line):
                # Is a paragraph start
                if page_buf != "":
                    buffer["pages"].append(
                        {"page": page, "text": format_text(page_buf)}
                    )
                page_buf = line + " "
                sect.append(buffer)
                buffer = {"section": line, "pages": []}
            else:
                # Part of existing paragraph
                page_buf += line + " "

        if page_buf != "":
            buffer["pages"].append({"page": page, "text": format_text(page_buf)})

    # Off-by-one
    if len(buffer["pages"]) > 0:
        sect.append(buffer)

    return sect


def rechunk_sections(sections):
    pages = {}

    for sect in sections:
        for page_idx, page_content in enumerate(sect["pages"]):
            page = page_content["page"]
            if not page in pages:
                pages[page] = []

            # Common problem: index reference is in the very end of the page,
            # Fix: attach the following page to the current one if it exists
            text = page_content["text"]
            if page_idx != len(sect["pages"]) - 1:
                text += sect["pages"][page_idx + 1]["text"]

            pages[page].append({"section": sect["section"], "page": page, "text": text})

    for page in pages.keys():
        print(page, len(pages[page]))

    return pages


def contains_index(index_title, text) -> bool:
    words = index_title.lower().split(" ")
    text = text.lower()
    num_contains = 0

    for word in words:
        if word in text:
            num_contains += 1

    if len(words) <= 1:
        return num_contains == len(words)

    return num_contains >= len(words) - 1


def crop_contents(title, cont):
    start_pos = len(cont)
    end_pos = 0

    cont = cont.lower()
    words = title.lower().split(" ")
    for word in words:
        pos = cont.find(word)
        if pos == -1:
            continue

        new_start_pos = max(0, pos - MAX_WORD_WINDOW)
        new_end_pos = min(len(cont), pos + MAX_WORD_WINDOW)

        start_pos = min(new_start_pos, start_pos)
        end_pos = max(new_end_pos, end_pos)

    if start_pos > end_pos:
        # No word match?
        return ""

    return cont[start_pos:end_pos]


def get_pos_text_for_indexes(indexes, page_to_sect):
    index_to_text = []

    for index in indexes:
        if index["title"] == "":
            continue

        page = index["page"]

        if not page in page_to_sect:
            # Skip the index we have no data about
            continue

        page_contents = ""
        section_title = ""
        sects = page_to_sect[page]
        for section in sects:
            # Which section matches?
            if contains_index(index["title"], section["text"]):
                page_contents = section["text"]
                section_title = section["section"]

        if page_contents == "":
            continue

        # Leave only the relevant parts
        page_contents = crop_contents(index["title"], page_contents)

        if page_contents == "":
            continue

        index_to_text.append(
            {
                "title": index["title"].lower(),
                "page": page,
                "section": section_title,
                "text": page_contents,
            }
        )

    return index_to_text


section_names = read_all_sections()
sections = read_section_content(section_names)
page_to_sect = rechunk_sections(sections)
indexes = read_all_indexes()
pos_text_indexes = get_pos_text_for_indexes(indexes, page_to_sect)


# Train/Validation Split
shuffled = random.sample(section_names, len(section_names))

train_sections = shuffled[:TRAIN_VAL_SPLIT_POINT]
val_sections = shuffled[TRAIN_VAL_SPLIT_POINT:]

data_train = []
data_val = []
for idx in pos_text_indexes:
    if idx["section"] in train_sections:
        data_train.append(idx)
    if idx["section"] in val_sections:
        data_val.append(idx)


def find_hard_negative(bm25, all_texts, query, pos):
    scores = bm25.get_scores(query.lower().split())
    indices = range(len(all_texts))
    # First index in ranked is the best match
    ranked = sorted(indices, key=lambda i: -scores[i])
    # This is a list of negative texts sorted from closest match to weakest match
    all_negatives = []

    for i in range(len(ranked)):
        idx = ranked[i]
        text = all_texts[idx]
        if text == pos:
            continue
        all_negatives.append(text)

    # Pick very hard to distinguish answers
    idx = random.randint(0, 2)
    return all_negatives[idx]


def final_format(dataset):
    all_texts = [item["text"] for item in dataset]
    corpus = [t.lower().split() for t in all_texts]
    bm25 = BM25Okapi(corpus)

    final = []
    for item in dataset:
        # Apply Data Augmentation
        for template in templates:
            query = item["title"]
            query = template.format(item=query)
            pos = item["text"]
            # Each augmented row may have a different negative
            neg = find_hard_negative(bm25, all_texts, query, pos)

            final.append(
                {
                    "query": query.strip(),
                    "pos": pos.strip(),
                    "neg": neg.strip(),
                }
            )

    random.shuffle(final)
    return final


final_train = final_format(data_train)
final_val = final_format(data_val)

save_jsonl(final_train, OUT_PATH.format(name="book_train_triplets.jsonl"))
save_jsonl(final_val, OUT_PATH.format(name="book_val_triplets.jsonl"))
