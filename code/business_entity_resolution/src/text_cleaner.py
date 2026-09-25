import re
import unicodedata
import ftfy
from unidecode import unidecode

LEGAL_SUFFIXES = {
    'pvt ltd', 'private limited', 'ltd', 'limited', 'inc', 'incorporated',
    'corp', 'corporation', 'llc', 'llp', 'co', 'company', 'gmbh', 'sarl', 
    'sa', 'sas', 'eurl', 'snc', 'ste', 'societe', 'enterprises', 'ventures',
    'services', 'solutions'
}

ADDR_MAP = {
    r'\brd\b': 'road',
    r'\bst\b': 'street',
    r'\bave\b': 'avenue',
    r'\bdr\b': 'drive',
    r'\bln\b': 'lane',
    r'\bblvd\b': 'boulevard',
    r'\bct\b': 'court',
    r'\bapt\b': 'apartment',
    r'\bste\b': 'suite',
    r'\bfl\b': 'floor',
    r'\br\b': 'rue',
    r'\bbd\b': 'boulevard',
    r'\bnull\b': '',
    r'\bnan\b': '',
}


def fix_mojibake(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    text = ftfy.fix_text(text)
    if "à" in text or "Ã" in text:
        try:
            text = text.encode('latin1').decode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return text


def normalize_text(text: str) -> str:
    text = fix_mojibake(text)
    # Transliterate any script (Bengali, Hindi, Tamil, Kannada, French)
    text = unidecode(text)
    text = unicodedata.normalize('NFKD', text)
    text = text.lower()
    # Remove web junk, brackets like [[inc]]
    text = re.sub(r'https?://|www\.|\.com|\.in|\.org', '', text)
    text = re.sub(r'\[\[|\]\]', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def normalize_address(address: str) -> str:
    if not isinstance(address, str) or address.strip().lower() in ('nan', 'null', ''):
        return ""
    addr = normalize_text(address)
    for pat, rep in ADDR_MAP.items():
        addr = re.sub(pat, rep, addr)
    return re.sub(r'\s+', ' ', addr).strip()


def extract_numbers(text: str) -> set:
    if not text:
        return set()
    return set(re.findall(r'\b\d+\b', text))