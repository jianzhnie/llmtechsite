#!/usr/bin/env python3
"""Pre-process markdown files to escape patterns that conflict with Vue template compiler."""
import re, os

DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'docs')

# Known HTML tags to NOT escape (lowercase)
HTML_TAGS = {
    'a', 'abbr', 'address', 'area', 'article', 'aside',
    'b', 'base', 'bdi', 'bdo', 'blockquote', 'body', 'br', 'button',
    'canvas', 'caption', 'cite', 'code', 'col', 'colgroup',
    'data', 'datalist', 'dd', 'del', 'details', 'dfn', 'dialog', 'div', 'dl', 'dt',
    'em', 'embed',
    'fieldset', 'figcaption', 'figure', 'footer', 'form',
    'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'head', 'header', 'hgroup', 'hr', 'html',
    'i', 'iframe', 'img', 'input', 'ins',
    'kbd',
    'label', 'legend', 'li', 'link',
    'main', 'map', 'mark', 'math', 'menu', 'meta', 'meter',
    'nav', 'noscript',
    'object', 'ol', 'optgroup', 'option', 'output',
    'p', 'param', 'picture', 'portal', 'pre', 'progress',
    'q',
    'rp', 'rt', 'ruby',
    's', 'samp', 'script', 'section', 'select', 'slot', 'small', 'source', 'span', 'strong', 'style', 'sub', 'summary', 'sup', 'svg',
    'table', 'tbody', 'td', 'template', 'textarea', 'tfoot', 'th', 'thead', 'time', 'title', 'tr', 'track',
    'u', 'ul',
    'var', 'video',
    'wbr',
}
# Uppercase versions
HTML_TAGS |= {t.upper() for t in HTML_TAGS}

def is_html_tag(name):
    return name.lower() in HTML_TAGS

# Known non-HTML tokens that look like HTML tags but aren't
# ML tokens + AI reasoning tags + XML-style tokens
SPECIAL_TOKENS = 'EOS|BOS|UNK|PAD|SEP|CLS|MASK|EOF|EOC|MSK|eos|bos|unk|pad|sep|cls|mask|eof|eoc|msk|think|answer|s'

# Patterns for code blocks (use HTML entities to preserve display)
CODE_PATTERNS = [
    # Special tokens: <EOS>, <eos>, <think>, <answer>, <pad>, <unk>, <mask>, <s>, etc.
    (re.compile(r'(?<![`])(</?(' + SPECIAL_TOKENS + r')>)', re.IGNORECASE), lambda m: f'&lt;{m.group(1)}&gt;'),
    # Java/C++ generics
    (re.compile(r'(?<![`\\])(?<![a-zA-Z])([A-Z][a-zA-Z]*(?:\.[a-zA-Z]+)*(?:::[a-zA-Z]+)*)\s*<([A-Z][a-zA-Z]*)>'), r'\1&lt;\2&gt;'),
    # Vue template interpolation
    (re.compile(r'\{\{(?!\s)'), '&#123;&#123;'),
    (re.compile(r'(?<!\s)\}\}'), '&#125;&#125;'),
]

# Patterns for regular text (use backticks for inline code rendering)
TEXT_PATTERNS = [
    (re.compile(r'(?<![`])(</?(' + SPECIAL_TOKENS + r')>)', re.IGNORECASE), lambda m: f'`{m.group(1)}`'),
    (re.compile(r'(?<![`\\])(?<!\w)([A-Z][a-zA-Z]*(?:\.[a-zA-Z]+)*(?:::[a-zA-Z]+)*)\s*<([A-Z][a-zA-Z]*)>'), r'`\1<\2>`'),
    (re.compile(r'\{\{(?!\s)'), '{ {'),
    (re.compile(r'(?<!\s)\}\}'), '} }'),
]

# Skip these files (complex math/LaTeX content that's hard to auto-fix)
SKIP_FILES = {
    'aigc/llm_pretrain/rope.md',
    'ascend/MindSpeed-RL使用指南.md',
}

def fix_file(filepath):
    rel_path = os.path.relpath(filepath, DOCS_DIR)
    if rel_path in SKIP_FILES:
        return False
    with open(filepath, 'r') as f:
        content = f.read()

    original = content
    lines = content.split('\n')
    new_lines = []
    in_code_block = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith('```'):
            in_code_block = not in_code_block

        if in_code_block:
            for regex, replacement in CODE_PATTERNS:
                line = regex.sub(replacement, line)
        else:
            for regex, replacement in TEXT_PATTERNS:
                line = regex.sub(replacement, line)

        new_lines.append(line)

    result = '\n'.join(new_lines)
    if result != original:
        with open(filepath, 'w') as f:
            f.write(result)
        return True
    return False

def main():
    count = 0
    for root, dirs, files in os.walk(DOCS_DIR):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != 'node_modules' and d != '.vitepress']
        for f in files:
            if f.endswith('.md'):
                if fix_file(os.path.join(root, f)):
                    rel = os.path.relpath(os.path.join(root, f), DOCS_DIR)
                    print(f'Fixed: {rel}')
                    count += 1
    print(f'\nFixed {count} files.')
    return 0

if __name__ == '__main__':
    exit(main())
