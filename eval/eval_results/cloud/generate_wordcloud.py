#!/usr/bin/env python3
"""
Generate word cloud from task goals in tiny-bench dataset.
Mimics the style of the reference word cloud with colorful multi-hue palette.
"""

import json
import os
import re
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import random

# Define custom color function with deeper colors
def custom_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    """Custom color function with deeper blues, greens, yellows, purples palette"""
    colors = [
        '#0047AB',  # cobalt blue (deep)
        '#00008B',  # dark blue
        '#191970',  # midnight blue
        '#006400',  # dark green
        '#228B22',  # forest green
        '#2E8B57',  # sea green
        '#B8860B',  # dark goldenrod
        '#CC7000',  # dark orange
        '#8B4513',  # saddle brown
        '#800080',  # purple
        '#4B0082',  # indigo
        '#8B0000',  # dark red
        '#2F4F4F',  # dark slate gray
        '#483D8B',  # dark slate blue
        '#008B8B',  # dark cyan
    ]
    return random.choice(colors)


def extract_task_goals(data_dir):
    """Extract all task goals from jsonl files"""
    all_goals = []

    for fname in os.listdir(data_dir):
        if fname.endswith('.jsonl'):
            fpath = os.path.join(data_dir, fname)
            with open(fpath, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if 'task_goal' in data and data['task_goal']:
                            all_goals.append(data['task_goal'].lower())
                    except:
                        pass

    return all_goals


def extract_object_words(goals):
    """Extract object words from task goals and count frequencies"""

    # Stopwords to exclude
    stopwords = {
        'a', 'an', 'the', 'to', 'into', 'onto', 'on', 'in', 'from', 'with',
        'and', 'or', 'of', 'by', 'at', 'for', 'up', 'down', 'out', 'then',
        'them', 'it', 'its', 'their', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'shall', 'as', 'this', 'that',
        'these', 'those', 'there', 'here', 'where', 'when', 'which', 'who',
        'whom', 'whose', 'why', 'how', 'all', 'each', 'every', 'both', 'few',
        'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
        'own', 'same', 'so', 'than', 'too', 'very', 'can', 'just', 'your',
        'n', 'a', 'edited', 'task', 'goal'
    }

    # Action verbs to exclude
    action_words = {
        'put', 'putting', 'place', 'placing', 'pick', 'picking', 'push', 'pushing',
        'pull', 'pulling', 'open', 'opening', 'close', 'closing', 'stack', 'stacking',
        'arrange', 'arranging', 'move', 'moving', 'take', 'taking', 'slide', 'sliding',
        'press', 'pressing', 'turn', 'turning', 'rotate', 'rotating', 'lift', 'lifting',
        'drop', 'dropping', 'pour', 'pouring', 'clean', 'cleaning', 'wipe', 'wiping',
        'fold', 'folding', 'unfold', 'unfolding', 'insert', 'inserting', 'remove',
        'removing', 'transfer', 'transferring', 'separate', 'separating', 'align',
        'aligning', 'unplug', 'unplugging', 'return', 'returning', 'cover', 'covering',
        'using', 'designated', 'original', 'positions', 'top', 'stacked', 'among',
        'three', 'two', 'one', 'marked', 'position', 'area', 'items', 'intermediary',
        'throwing', 'throw', 'grab', 'grabbing', 'hold', 'holding', 'flip', 'flipping',
        # Additional verbs and verb-like words
        'switch', 'switches', 'switching', 'drink', 'drinking', 'slice', 'slicing',
        'clip', 'clipping', 'plug', 'plugging', 'squeeze', 'squeezing',
        'securing', 'depositing', 'off', 'times', 'ones', 'base'
    }

    # Colors to exclude (not objects)
    color_words = {
        'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'white',
        'black', 'grey', 'gray', 'brown', 'golden', 'silver', 'dark', 'light',
        'colored', 'coloured', 'transparent', 'clear'
    }

    # Other non-object words to exclude
    non_object_words = {
        'small', 'large', 'big', 'medium', 'left', 'right', 'front', 'back',
        'inside', 'outside', 'above', 'below', 'towards', 'together', 'apart',
        'first', 'second', 'third', 'next', 'last', 'another', 'same', 'different',
        'new', 'old', 'empty', 'full', 'half', 'whole', 'single', 'double',
        'crumpled', 'building', 'higher', 'lower', 'piece', 'pieces', 'strip',
        'make', 'scotch', 'duct', 'baking', 'power', 'control', 'motion', 'location',
        'object', 'item', 'thing', 'stuff', 'material',
        # Shape words
        'rectangular', 'cuboid', 'cylinder', 'square', 'round', 'circular',
        # Material words
        'metal', 'plastic', 'wooden', 'glass',
        # Other non-objects
        'middle', 'structure', 'snack', 'desktop', 'hole', 'syrup', 'water',
        'table'
    }

    # Count all words
    all_words = []
    for goal in goals:
        words = re.findall(r'\b[a-zA-Z]+\b', goal)
        all_words.extend(words)

    word_counts = Counter(all_words)

    # Filter to object words only
    object_words = {}
    for word, count in word_counts.items():
        if (word not in stopwords and
            word not in action_words and
            word not in color_words and
            word not in non_object_words and
            len(word) > 2 and
            count >= 10):  # minimum frequency threshold
            object_words[word] = count

    return object_words


def generate_wordcloud(word_freq, output_dir):
    """Generate and save word cloud"""

    # Create word cloud with Times New Roman font and denser layout
    wc = WordCloud(
        width=1000,
        height=700,
        background_color='white',
        max_words=200,
        max_font_size=150,
        min_font_size=10,
        random_state=42,
        prefer_horizontal=0.9,
        relative_scaling=0.45,
        color_func=custom_color_func,
        collocations=False,
        font_path='/home/zux8535/.local/lib/python3.13/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSerif.ttf',
        margin=2,  # balanced margin
        include_numbers=False
    )

    # Generate from frequencies
    wc.generate_from_frequencies(word_freq)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    plt.tight_layout(pad=0)

    # Save as PNG
    png_path = os.path.join(output_dir, 'task_wordcloud.png')
    plt.savefig(png_path, format='png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Saved PNG: {png_path}")

    # Save as PDF
    pdf_path = os.path.join(output_dir, 'task_wordcloud.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Saved PDF: {pdf_path}")

    plt.close()

    return png_path, pdf_path


def main():
    # Paths
    data_dir = "/projects/p32958/chengxuan/ProgressLM/data/benchmark/tiny-bench"
    output_dir = "/projects/p32958/chengxuan/ProgressLM/eval/eval_results/cloud"

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    print("Extracting task goals...")
    goals = extract_task_goals(data_dir)
    print(f"Found {len(goals)} task goals ({len(set(goals))} unique)")

    print("\nExtracting object words...")
    word_freq = extract_object_words(goals)
    print(f"Found {len(word_freq)} object words")

    # Print top 20 words
    print("\nTop 20 words:")
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]
    for word, count in sorted_words:
        print(f"  {word}: {count}")

    print("\nGenerating word cloud...")
    png_path, pdf_path = generate_wordcloud(word_freq, output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
