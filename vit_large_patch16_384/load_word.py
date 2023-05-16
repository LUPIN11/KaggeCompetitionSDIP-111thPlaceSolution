word_list = []
with open('text8.txt', 'r') as f:
    for line in f:
        words = line.strip().split()
        word_list.extend(words)
print(len(word_list))
