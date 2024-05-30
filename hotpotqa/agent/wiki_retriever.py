"""
input: model: retrieval model; sentences_a: search entity; data: database
return: index and sentence
"""
def get_similar_content(model, title_a, data):
    sentences_b = [' '.join(d) for d in data['context']['sentences']]
    title_b = data['context']['title']
    similarities = model.similarity(title_a, title_b)
    score = []
    for i in range(len(title_b)):
        score.append((similarities[i], i))

    score = sorted(score, key=lambda x: x[0], reverse=True)
    print(score)
    return score, sentences_b


def get_similar_content_sentence(model, sentences_a, data):
    sentences_b = [' '.join(d) for d in data['context']['sentences']]
    similarities = model.similarity(sentences_a, sentences_b)
    score = []
    for i in range(len(sentences_b)):
        score.append((similarities[i], i))

    score = sorted(score, key=lambda x: x[0], reverse=True)
    print(score)
    return score, sentences_b
