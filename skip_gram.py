import utils


def get_data():
    with open('data/text8') as f:
        return f.read()


if __name__ == '__main__':
    text = get_data()
    words = utils.preprocess(text)
    print(words[:100])
