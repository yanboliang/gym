import numpy as np

from keras.utils import to_categorical

from transformer import Transformer
from test_utils import printoptions
from model_params import TransformerBaseParams


if __name__ == '__main__':
    np.random.seed(1)

    params = TransformerBaseParams()

    num_examples = 100
    seq_len_x = 10
    seq_len_y = 10

    def set_input_padding(raw):
        nsize = len(raw)
        rlen = len(raw[0])
        assert rlen > 3
        for i in range(nsize):
            tlen = np.random.randint(3, rlen + 1)
            raw[i][tlen - 1] = 1
            for j in range(tlen, rlen):
                raw[i][j] = 0

    my_input_x_raw = np.random.randint(2, params.vocab_size, size=(num_examples, seq_len_x))
    my_input_y_raw = np.random.randint(2, params.vocab_size, size=(num_examples, seq_len_y))
    set_input_padding(my_input_x_raw)
    set_input_padding(my_input_y_raw)

    transformer = Transformer(params)

    transformer.compile(optimizer="Adam",
                  loss=transformer.get_loss())

    transformer.fit(x=[my_input_x_raw, my_input_y_raw],
                    y=my_input_y_raw,
                    epochs=1,
                    batch_size=256)

    pred = transformer.predict(x=[my_input_x_raw[0:10], None], batch_size=256)

    print("pred result (first 100):")
    with printoptions(threshold=3000):
        print(pred)
