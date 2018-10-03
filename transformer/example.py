import numpy as np

from transformer import transformer, get_transformer_loss, transformer_predict
from test_utils import printoptions
from model_params import TransformerBaseParams


if __name__ == '__main__':
    np.random.seed(1)

    params = TransformerBaseParams()

    num_examples = 100
    x_len = 10
    y_len = 10

    def set_input_padding(raw):
        num_examples = len(raw)
        num_timesteps = len(raw[0])
        assert num_timesteps > 3
        for i in range(num_examples):
            num_non_padding = np.random.randint(3, num_timesteps + 1)
            raw[i][num_non_padding - 1] = 1
            for j in range(num_non_padding, num_timesteps):
                raw[i][j] = 0

    x = np.random.randint(2, params.vocab_size, size=(num_examples, x_len))
    y = np.random.randint(2, params.vocab_size, size=(num_examples, y_len))

    set_input_padding(x)
    set_input_padding(y)

    transformer = transformer(params)

    transformer.summary()

    transformer.compile(optimizer="Adam", loss=get_transformer_loss(params))

    transformer.fit(x=[x, y], y=y, epochs=1, batch_size=256)

    pred = transformer_predict(transformer, params, x[0:10])

    print("pred result (first 100):")
    with printoptions(threshold=3000):
        print(pred)
