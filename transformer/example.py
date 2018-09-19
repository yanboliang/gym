import numpy as np

from transformer import Transformer
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

    transformer = Transformer(params)

    transformer.compile(optimizer="Adam", loss=transformer.get_loss())

    transformer.fit(x=[x, y], y=y, epochs=1, batch_size=256)

    pred = transformer.predict(x=[x[0:10], None], batch_size=256)

    print("pred result (first 100):")
    with printoptions(threshold=3000):
        print(pred)
