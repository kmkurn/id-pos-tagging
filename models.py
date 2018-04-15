from torch.autograd import Variable as Var
import torch
import torch.nn as nn
import torch.nn.init as init


class FeedforwardTagger(nn.Module):
    def __init__(self,
                 num_words: int,
                 num_tags: int,
                 word_embedding_size: int = 100,
                 window: int = 2,
                 hidden_size: int = 100,
                 dropout: float = 0.5,
                 padding_idx: int = 0,
                 ) -> None:
        super().__init__()

        self.num_words = num_words
        self.num_tags = num_tags
        self.word_embedding_size = word_embedding_size
        self.window = window
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.padding_idx = padding_idx

        self.word_embedding = nn.Embedding(num_words, word_embedding_size, padding_idx=padding_idx)
        num_ctx_words = 2 * window + 1
        self.ff = nn.Sequential(
            nn.Linear(word_embedding_size * num_ctx_words, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_tags),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.xavier_uniform(self.ff[0].weight, gain=init.calculate_gain('tanh'))
        init.constant(self.ff[0].bias, 0)
        init.xavier_uniform(self.ff[-1].weight)
        init.constant(self.ff[-1].bias, 0)

    def forward(self, words: Var) -> Var:
        if words.dim() != 2:
            raise ValueError('expected words to have dim of 2, got', words.dim())

        batch_size, seq_length = words.size()

        padded = torch.cat((self._get_padding(batch_size), words), 1)  # pad left
        padded = torch.cat((padded, self._get_padding(batch_size)), 1)  # pad right
        # shape: (batch_size, seq_length + 2*window, word_embedding_size)
        embedded = self.word_embedding(padded)

        result = []
        for i in range(seq_length):
            lo, hi = i, i + 2 * self.window + 1
            inputs = embedded[:, lo:hi, :].contiguous()
            # shape: (batch_size, (2*window + 1) * word_embedding_size)
            inputs = inputs.view(batch_size, -1)
            # shape: (batch_size, num_tags)
            outputs = self.ff(inputs)
            result.append(outputs)
        # shape: (batch_size, seq_length, num_tags)
        return torch.stack(result, dim=1)

    def predict(self, words: Var) -> Var:
        training = self.training
        self.eval()
        # shape: (batch_size, seq_length, num_tags)
        outputs = self(words)
        # shape: (batch_size, seq_length)
        _, predictions = outputs.max(dim=-1)
        self.train(training)
        return predictions

    def _get_padding(self, batch_size: int) -> Var:
        return Var(torch.Tensor(batch_size, self.window).fill_(self.padding_idx).long())
