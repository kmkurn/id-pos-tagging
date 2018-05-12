from contextlib import contextmanager
from typing import List

from torch.autograd import Variable as Var
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchcrf import CRF
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class ContextWindow(nn.Module):
    def __init__(self, window: int) -> None:
        super(ContextWindow, self).__init__()
        self.window = window

    def forward(self, inputs: Var) -> Var:
        assert inputs.dim() == 3

        if self.window == 0:
            return inputs

        batch_size, seq_length = inputs.size(0), inputs.size(1)

        # shape: (batch_size, seq_length + window, size)
        inputs = self._pad(inputs, pad_first=True)
        # shape: (batch_size, seq_length + 2*window, size)
        inputs = self._pad(inputs, pad_first=False)

        temps = []
        for i in range(seq_length):
            lo, hi = i, i + 2 * self.window + 1
            # shape: (batch_size, (2*window + 1) * size)
            temp = inputs[:, lo:hi, :].contiguous().view(batch_size, -1)
            temps.append(temp)
        # shape: (batch_size, seq_length, (2*window + 1) * size)
        outputs = torch.stack(temps, dim=1)

        return outputs

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(window={self.window})'

    def _pad(self, x: Var, pad_first: bool = True) -> Var:
        assert x.dim() == 3

        padding = self._get_padding_for(x)
        inputs = (padding, x) if pad_first else (x, padding)
        # shape: (batch_size, seq_length + window, size)
        res = torch.cat(inputs, dim=1)

        return res

    def _get_padding_for(self, x: Var) -> Var:
        assert x.dim() >= 3
        return Var(x.data.new(x.size(0), self.window, x.size(2)).zero_())


class Concatenate(nn.Module):
    def __init__(self, modules: nn.ModuleList) -> None:
        super(Concatenate, self).__init__()
        self.__modules = modules

    def forward(self, inputs: Var) -> Var:
        assert inputs.size(-1) == len(self.__modules)

        res = [m(inputs[..., i]) for i, m in enumerate(self.__modules)]
        # shape: (*, size)
        outputs = torch.cat(res, dim=-1)

        return outputs


class BiLSTMEmbedder(nn.Module):
    def __init__(
            self,
            embedder: nn.Module,
            embedder_size: int,
            hidden_size: int,
            padding_idx: int = 0,
    ) -> None:
        super(BiLSTMEmbedder, self).__init__()
        self.embedder = embedder
        self.lstm = nn.LSTM(
            embedder_size, hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        self.padding_idx = padding_idx
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if hasattr(self.embedder, 'reset_parameters'):
            self.embedder.reset_parameters()
        self.lstm.reset_parameters()

    def forward(self, inputs: Var) -> Var:
        assert inputs.dim() == 3

        # shape: (batch_size, seq_length, embedder_size)
        embedded = self.embedder(inputs)
        # shape: (batch_size, seq_length)
        mask = torch.sum((inputs != self.padding_idx).long(), dim=-1) != 0
        # shape: (batch_size,)
        seq_lengths = torch.sum(mask.long(), dim=1)
        seq_lengths, sent_perm = seq_lengths.sort(0, descending=True)
        # shape: (batch_size, seq_length, embedder_size), sorted by actual seq length
        embedded = embedded[sent_perm]

        packed_input = pack_padded_sequence(
            embedded, seq_lengths.data.cpu().numpy(), batch_first=True)
        lstm_out, _ = self.lstm(packed_input)
        # shape: (batch_size, seq_length, 2 * hidden_size)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        _, original_perm = sent_perm.sort(0, descending=False)
        # shape: (batch_size, seq_length, 2 * hidden_size), original order
        outputs = lstm_out[original_perm]

        return outputs


class EmissionScorer(nn.Module):
    def __init__(
            self,
            num_tags: int,
            embedder: nn.Module,
            embedder_size: int,
            hidden_size: int = 100,
            dropout: float = 0.5,
    ) -> None:
        super(EmissionScorer, self).__init__()
        self.embedder = embedder
        self.ff = nn.Sequential(
            nn.Linear(embedder_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_tags),
        )
        self.reset_parameters()

    @property
    def num_tags(self) -> int:
        return self.ff[-1].out_features

    def reset_parameters(self) -> None:
        if hasattr(self.embedder, 'reset_parameters'):
            self.embedder.reset_parameters()
        init.xavier_uniform(self.ff[0].weight, gain=init.calculate_gain('tanh'))
        init.constant(self.ff[0].bias, 0.)
        init.xavier_uniform(self.ff[-1].weight, gain=init.calculate_gain('linear'))
        init.constant(self.ff[-1].bias, 0.)

    def forward(self, inputs: Var) -> Var:
        assert inputs.dim() >= 2

        # shape: (batch_size, seq_length, size)
        embedded = self.embedder(inputs)
        # shape: (batch_size, seq_length, num_tags)
        emissions = self.ff(embedded)

        return emissions


class GreedyTagger(nn.Module):
    def __init__(self, scorer: EmissionScorer, padding_idx: int = 0) -> None:
        super(GreedyTagger, self).__init__()
        self.scorer = scorer
        self.padding_idx = padding_idx
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.scorer.reset_parameters()

    def forward(self, inputs: Var, tags: Var) -> Var:
        assert inputs.dim() == 3
        assert inputs.size()[:2] == tags.size()

        batch_size = tags.size(0)

        # shape: (batch_size, seq_length, num_tags)
        emissions = self.scorer(inputs)
        # shape: (batch_size * seq_length, num_tags)
        emissions = emissions.view(-1, emissions.size(-1))
        # shape: (batch_size * seq_length,)
        tags = tags.view(-1)
        # shape: (batch_size * seq_length,)
        loss = F.cross_entropy(emissions, tags, reduce=False)
        # shape: (batch_size, seq_length)
        loss = loss.view(batch_size, -1)
        # shape: (batch_size, seq_length)
        mask = torch.sum((inputs != self.padding_idx).long(), dim=-1) != 0
        # shape: (batch_size,)
        return torch.sum(loss * mask.float(), dim=-1)

    def decode(self, inputs: Var) -> List[List[int]]:
        assert inputs.dim() == 3

        with evaluation(self):
            # shape: (batch_size, seq_length, num_tags)
            emissions = self.scorer(inputs)

        # shape: (batch_size, seq_length)
        _, best_tags = emissions.max(dim=-1)
        # shape: (batch_size, seq_length)
        mask = torch.sum((inputs != self.padding_idx).long(), dim=-1) != 0

        best_tags, mask = best_tags.data, mask.data
        result = []
        for best, mask_ in zip(best_tags, mask):
            length = mask_.long().sum()
            result.append(best[:length].tolist())
        return result


class CRFTagger(nn.Module):
    def __init__(self, scorer: EmissionScorer, padding_idx: int = 0) -> None:
        super(CRFTagger, self).__init__()
        self.scorer = scorer
        self.padding_idx = padding_idx
        self.crf = CRF(scorer.num_tags)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.scorer.reset_parameters()
        self.crf.reset_parameters()

    def forward(self, inputs: Var, tags: Var) -> Var:
        assert inputs.dim() == 3
        assert inputs.size()[:2] == tags.size()

        # shape: (seq_length, batch_size)
        tags = tags.transpose(0, 1).contiguous()
        # shape: (batch_size, seq_length)
        mask = torch.sum((inputs != self.padding_idx).long(), dim=-1) != 0
        # shape: (seq_length, batch_size)
        mask = mask.transpose(0, 1).contiguous()

        # shape: (batch_size, seq_length, num_tags)
        emissions = self.scorer(inputs)
        # shape: (seq_length, batch_size, num_tags)
        emissions = emissions.transpose(0, 1).contiguous()
        # shape: (batch_size,)
        loss = -self.crf(emissions, tags, mask=mask, reduce=False)

        return loss

    def decode(self, inputs: Var) -> List[List[int]]:
        assert inputs.dim() == 3

        # shape: (batch_size, seq_length)
        mask = torch.sum((inputs != self.padding_idx).long(), dim=-1) != 0
        # shape: (seq_length, batch_size)
        mask = mask.transpose(0, 1).contiguous()

        with evaluation(self):
            # shape: (batch_size, seq_length, num_tags)
            emissions = self.scorer(inputs)

        # shape: (seq_length, batch_size, num_tags)
        emissions = emissions.transpose(0, 1).contiguous()

        return self.crf.decode(emissions, mask=mask)


@contextmanager
def evaluation(model):
    training = model.training
    model.eval()
    yield
    model.train(training)
