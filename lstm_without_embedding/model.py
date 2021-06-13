# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                      imports
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import torch.nn as nn
import torch.nn.init as init  # initializers
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
import torch
from rich.console import Console
from pathlib import Path
# CUDA_FLAG = torch.cuda.is_available()
CUDA_FLAG = False


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                      define model
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class lstm_without_embedding_model(nn.Module):
    """defines the `lstm model` with `embedding`


    Attributes:
    -----------
    args : argparse.ArgumentParser()
        contains all the hyper-parameters user has passed
    embed : nn.Embedding
        the embedding layer, find its parameters from `args` attribute
    lstm : nn.LSTM
        `lstm` layer, find its parameters from `args` attribute
    initialize : function
        it returns the initializers `h_0` and `c_0` to initialize the LSTM model
    relu : nn.ReLU
        relu activation layer
    softmax : nn.Softmax
        softmax activation layer
    linear1 : nn.Linear
        ist linear layer after LSTM, find its parameters from `args` attribute
    out : nn.Linear
        last linear layer, find its parameters from `args` attribute



    """

    def __init__(self, args):
        """ get all hyper-parameters as argparse object

        :param args: contains all the hyper parameters of the model passed in as command line args
        :type args: argparse.ArgumentParser()

        """
        super(lstm_without_embedding_model, self).__init__()
        self.args = args

        if self.args.max_norm != "None":
            # Console().print(f"MAX_NORM ---> {self.args.max_norm}")
            self.embed = nn.Embedding(num_embeddings=self.args.vocab,
                                      embedding_dim=self.args.embed_dim,
                                      max_norm=self.args.max_norm,
                                      scale_grad_by_freq=False)
        else:
            self.embed = nn.Embedding(num_embeddings=self.args.vocab,
                                      embedding_dim=self.args.embed_dim,
                                      max_norm=None,
                                      scale_grad_by_freq=True)

        self.lstm = nn.LSTM(input_size=self.args.embed_dim,
                            hidden_size=self.args.lstm_hidden,
                            num_layers=self.args.num_layers,
                            dropout=self.args.dropout,
                            batch_first=True)

        # call the h,C initialization function to get initial tensors for `h` and `c`
        # containes tuple(h_0, c_0)

        self.relu = nn.ReLU(inplace=False)
        self.softmax = nn.Softmax(dim=-1)
        self.linear1 = nn.Linear(in_features=self.args.lstm_hidden,
                                 out_features=self.args.hidden_1)
        self.out = nn.Linear(in_features=self.args.hidden_1,
                             out_features=self.args.classes)

    def last_timestep(self, unpacked, lengths):
        # Index of the last output for each sequence.
        idx = (lengths - 1).view(-1, 1).expand(unpacked.size(0),
                                               unpacked.size(2)).unsqueeze(1)
        return unpacked.gather(1, idx).squeeze()

    def init_hidden(self):
        """Initialize the weights"""

        # num_layers * num_directions, batch, hidden_size)
        # num_layers * num_directions, batch, hidden_size
        if CUDA_FLAG:
            return (
                torch.zeros(
                    size=(self.args.num_layers * self.args.num_directions, self.args.batch, self.args.lstm_hidden),
                    requires_grad=True).cuda(),
                torch.zeros(
                    size=(self.args.num_layers * self.args.num_directions, self.args.batch, self.args.lstm_hidden),
                    requires_grad=True).cuda()
            )
        else:
            # print(self.args.num_layers)
            # print(self.args.num_directions)
            return (

                torch.zeros(size=(
                    self.args.num_layers * self.args.num_directions, self.args.batch, self.args.lstm_hidden),
                    requires_grad=True),
                torch.zeros(
                    size=(self.args.num_layers * self.args.num_directions, self.args.batch, self.args.lstm_hidden),
                    requires_grad=True)
            )

    def forward(self, inp):
        """applies forward propagation loop

        :param inp: shape should be [N, D] where N is mini-batch size and D is seq-lenght
        :type inp: torch.Tensor
        :param seq_len: shape should be [N,] where N is mini-batch size
        :type seq_len: torch.Tensor

        """
        # print(inp)
        # stateless lstm development
        self.hidden = self.init_hidden()
        # out = self.embed(inp)
        # embedding layer cannot be used
        # input ==> [N, dim] ---> [N, seq, dim]
        out = torch.unsqueeze(inp, dim=1)  # [N, 1, dim]

        # out = pack_padded_sequence(out, seq_len, batch_first=True, enforce_sorted=False)
        # out, self.hidden = self.lstm(out, self.hidden)
        out, self.hidden = self.lstm(out)
        # self.hidden[0] = self.hidden[0].cuda()
        # self.hidden[1] = self.hidden[1].cuda()
        # out, __ = pad_packed_sequence(sequence=out,
        #                               batch_first=True,
        #                               )
        # out = self.last_timestep(out, seq_len) # [batch size, seq_len, dims] ==> [batch size, dims]
        out = out[:, -1, :]
        # Console().print(f"shape after last_time_steps ----> {out.shape}")
        # out[:,-1,:] i.e. last time step contains many zeros after being passed through pad_packed_sequences
        # Solution: Use the cell-state or hidden-state
        # Console().print(f"out shape ==> {out.shape}")
        # Console().print(f"H shape ==> {self.hidden[0].shape}")
        # Console().print(f"C shape ==> {self.hidden[1].shape}")
        #
        # Console().print(f"last time step of [red]OUT[/red] {out[:,-1,:]}")
        # Console().print(f"last hidden [red]H[/red] {self.hidden[0][-1]}")
        # Console().print(f"last cell-state [red]C[/red] {self.hidden[1][-1]}")
        #
        # Choosing 'H' and leaving 'C'
        # print(f"%%%%%%%%%%%% {_[0].shape} %%%%%%%%%%")
        # out = self.hidden[0][-1]  # [num_layers * num_directions, batch, hidden_size]

        # out = out[:,-1,:]

        # out shape --> [batch, hidden_size]
        # Console().print(f"[cyan]last H state[/cyan] [color(120)]{out.shape}[/color(120)]" )
        out = self.linear1(out)
        out = self.relu(out)
        out = self.out(out)
        out = self.softmax(out)
        # Console().print(f"[cyan]after softmax[/cyan] [color(120)]{out.shape}[/color(120)]" )

        return out




def save_ckpt(state, is_best):
    """
    saves the model checkpoints to `/command2code/lstm_without_embedding/saved_models/`

    Parameters
    ----------
    state : dictionary
        "epoch", "state_dict", "optimizer" are keys within state
    is_best : bool
        whether to save the model best checkpoint or not

    Returns
    -------
    None
    """
    Path("./lstm_without_embedding/saved_models").mkdir(exist_ok=True, parents=True)
    checkpoint_dir = Path("./lstm_without_embedding/saved_models")
    if not is_best:
        f_path = checkpoint_dir / f'checkpoint_epoch_{state["epoch"]}.pt'
        torch.save(state, f_path)
    if is_best:
        best_fpath = checkpoint_dir / 'best_model.pt'
        torch.save(state, best_fpath)
