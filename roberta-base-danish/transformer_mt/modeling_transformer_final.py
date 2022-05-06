import os
import json
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_mt.modeling_attention import MultiHeadAttention
from transformer_mt.utils import pad
from transformers import AutoTokenizer, AutoModelForMaskedLM



Hypothesis = namedtuple("Hypothesis", ["value", "score"])

class TransformerDecoderLayer(nn.Module):
    def __init__(self, hidden, num_heads, fcn_hidden, dropout=0.0):
        super().__init__()


        self.self_attention = MultiHeadAttention(
            input_size=hidden,
            hidden=hidden,
            num_heads=num_heads,
            causal=True,
        )
        
        self.cross_attention = MultiHeadAttention(
            input_size=hidden,
            hidden=hidden,
            num_heads=num_heads,
            causal=False,
        )

        self.self_att_layer_norm = nn.LayerNorm(hidden)
        self.cross_att_layer_norm = nn.LayerNorm(hidden)

        self.fcn = nn.Sequential(
            nn.Linear(hidden, fcn_hidden),
            nn.ReLU(),
            nn.Linear(fcn_hidden, hidden),
        )
        self.fcn_layer_norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        
        # YOUR CODE ENDS HERE 
    
    def forward(self, decoder_hidden_states, encoder_hidden_states, key_padding_mask=None):

        residual_1 = decoder_hidden_states
        out = self.self_attention(decoder_hidden_states, key_padding_mask=None)
        out = self.self_att_layer_norm(residual_1 + out)
        residual_2 = out
        out = self.cross_attention(q = out, kv = encoder_hidden_states, key_padding_mask = key_padding_mask)

        out = self.cross_att_layer_norm(out+residual_2)
        out = self.fcn(out)
        out = self.dropout(out)
        residual_3 = out
        out = self.fcn_layer_norm(out+residual_3)

        return out


class TransfomerEncoderDecoderModel(nn.Module):
    def __init__(
        self,
        *,
        num_layers,
        hidden,
        num_heads,
        fcn_hidden,
        max_seq_len,
        src_vocab_size,
        tgt_vocab_size,
        dropout=0.1,
    ):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.num_layers = num_layers
        self.hidden = hidden
        self.num_heads = num_heads
        self.fcn_hidden = fcn_hidden
        self.dropout_rate = dropout
        self.max_seq_len = max_seq_len

        self.decoder_embeddings = nn.Embedding(self.tgt_vocab_size, self.hidden)
        self.positional_emb = nn.Embedding(self.max_seq_len, self.hidden)

        self.out_proj = nn.Linear(self.hidden, self.tgt_vocab_size)
        
        self.dropout = nn.Dropout(self.dropout_rate)
        
#         self.encoder = encoder_model
        self.encoder = AutoModelForMaskedLM.from_pretrained("flax-community/roberta-base-danish", output_hidden_states=True)
        
        self.decoder_layers = nn.ModuleList([TransformerDecoderLayer(hidden = self.hidden,
                                                                    num_heads = self.num_heads,
                                                                    fcn_hidden = self.fcn_hidden,
                                                                    dropout=self.dropout_rate
                                                                    )   
                                             for _ in range(self.num_layers)
                                            ])
        
        # YOUR CODE ENDS HERE

    def _add_positions(self, sequence_tensor):
    
        seq_len = sequence_tensor.shape[1]
        positions = torch.arange(seq_len, device=sequence_tensor.device)
        positional_emb = self.positional_emb(positions)
        output = sequence_tensor + positional_emb
        return output

    def forward(
        self,
        input_ids=None,
        encoder_hidden_states=None,
        decoder_input_ids=None,
        key_padding_mask=None,
    ):
        
        if input_ids is None and encoder_hidden_states is None:
            raise ValueError("You should provide either input_ids or encoder_hidden_states")

        if encoder_hidden_states is None:
            encoder_hidden_states = self.encoder(input_ids, output_hidden_states=True)
            encoder_hidden_states = encoder_hidden_states.hidden_states[-1]
#             print( encoder_hidden_states.shape)

        logits = self._decode(encoder_hidden_states, decoder_input_ids, key_padding_mask)
#         print(logits.shape)


        return logits

    def _decode(self, encoder_hidden_states, decoder_input_ids, key_padding_mask):

        decoder_embedding =  self.decoder_embeddings(decoder_input_ids)
        decoder_embedding = self._add_positions(decoder_embedding)

        for l in self.decoder_layers:
            decoder_embedding = l(decoder_hidden_states = decoder_embedding, encoder_hidden_states=encoder_hidden_states, key_padding_mask = key_padding_mask)

        logits = self.out_proj(decoder_embedding)
        ## YOUR CODE ENDS HERE
        return logits


    @torch.inference_mode()
    def generate(
        self,
        input_ids,
        *,
        bos_token_id,
        eos_token_id,
        pad_token_id=None,
        key_padding_mask=None,
        max_length=50,
        beam_size=5,
        kind="beam_search",
    ):

        if kind not in ["greedy", "beam_search"]:
            raise ValueError("Unknown kind of generation: {}".format(kind))
        if kind == "beam_search" and pad_token_id is None:
            raise ValueError("Beam search requires a pad_token_id to be provided")

        if kind == "greedy":
            return self._generate_greedy(
                input_ids=input_ids,
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                key_padding_mask=key_padding_mask,
                max_length=max_length,
            )
        
        # beam search only supports batch size 1
        beam_search_generations = []
        for i in range(input_ids.size(0)):
            _input_ids = input_ids[i].unsqueeze(0)
            _key_padding_mask = key_padding_mask[i].unsqueeze(0) if key_padding_mask is not None else None

            generated = self._generate_beam_search(
                input_ids=_input_ids,
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                key_padding_mask=_key_padding_mask,
                max_length=max_length,
                beam_size=beam_size,
            )

            beam_search_generations.append(generated[0].detach().cpu().tolist())
        
        return pad(beam_search_generations, pad_id=eos_token_id)

    @torch.inference_mode()
    def _generate_greedy(
        self,
        input_ids,
        *,
        bos_token_id,
        eos_token_id,
        key_padding_mask=None,
        max_length=50,
    ):

       # encoder_hidden_states = self._encode(input_ids, key_padding_mask)
        encoder_hidden_states = self.encoder(input_ids, output_hidden_states=True, attention_mask=key_padding_mask)
        encoder_hidden_states = encoder_hidden_states.hidden_states[-1]


        decoder_input_ids = torch.full((input_ids.shape[0], 1), bos_token_id, dtype=torch.long, device=input_ids.device)
        translation = torch.zeros((input_ids.shape[0], 0), dtype=torch.long, device=input_ids.device)

        eos_flags = torch.zeros((input_ids.shape[0],), dtype=torch.uint8, device=input_ids.device)

        for _ in range(max_length):
            logits = self._decode(encoder_hidden_states, decoder_input_ids, key_padding_mask)
            logits = logits[:, -1, :]

            next_token_id = torch.argmax(logits, dim=-1)

            decoder_input_ids = torch.cat((decoder_input_ids, next_token_id.unsqueeze(1)), dim=1)
            translation = torch.cat((translation, next_token_id.unsqueeze(1)), dim=1)

            eos_flags |= (next_token_id == eos_token_id)

            if eos_flags.all():
                break

        return translation

    @torch.inference_mode()
    def _generate_beam_search(
        self,
        input_ids,
        *,
        bos_token_id,
        eos_token_id,
        key_padding_mask=None,
        beam_size=5,
        max_length=50,
    ):

        assert len(input_ids) == 1, "Beam search is only supported for a single input sequence"
        #encoder_hidden_states = self._encode(input_ids, key_padding_mask)
        encoder_hidden_states = self.encoder(input_ids, output_hidden_states=True, attention_mask=key_padding_mask)
        encoder_hidden_states = encoder_hidden_states.hidden_states[-1]
        device = input_ids.device

        hypotheses = [[bos_token_id]]
        hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=device)
        completed_hypotheses = []

        for _ in range(max_length):
            if len(completed_hypotheses) >= beam_size:
                break

            hyp_num = len(hypotheses)
            expanded_encoder_hidden_states = encoder_hidden_states.expand(
                hyp_num,
                encoder_hidden_states.size(1),
                encoder_hidden_states.size(2),
            )

            # [batch_size*hyp_num=1*hyp_num, seq_len, hidden]
            hypotheses_tensor = torch.tensor(hypotheses, dtype=torch.int64, device=device)
            logits = self._decode(expanded_encoder_hidden_states, hypotheses_tensor, key_padding_mask)
            logits = logits[:, -1, :]  # [vocab_size]

            log_p_t = F.log_softmax(logits, dim=-1)
            live_hyp_num = beam_size - len(completed_hypotheses)

            # [hyp_num] -> [1, hyp_num] -> [hyp_num, vocab_size] -> [hyp_num * vocab_size]
            new_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t).view(-1)
            # [live_hyp_num], [live_hyp_num]
            # for indices, the values range from 0 to hyp_num * vocab_size
            top_new_hyp_scores, top_new_hyp_pos = torch.topk(new_hyp_scores, k=live_hyp_num)

            # hypotheses ids in hyp_scores tensor [hyp_num,]
            prev_hyp_ids = torch.div(top_new_hyp_pos, self.tgt_vocab_size, rounding_mode='floor')

            # ids of the next words for each hypothesis
            token_ids = top_new_hyp_pos % self.tgt_vocab_size

            new_hypotheses = []
            new_hyp_scores = []

            # iterate live_hyp_num times
            for prev_hyp_id, hyp_token_id, cand_new_hyp_score in zip(prev_hyp_ids, token_ids, top_new_hyp_scores):
                prev_hyp_id = prev_hyp_id.item()
                hyp_token_id = hyp_token_id.item()
                cand_new_hyp_score = cand_new_hyp_score.item()

                new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_token_id]
                if hyp_token_id == eos_token_id:
                    completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1], score=cand_new_hyp_score))
                else:
                    new_hypotheses.append(new_hyp_sent)
                    new_hyp_scores.append(cand_new_hyp_score)

            if len(completed_hypotheses) == beam_size:
                break

            hypotheses = new_hypotheses
            hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=device)

        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:], score=hyp_scores[0].item()))
        
        completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)
        return torch.LongTensor(completed_hypotheses[0].value).unsqueeze(0)

    def save_pretrained(self, save_path):
        """Save the model weights to a directory

        Args:
            save_path: directory to save the model
        """
        config = {
            "num_layers": self.num_layers,
            "hidden": self.hidden,
            "num_heads": self.num_heads,
            "fcn_hidden": self.fcn_hidden,
            "src_vocab_size": self.src_vocab_size,
            "tgt_vocab_size": self.tgt_vocab_size,
            "max_seq_len": self.max_seq_len,
            "dropout": self.dropout_rate,
        }

        with open(os.path.join(save_path, "model_config.json"), "w") as f:
           json.dump(config, f)

        state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(save_path, "model.pt"))
    
    @classmethod
    def from_pretrained(cls, save_path, map_location=None):
        """Load the model weights from a directory

        Args:
            save_path: directory to load the model
        """
        if map_location is None and not torch.cuda.is_available():
            map_location = "cpu"

        with open(os.path.join(save_path, "model_config.json"), "r") as f:
            config = json.load(f)

        model = cls(**config)
        state_dict = torch.load(os.path.join(save_path, "model.pt"), map_location=map_location)
        model.load_state_dict(state_dict)
        return model
