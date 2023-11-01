from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn


class BertForRelationClassification(nn.Module):
    def __init__(self, model_name, num_labels):
        super(BertForRelationClassification, self).__init__()

        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier = nn.Linear(
            self.bert.config.hidden_size * 2, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, token_type_ids, entity_positions, labels=None):
        # Forward pass through BERT model
        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_states = outputs.last_hidden_state

        predicted_labels = []
        loss = torch.tensor(0.0, requires_grad=True)

        for i, one_sent_entity_indices in enumerate(entity_positions):
            one_sent_predicted_labels = []
            for j, (entity1_pos, entity2_pos) in enumerate(one_sent_entity_indices):
                # Extract hidden states corresponding to entity positions
                entity1_hidden_state = hidden_states[i,
                                                     entity1_pos[0]:entity1_pos[1], :].mean(dim=0)
                entity2_hidden_state = hidden_states[i,
                                                     entity2_pos[0]:entity2_pos[1], :].mean(dim=0)
                # Concatenate and pass through classification head
                concat_hidden = torch.cat(
                    [entity1_hidden_state, entity2_hidden_state], dim=-1)
                logits = self.classifier(concat_hidden)
                probs = torch.sigmoid(logits)
                predicted_label = [1 if prob > 0.5 else 0 for prob in probs]
                one_sent_predicted_labels.append(predicted_label)

                if labels is not None:
                    # Compute loss and add it to the total loss
                    input = logits.unsqueeze(0)
                    target = torch.tensor(
                        labels[i][j], dtype=torch.float32).unsqueeze(0).to(input.device)
                    loss = loss + self.loss_fn(input, target)

            predicted_labels.append(one_sent_predicted_labels)

        outputs = (predicted_labels,) + outputs[2:]
        if labels is not None:
            outputs = (loss,) + outputs

        return outputs
