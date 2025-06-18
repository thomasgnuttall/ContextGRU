import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as packer, pad_packed_sequence as padder

def compute_encoder_gradients(model, batch):
    model.eval()
    model.zero_grad()
    
    x_prec, x_curr, x_succ, len_prec, len_curr, len_succ, _, _ = batch

    
    # Run the embed method and retain gradients
    x_curr_packed = packer(x_curr, len_curr.to('cpu'), batch_first=True, enforce_sorted=False)
    output_curr, hidden_curr = model.curr_encoder(x_curr_packed)
    output_curr_padded, _ = padder(output_curr, batch_first=True)
    attn_curr = model.attention(output_curr_padded, hidden_curr[-1:])
    attn_curr.requires_grad_(True)
    attn_curr.retain_grad()

    if model.with_context:
        x_prec_packed = packer(x_prec, len_prec.to('cpu'), batch_first=True, enforce_sorted=False)
        output_prec, hidden_prec = model.prec_encoder(x_prec_packed)
        output_prec_padded, _ = padder(output_prec, batch_first=True)
        attn_prec = model.attention(output_prec_padded, hidden_curr[-1:])
        attn_prec.requires_grad_(True)
        attn_prec.retain_grad()

        x_succ_packed = packer(x_succ, len_succ.to('cpu'), batch_first=True, enforce_sorted=False)
        output_succ, hidden_succ = model.succ_encoder(x_succ_packed)
        output_succ_padded, _ = padder(output_succ, batch_first=True)
        attn_succ = model.attention(output_succ_padded, hidden_curr[-1:])
        attn_succ.requires_grad_(True)
        attn_succ.retain_grad()

        attn_output = torch.cat([attn_prec, attn_curr, attn_succ], dim=-1)
    else:
        attn_output = attn_curr

    # Forward through classifier
    logits = model.classifier(attn_output)
    preds = logits.argmax(dim=1)

    # One-sample at a time: compute gradient of the predicted class score
    grad_scores = []
    for i in range(logits.size(0)):
        model.zero_grad()
        score = logits[i, preds[i]]
        score.backward(retain_graph=True)

        if model.with_context:
            g1 = attn_prec.grad[i].norm().item()
            g2 = attn_curr.grad[i].norm().item()
            g3 = attn_succ.grad[i].norm().item()
            grad_scores.append((g1, g2, g3))

            # Zero grads manually
            attn_prec.grad.zero_()
            attn_curr.grad.zero_()
            attn_succ.grad.zero_()
        else:
            g2 = attn_curr.grad[i].norm().item()
            grad_scores.append((g2,))

            attn_curr.grad.zero_()

    # Average importance across batch
    grad_scores_tensor = torch.tensor(grad_scores)
    avg_importance = grad_scores_tensor.mean(dim=0)

    if model.with_context:
        print(f"Average gradient norm (importance):\n  Prec: {avg_importance[0]:.4f}, Curr: {avg_importance[1]:.4f}, Succ: {avg_importance[2]:.4f}")
    else:
        print(f"Average gradient norm (importance):\n  Curr: {avg_importance[0]:.4f}")

    return avg_importance, grad_scores_tensor
