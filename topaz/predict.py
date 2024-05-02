from __future__ import absolute_import, print_function, division

import torch
try:
    import intel_extension_for_pytorch as ipex
except:
    pass

def batches(X, batch_size=1):
    batch = []
    for x in X:
        batch.append(torch.from_numpy(x).float())
        if len(batch) >= batch_size:
            batch = torch.stack(batch, 0)
            yield batch
            batch = []
    if len(batch) > 0:
        batch = torch.stack(batch, 0)
        yield batch


def score_stream(model, images, device='cpu', batch_size=1):
    with torch.no_grad():
        for x in batches(images, batch_size=batch_size):
            x = x.unsqueeze(1)
            x = x.to(device)
            logits = model(x).squeeze(1).cpu().numpy()
            for i in range(len(logits)):
                yield logits[i]


def score(model, images, device='cpu', batch_size=1):
    scores = []
    for y in score_stream(model, images, device=device, batch_size=batch_size):
        scores.append(y)
    return scores






